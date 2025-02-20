import asyncio
import json
import logging
import time
import numpy as np
import sounddevice as sd
import webrtcvad
import collections
import aioredis
import threading

from typing import Optional
from transformers import AutoTokenizer
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import os
# For spectrogram alignment
import librosa
from scipy.signal import correlate2d

from dtaidistance import dtw


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------
# CONFIG
# ------------------------------
WHISPER_PROMPT = "<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"
LANGUAGE_CODE = "de"
WHISPER_MODEL_NAME = "whisper"
# TRITON_SERVER_URL = "localhost:8001"
TRITON_SERVER_URL= 'triton_latest:8001'

# REDIS_HOST = "localhost"
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = 6379
TRANSCRIPTION_CHANNEL = "transcriptions"

SAMPLE_RATE = 16000
FRAME_DURATION = 30     # ms
VAD_AGGRESSIVENESS = 1  # 0-3
MAX_AUDIO_DURATION = 3.0
COOLDOWN_PERIOD = 0.2

# Overlap settings
PREV_OVERLAP_SEC = 0.5   # how much from the end of the previous chunk
CURR_OVERLAP_SEC = 1.0   # how much from the start of the current chunk
MATCH_THRESHOLD = 0.7    # 80% correlation threshold

# Phrases to skip in transcriptions
SKIP_PHRASES = ["Vielen Dank.", "Tschüss", "Bis zum nächsten Mal."]

# Redis
redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
    if not triton_client.is_server_ready():
        raise RuntimeError("Triton server not ready.")
except Exception as e:
    logging.error(f"Error creating Triton client: {e}")
    triton_client = None

######################################################################
# TEST
######################################################################
NOISE_FLOOR_DB = -40  # Below this considered noise
MIN_VOICE_FREQ = 85   # Hz (human voice range)
MAX_VOICE_FREQ = 255  # Hz
SILENCE_THRESHOLD = 0.025  # RMS threshold

# Config updates
MIN_CHUNK_LENGTH = 0.3  # 300ms minimum audio to process
DTW_WINDOW = 0.8  # Seconds to consider for alignment
MFCC_CONFIG = {
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 400
}

# Add these config parameters at the top
MIN_VOICE_LENGTH = 0.5  # Minimum voice duration in seconds
ENERGY_THRESHOLD = 0.02  # Adjust based on your microphone
HARMONIC_THRESHOLD = 0.15  # Lowered for better voice detection

def find_optimal_overlap(prev_chunk: np.ndarray, current_chunk: np.ndarray, sr: int) -> int:
    """Robust DTW alignment with enhanced error handling"""
    # Convert and validate audio chunks
    def prepare(chunk):
        # Ensure we're working with float32 audio
        if chunk.dtype == np.int16:
            chunk = chunk.astype(np.float32) / 32768.0
        return librosa.util.normalize(chunk)
    
    if len(prev_chunk) < sr//10 or len(current_chunk) < sr//10:
        return 0  # Fallback to no alignment
    
    try:
        prev_audio = prepare(prev_chunk[-int(DTW_WINDOW*sr):])
        curr_audio = prepare(current_chunk[:int(DTW_WINDOW*sr)])
        
        # Extract features with error checking
        if len(prev_audio) < MFCC_CONFIG['n_fft'] or len(curr_audio) < MFCC_CONFIG['n_fft']:
            return 0
        
        mfcc_prev = librosa.feature.mfcc(y=prev_audio, sr=sr, **MFCC_CONFIG)
        mfcc_curr = librosa.feature.mfcc(y=curr_audio, sr=sr, **MFCC_CONFIG)
        
        # Handle empty MFCC features
        if mfcc_prev.size == 0 or mfcc_curr.size == 0:
            return 0
            
        # Fast DTW with constrained path
        try:
            distance, paths = dtw.warping_paths(mfcc_prev.T, mfcc_curr.T)
            path = dtw.best_path(paths)
            avg_shift = np.mean([p[1] - p[0] for p in path])
        except Exception as e:
            logging.warning(f"DTW path error: {str(e)}")
            return energy_based_alignment(prev_chunk, current_chunk, sr)
        
        overlap_samples = int(avg_shift * MFCC_CONFIG['hop_length'])
        return np.clip(overlap_samples, 0, len(current_chunk))
        
    except Exception as e:
        logging.warning(f"DTW alignment failed: {str(e)}")
        return energy_based_alignment(prev_chunk, current_chunk, sr)

def is_voice(audio: np.ndarray, sr: int) -> bool:
    """Reliable voice detection with array-safe operations"""
    if len(audio) < 512:
        return False
    
    audio = audio.astype(np.float32) / 32768.0
    
    # Stage 1: Energy check
    rms = np.sqrt(np.mean(np.square(audio)))
    if rms < SILENCE_THRESHOLD:
        return False
    
    # Stage 2: Spectral flatness with array safety
    with np.errstate(divide='ignore', invalid='ignore'):
        spectral_flatness = np.exp(np.nanmean(np.log(np.clip(audio**2, 1e-10, None))))
        spectral_flatness /= np.nanmean(audio**2)
    
    if not np.isfinite(spectral_flatness) or spectral_flatness > 0.85:
        return False
    
    # Stage 3: Harmonic content with minimum duration check
    try:
        if len(audio) >= int(sr * MIN_VOICE_LENGTH):
            harmonic_ratio = np.mean(librosa.effects.harmonic(audio))
            return harmonic_ratio > HARMONIC_THRESHOLD
        return True  # Allow short segments
    except Exception as e:
        logging.warning(f"Harmonic check error: {str(e)}")
        return rms > ENERGY_THRESHOLD

def adaptive_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Safe denoising with length validation"""
    if len(audio) < 2048:
        return audio
    
    try:
        audio_float = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        D = librosa.stft(audio_float, n_fft=512, hop_length=160)
        mag = np.abs(D)
        
        # Dynamic noise floor
        noise_frames = min(5, mag.shape[1])
        noise_profile = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
        
        # Soft masking with noise compensation
        mask = np.clip(mag / (noise_profile + 1e-10), 0, 1)
        mask = np.where(mag > 2*noise_profile, 1.0, mask)
        
        return librosa.istft(mag * mask * np.exp(1j * np.angle(D)), length=len(audio))
    except Exception as e:
        logging.error(f"Denoising failed: {str(e)}")
        return audio

def energy_based_alignment(prev_chunk: np.ndarray, current_chunk: np.ndarray, sr: int) -> int:
    """Fallback alignment using RMS energy correlation"""
    window_size = int(0.2 * sr)  # 200ms
    prev_energy = librosa.feature.rms(y=prev_chunk.astype(np.float32), frame_length=window_size, hop_length=window_size)
    curr_energy = librosa.feature.rms(y=current_chunk.astype(np.float32), frame_length=window_size, hop_length=window_size)
    
    # Find best correlation
    corr = np.correlate(prev_energy.flatten(), curr_energy.flatten(), mode='valid')
    if len(corr) == 0:
        return 0
    best_offset = np.argmax(corr)
    return best_offset * window_size



def denoise_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Safe spectral gating with input validation"""
    if len(audio) < 512:  # Minimum length for STFT
        return audio
    
    try:
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            
        # STFT analysis
        S = librosa.stft(audio, n_fft=512, hop_length=160)
        magnitude, phase = librosa.magphase(S)

        # Noise profile (first 3 frames ~300ms)
        noise_frames = 3
        if magnitude.shape[1] > noise_frames:
            noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
        else:
            noise_profile = np.median(magnitude, axis=1, keepdims=True)

        # Spectral gating with threshold
        magnitude_db = librosa.amplitude_to_db(magnitude)
        noise_threshold = librosa.amplitude_to_db(noise_profile) + NOISE_FLOOR_DB
        
        mask = magnitude_db > noise_threshold
        magnitude_clean = magnitude * mask

        # Reconstruct audio
        return librosa.istft(magnitude_clean * phase, length=len(audio))
    except Exception as e:
        logging.error(f"Denoising failed: {e}")
        return audio


# def find_optimal_overlap(prev_chunk: np.ndarray, current_chunk: np.ndarray, sr: int) -> int:
#     """Find optimal overlap using Dynamic Time Warping"""
#     # Extract features
#     prev_mfcc = librosa.feature.mfcc(y=prev_chunk, sr=sr, n_mfcc=13)
#     curr_mfcc = librosa.feature.mfcc(y=current_chunk, sr=sr, n_mfcc=13)

#     # Calculate DTW distance matrix
#     distance_matrix = dtw.distance_matrix(prev_mfcc.T, curr_mfcc.T)
    
#     # Find optimal path
#     path = dtw.warping_path(distance_matrix)
    
#     # Find maximum overlap point
#     overlap_point = max(0, len(prev_chunk) - np.argmax(path[:,1]))
#     return min(overlap_point, len(current_chunk))

# def find_optimal_overlap(prev_chunk: np.ndarray, current_chunk: np.ndarray, sr: int) -> int:
#     """Find optimal overlap using Dynamic Time Warping with proper audio conversion"""
#     # Convert int16 to float32 and normalize
#     def prepare_audio(chunk):
#         if chunk.dtype == np.int16:
#             chunk = chunk.astype(np.float32) / 32768.0
#         return librosa.util.normalize(chunk)
    
#     # Handle empty chunks
#     if len(prev_chunk) == 0 or len(current_chunk) == 0:
#         return 0

#     try:
#         prev_audio = prepare_audio(prev_chunk)
#         curr_audio = prepare_audio(current_chunk)
        
#         # Extract MFCC features with proper configuration
#         prev_mfcc = librosa.feature.mfcc(y=prev_audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
#         curr_mfcc = librosa.feature.mfcc(y=curr_audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
        
#         # Calculate DTW distance matrix
#         distance_matrix = dtw.distance_matrix_fast(prev_mfcc.T, curr_mfcc.T)
        
#         # Find optimal path
#         path = dtw.warping_path(distance_matrix)
        
#         # Convert MFCC frame offset to audio samples
#         hop_length = 160  # Must match hop_length used in MFCC
#         overlap_samples = path[-1][1] * hop_length
        
#         return min(overlap_samples, len(current_chunk))
#     except Exception as e:
#         logging.error(f"DTW alignment failed: {e}")
#         return 0

######################################################################
# HELPER FUNCTIONS
######################################################################
def compute_mel_spectrogram(audio_float32: np.ndarray, sr: int) -> np.ndarray:
    """
    Convert 1D float32 audio to a log-mel-spectrogram using librosa.
    We'll do a standard n_fft=512, hop_length=128, n_mels=40 approach.
    """
    n_fft = 512
    hop_length = 128
    n_mels = 40
    S = librosa.feature.melspectrogram(
        y=audio_float32,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def sliding_correlation_score(ref_spec: np.ndarray, tgt_spec: np.ndarray) -> float:
    """
    Perform a 2D cross-correlation with mode='valid' so we do a sliding window
    of the smaller array over the larger one. Then find the best (max) correlation
    value and convert it to a normalized [0..1] range.

    This is a rough approach. Real alignment might do dynamic time warping (DTW),
    or a more advanced method. But let's do a simpler correlation.

    Return the best normalized correlation score [0..1].
    """
    # We assume ref_spec and tgt_spec can differ in time dimension.
    # We'll do correlate2d in 'valid' mode, ensuring we only align
    # windows that fully fit.
    corr = correlate2d(ref_spec, tgt_spec, mode='valid')

    # max_corr is the raw correlation peak
    max_corr = np.max(corr)

    # A naive way to normalize is to take sqrt of sum of squares of each
    # spectrogram and multiply them. This is akin to normalizing in cross-correlation.
    # We'll do a quick hack. For 2D:
    #   norm_factor = sqrt( sum(ref^2)* sum(tgt^2) )
    # to get a correlation coefficient in [-1..1].
    ref_power = np.sum(ref_spec * ref_spec)
    tgt_power = np.sum(tgt_spec * tgt_spec)
    if ref_power <= 1e-8 or tgt_power <= 1e-8:
        return 0.0

    norm_factor = np.sqrt(ref_power * tgt_power)
    normalized_corr = max_corr / (norm_factor + 1e-10)

    # Because we're dealing with negative dB values, correlation might be less direct,
    # but let's treat it as an approximate measure in [0..1].
    return float(np.clip(normalized_corr, -1.0, 1.0))

def align_and_trim_chunk(
    prev_overlap_audio: np.ndarray,
    current_chunk: np.ndarray,
    sr: int,
    threshold: float
) -> np.ndarray:
    """
    Compare last PREV_OVERLAP_SEC of `prev_overlap_audio` to the first
    CURR_OVERLAP_SEC of `current_chunk` via a more "loose" sliding correlation.

    If we find a correlation above 'threshold' (e.g. 0.8),
    we assume there's an overlap and remove that portion from the start
    of current_chunk.

    Return only the "fresh" portion of current_chunk (the part that isn't
    duplicated from prev_overlap_audio).
    """

    overlap_len_prev = int(PREV_OVERLAP_SEC * sr)
    if len(prev_overlap_audio) <= overlap_len_prev:
        ref_part = prev_overlap_audio
    else:
        ref_part = prev_overlap_audio[-overlap_len_prev:]  # last 1s

    overlap_len_curr = int(CURR_OVERLAP_SEC * sr)
    if len(current_chunk) <= overlap_len_curr:
        target_part = current_chunk
    else:
        target_part = current_chunk[:overlap_len_curr]  # first 1s

    try:
        overlap_samples = energy_based_alignment(ref_part, target_part, sr)
        return current_chunk[overlap_samples:]
    except Exception as e:
        logging.error(f"Alignment failed: {str(e)}")
        return current_chunk

    # Edge case: too small to do correlation
    # if len(ref_part) < 256 or len(target_part) < 256:
    #     return current_chunk

    # Convert both to float32, compute mel-spec
 #   ref_float = ref_part.astype(np.float32) / 32768.0
 #   tgt_float = target_part.astype(np.float32) / 32768.0

#    ref_spec = compute_mel_spectrogram(ref_float, sr)
 #   tgt_spec = compute_mel_spectrogram(tgt_float, sr)

 #   corr_score = sliding_correlation_score(ref_spec, tgt_spec)
 #   logging.debug(f"[align_and_trim_chunk] Found correlation: {corr_score:.3f}")

    
    optimal_overlap = find_optimal_overlap(ref_part, target_part, sr)
    trimmed_chunk = current_chunk[optimal_overlap:]

    # trimmed_chunk = current_chunk
    
    
    # If we exceeded threshold, we assume partial overlap. We'll compute the
    # shift that gave the best correlation in 'valid' mode.
    # if corr_score >= threshold:
    #     # We'll do a full 2D cross_correlation in 'full' mode to find the best offset
    #     # so we know how many samples to remove. Then we'll only remove if offset < 0
    #     # (meaning chunk2 is duplicating chunk1).
    #     full_corr = correlate2d(ref_spec, tgt_spec, mode='full')
    #     max_y, max_x = np.unravel_index(np.argmax(full_corr), full_corr.shape)
    #     # For 2D correlation in 'full' mode, the center alignment is:
    #     center_x = ref_spec.shape[1] - 1
    #     time_shift = max_x - center_x

    #     hop_length = 128  # from compute_mel_spectrogram
    #     shift_in_samples = time_shift * hop_length

    #     logging.debug(f"[align_and_trim_chunk] Best offset shift_in_samples={shift_in_samples}")

    #     # If shift_in_samples < 0 => chunk2 starts earlier => remove that many samples from chunk2
    #     if shift_in_samples < 0:
    #         remove_count = abs(int(shift_in_samples))
    #         if remove_count < len(current_chunk):
    #             trimmed_chunk = current_chunk[remove_count:]
    #         else:
    #             trimmed_chunk = np.array([], dtype=current_chunk.dtype)

    return trimmed_chunk

def normalize_audio(audio_int16: np.ndarray) -> np.ndarray:
    """
    Convert int16 -> float32 in [-1,1].
    """
    audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max
    mx = np.max(np.abs(audio_float32))
    if mx > 1e-8:
        audio_float32 /= mx
    return np.clip(audio_float32, -1.0, 1.0)

def transcribe_audio(
    audio_data: np.ndarray,
    whisper_prompt: str,
    language: str,
    model_name: str,
    triton_client: grpcclient.InferenceServerClient
) -> Optional[str]:
    """
    Call Triton for Whisper transcription.
    """
    if triton_client is None:
        logging.error("No Triton client.")
        return None
    try:
        if not triton_client.is_model_ready(model_name):
            logging.error(f"Model {model_name} not ready on Triton server.")
            return None

        audio_data = audio_data.astype(np.float32)
        audio_data = np.expand_dims(audio_data, axis=0)  # (1, samples)

        input_wav = grpcclient.InferInput("WAV", audio_data.shape, np_to_triton_dtype(audio_data.dtype))
        input_wav.set_data_from_numpy(audio_data)

        input_text = grpcclient.InferInput("TEXT_PREFIX", [1,1], "BYTES")
        input_text.set_data_from_numpy(np.array([[whisper_prompt.encode()]], dtype=object))

        outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTS")]
        resp = triton_client.infer(
            model_name=model_name,
            inputs=[input_wav, input_text],
            outputs=outputs
        )
        transcription = resp.as_numpy("TRANSCRIPTS")[0]
        if isinstance(transcription, np.ndarray):
            transcription = b" ".join(transcription).decode("utf-8")
        else:
            transcription = transcription.decode("utf-8")

        return transcription
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return None

async def publish_message(message: dict, channel: str):
    """
    Publish JSON to Redis.
    """
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message: {message}")
    except Exception as e:
        logging.error(f"Redis error: {e}")

######################################################################
# MAIN RECORDING/VAD LOGIC
######################################################################
def record_audio_and_publish(
    whisper_prompt: str,
    language: str,
    whisper_model_name: str,
    transcription_channel: str,
    sample_rate: int,
    frame_duration_ms: int,
    max_audio_duration_s: float,
    cooldown_period_s: float,
    loop: asyncio.AbstractEventLoop
):
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    channels = 1
    dtype = 'int16'
    frame_size = int(sample_rate * frame_duration_ms / 1000)

    triggered = False
    buffer_bytes = b''
    buffer_start_time = time.time()

    last_transcription_time = 0.0
    transcribe_lock = threading.Lock()

    # We'll keep the last 1s of float audio from the *previous* chunk
    prev_overlap_audio: Optional[np.ndarray] = None

    async def process_transcription(final_chunk: bytes):
        nonlocal prev_overlap_audio, last_transcription_time
        with transcribe_lock:
            try:
                now = time.time()
                if now - last_transcription_time < cooldown_period_s:
                    return

                audio_int16 = np.frombuffer(final_chunk, dtype=np.int16)
                if len(audio_int16) < 160:  # Minimum 10ms of audio
                    return
                    
                # Convert and denoise
                audio_float = normalize_audio(audio_int16)
                
                # CRITICAL FIX: Bypass filters temporarily for testing
                # audio_float = adaptive_denoise(audio_float, SAMPLE_RATE)
                # if not is_voice(audio_float, SAMPLE_RATE):
                #     return

                # CRITICAL FIX: Direct Triton call with error handling
                try:
                    transcription = transcribe_audio(
                        audio_float, WHISPER_PROMPT, LANGUAGE_CODE, 
                        WHISPER_MODEL_NAME, triton_client
                    )
                except Exception as e:
                    logging.error(f"Triton error: {str(e)}")
                    return
                    
                if transcription:
                    clean_transcription = transcription.strip()
                    logging.info(f"Raw transcription: {clean_transcription}")
                    
                    # CRITICAL FIX: Direct Redis publish with await
                    await redis.publish(
                        TRANSCRIPTION_CHANNEL, 
                        json.dumps({"transcription": clean_transcription})
                    )
                    last_transcription_time = now

                # Update overlap buffer
                overlap_samples = int(PREV_OVERLAP_SEC * SAMPLE_RATE)
                prev_overlap_audio = audio_float[-overlap_samples:] if len(audio_float) > overlap_samples else audio_float.copy()
                
            except Exception as e:
                logging.error(f"Processing error: {str(e)}")

    def finalize_chunk():
        """
        Called when we decide the chunk is done (max duration or end of speech).
        We'll do spectrogram alignment with prev_overlap_audio to remove duplicates,
        transcribe only the "fresh" portion.
        """
        nonlocal buffer_bytes
        chunk_int16 = np.frombuffer(buffer_bytes, dtype=np.int16)

        if prev_overlap_audio is not None and len(prev_overlap_audio) > 0:
            aligned_chunk = align_and_trim_chunk(
                prev_overlap_audio,
                chunk_int16,
                sr=SAMPLE_RATE,
                threshold=MATCH_THRESHOLD
            )
            trimmed_bytes = aligned_chunk.tobytes()
        else:
            # No previous chunk overlap, just use entire chunk
            trimmed_bytes = buffer_bytes

        # Transcribe this chunk in the background
        asyncio.run_coroutine_threadsafe(
            process_transcription(trimmed_bytes),
            loop
        )

        # Reset chunk buffer
        buffer_bytes = b''

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, buffer_bytes, buffer_start_time

        if status:
            logging.warning(f"Audio status: {status}")

        audio_frame = indata.flatten().tobytes()
        is_speech = vad.is_speech(audio_frame, sample_rate)

        # CRITICAL FIX: Add buffer reset mechanism
        if not triggered:
            if is_speech:
                triggered = True
                buffer_bytes = audio_frame
                buffer_start_time = time.time()
                logging.debug("Speech trigger activated")
        else:
            buffer_bytes += audio_frame
            duration = time.time() - buffer_start_time

            # Force finalize after 3 seconds even if still speech
            if duration >= max_audio_duration_s:
                logging.debug("Max duration reached, finalizing")
                triggered = False
                finalize_chunk()
            elif not is_speech:
                # Wait for 3 consecutive non-speech frames
                if not hasattr(audio_callback, "silence_counter"):
                    audio_callback.silence_counter = 0
                audio_callback.silence_counter += 1
                
                if audio_callback.silence_counter >= 3:
                    logging.debug("Silence detected, finalizing")
                    triggered = False
                    audio_callback.silence_counter = 0
                    finalize_chunk()
            else:
                audio_callback.silence_counter = 0

    async def record():
        try:
            with sd.InputStream(
                channels=channels,
                samplerate=sample_rate,
                dtype=dtype,
                blocksize=frame_size,
                callback=audio_callback
            ):
                logging.info("Recording with partial spectrogram overlap alignment & skip logic...")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"InputStream error: {e}")

    asyncio.run_coroutine_threadsafe(record(), loop)

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        record_thread = threading.Thread(
            target=record_audio_and_publish,
            args=(
                WHISPER_PROMPT,
                LANGUAGE_CODE,
                WHISPER_MODEL_NAME,
                TRANSCRIPTION_CHANNEL,
                SAMPLE_RATE,
                FRAME_DURATION,
                MAX_AUDIO_DURATION,
                COOLDOWN_PERIOD,
                loop
            ),
            daemon=True
        )
        record_thread.start()
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        loop.stop()
        record_thread.join()
