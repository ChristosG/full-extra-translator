#!/usr/bin/env python
import asyncio
import json
import logging
import time
import numpy as np
import sounddevice as sd
import webrtcvad
import aioredis
import threading
import os
import librosa
from scipy.signal import lfilter, firwin, correlate2d
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient
from noisereduce import reduce_noise

# ------------------------------------------------------------------------------
# LOGGING & CONFIGURATION
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Whisper/Triton configuration
WHISPER_PROMPT = "<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"
LANGUAGE_CODE = "de"
WHISPER_MODEL_NAME = "whisper"
TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "triton_latest:8001")

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = 6379
TRANSCRIPTION_CHANNEL = "transcriptions"

# Audio capture and VAD configuration
SAMPLE_RATE = 16000
FRAME_DURATION = 30     # in ms
VAD_AGGRESSIVENESS = 1  # (0-3)
MAX_AUDIO_DURATION = 3.0  # seconds per chunk
COOLDOWN_PERIOD = 0.2   # seconds between transcriptions

# Overlap settings (for aligning chunk boundaries)
PREV_OVERLAP_SEC = 0.5   # use last 0.5 s from previous chunk

# Voice/noise thresholds
SILENCE_THRESHOLD = 0.0015  # RMS threshold for silence
MIN_VOICE_LENGTH = 0.3     # seconds
HARMONIC_THRESHOLD = 0.05  # threshold for harmonic content


# ------------------------------------------------------------------------------
# SETUP: REDIS AND TRITON CLIENT
# ------------------------------------------------------------------------------
redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
    if not triton_client.is_server_ready():
        raise RuntimeError("Triton server not ready.")
except Exception as e:
    logging.error(f"Error creating Triton client: {e}")
    triton_client = None

# ------------------------------------------------------------------------------
# AUDIO ENHANCEMENT & ALIGNMENT FUNCTIONS
# ------------------------------------------------------------------------------
def enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Advanced audio enhancement pipeline:
      1. Normalize and remove DC offset.
      2. Reduce spectral noise.
      3. Emphasize speech frequencies (300–4000 Hz) via FIR filtering.
      4. Apply dynamic range compression using STFT manipulation.
      5. Fade in/out to avoid edge artifacts.
    """
    # Normalize to float32 and remove DC offset
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    audio -= np.mean(audio)
    
    # 1. Spectral noise reduction
    clean_audio = reduce_noise(
        y=audio, 
        sr=sr,
        stationary=True,
        n_fft=512,
        hop_length=128,
        n_std_thresh_stationary=1.5,
        prop_decrease=0.95
    )
    
    # 2. Emphasize speech frequencies (300–4000 Hz)
    nyq = 0.5 * sr
    freq_range = [300 / nyq, 4000 / nyq]
    b = firwin(101, freq_range, pass_zero=False)
    filtered = lfilter(b, [1.0], clean_audio)
    
    # 3. Dynamic range compression via STFT manipulation
    stft = librosa.stft(filtered, n_fft=512, hop_length=160)
    magnitude, phase = librosa.magphase(stft)
    db_magnitude = librosa.amplitude_to_db(magnitude)
    db_magnitude = np.clip(db_magnitude, -30, None)
    compressed = librosa.istft(librosa.db_to_amplitude(db_magnitude) * phase,
                                 hop_length=160, length=len(audio))
    
    # 4. Fade in/out to prevent edge artifacts
    fade_samples = 256
    compressed[:fade_samples] *= np.linspace(0, 1, fade_samples)
    compressed[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Normalize the final output
    if np.max(np.abs(compressed)) > 0:
        compressed = compressed / np.max(np.abs(compressed))
    return compressed

def find_exact_overlap(prev_chunk: np.ndarray, current_chunk: np.ndarray, sr: int) -> int:
    """
    Determine the overlap offset (in samples) between the tail of the previous
    chunk and the beginning of the current chunk by combining MFCC and spectral contrast.
    """
    # Ensure float32
    prev_float = prev_chunk.astype(np.float32)
    curr_float = current_chunk.astype(np.float32)
    
    # Compute MFCC features
    mfcc_prev = librosa.feature.mfcc(y=prev_float, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
    mfcc_curr = librosa.feature.mfcc(y=curr_float, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
    
    # Compute spectral contrast features
    sc_prev = librosa.feature.spectral_contrast(y=prev_float, sr=sr, n_fft=512)
    sc_curr = librosa.feature.spectral_contrast(y=curr_float, sr=sr, n_fft=512)
    
    # Combine features along the frequency axis
    features_prev = np.vstack([mfcc_prev, sc_prev])
    features_curr = np.vstack([mfcc_curr, sc_curr])
    
    # Cross-correlation search over a window of ±125 ms (≈ ±2000 samples at hop=160)
    search_range = 2000
    cross_corr = np.zeros(2 * search_range)
    for i in range(-search_range, search_range):
        if i < 0:
            seg_prev = features_prev[:, :i]
            seg_curr = features_curr[:, -i:]
        else:
            seg_prev = features_prev[:, i:]
            seg_curr = features_curr[:, :-i] if i != 0 else features_curr
        if seg_prev.size == 0 or seg_curr.size == 0:
            continue
        norm = (np.linalg.norm(seg_prev) * np.linalg.norm(seg_curr)) + 1e-10
        cross_corr[i + search_range] = np.sum(seg_prev * seg_curr) / norm
    
    peak_idx = np.argmax(cross_corr)
    # Parabolic interpolation for sub-sample accuracy
    idx_range = np.arange(max(0, peak_idx - 2), min(len(cross_corr), peak_idx + 3))
    coeffs = np.polyfit(idx_range, cross_corr[idx_range], 2)
    if coeffs[0] == 0:
        exact_peak = peak_idx
    else:
        exact_peak = -coeffs[1] / (2 * coeffs[0])
    # Convert the offset into samples (adjusting for hop_length)
    return int((exact_peak - search_range) * 160)

def align_and_trim_chunk(prev_overlap: np.ndarray, current_chunk: np.ndarray, sr: int) -> np.ndarray:
    """
    Using a multi-stage approach, determine the exact overlap offset and remove the
    overlapping (duplicate) audio at the start of the current chunk.
    """
    # Stage 1: Get an estimated offset via feature-based cross-correlation
    estimated_offset = find_exact_overlap(prev_overlap, current_chunk, sr)
    
    # Stage 2: Refine using waveform cross-correlation on a 100 ms window
    window_size = int(0.1 * sr)
    if len(prev_overlap) < window_size or len(current_chunk) < window_size:
        final_offset = estimated_offset
    else:
        cc = np.correlate(prev_overlap[-window_size:].astype(np.float32),
                          current_chunk[:window_size].astype(np.float32),
                          mode='valid')
        waveform_offset = np.argmax(cc)
        # Combine both estimates with weighted averaging
        final_offset = int(0.7 * estimated_offset + 0.3 * waveform_offset)
    
    # Ensure the offset is within the current chunk’s bounds
    final_offset = np.clip(final_offset, 0, len(current_chunk) - 1)
    
    # Stage 3: Verify spectral continuity (using a 200 ms window)
    spec_window = int(0.2 * sr)
    if len(prev_overlap) >= spec_window and len(current_chunk) >= final_offset + spec_window:
        prev_spec = np.abs(librosa.stft(prev_overlap[-spec_window:], n_fft=512))
        curr_spec = np.abs(librosa.stft(current_chunk[final_offset:final_offset + spec_window], n_fft=512))
        spectral_diff = np.mean(np.abs(prev_spec - curr_spec))
        if spectral_diff > 0.5:
            # If the discontinuity is too high, assume no overlap was found.
            return current_chunk
    return current_chunk[final_offset:]

def normalize_audio(audio_int16: np.ndarray) -> np.ndarray:
    """
    Convert int16 audio to float32 in the range [-1, 1] and normalize.
    """
    audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max
    max_val = np.max(np.abs(audio_float32))
    if max_val > 1e-8:
        audio_float32 /= max_val
    return np.clip(audio_float32, -1.0, 1.0)

def is_voice(audio: np.ndarray, sr: int) -> bool:
    """
    A simple voice activity detector based on RMS energy and harmonic content.
    """
    if len(audio) < 512:
        return False
    rms = np.sqrt(np.mean(audio**2))
    if rms < SILENCE_THRESHOLD:
        return False
    try:
        if len(audio) >= int(sr * MIN_VOICE_LENGTH):
            harmonic_ratio = np.mean(librosa.effects.harmonic(audio))
            return harmonic_ratio > HARMONIC_THRESHOLD
        return True
    except Exception as e:
        logging.warning(f"Harmonic check error: {e}")
        return rms > SILENCE_THRESHOLD

# ------------------------------------------------------------------------------
# TRANSCRIPTION FUNCTION (Triton/Whisper)
# ------------------------------------------------------------------------------
def transcribe_audio(
    audio_data: np.ndarray,
    whisper_prompt: str,
    language: str,
    model_name: str,
    triton_client: grpcclient.InferenceServerClient
) -> str:
    """
    Call the Triton inference server to transcribe the processed audio using Whisper.
    """
    if triton_client is None:
        logging.error("No Triton client available.")
        return ""
    try:
        if not triton_client.is_model_ready(model_name):
            logging.error(f"Model {model_name} not ready on Triton server.")
            return ""

        # Ensure audio_data is float32 and has shape (1, samples)
        audio_data = audio_data.astype(np.float32)
        audio_data = np.expand_dims(audio_data, axis=0)

        input_wav = grpcclient.InferInput("WAV", audio_data.shape, np_to_triton_dtype(audio_data.dtype))
        input_wav.set_data_from_numpy(audio_data)

        input_text = grpcclient.InferInput("TEXT_PREFIX", [1, 1], "BYTES")
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
        return ""

# ------------------------------------------------------------------------------
# AUDIO RECORDING, VAD, AND CHUNK HANDLING
# ------------------------------------------------------------------------------
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

    # This will hold the last 0.5 s of the previous processed (enhanced) audio,
    # to be used for precise alignment of consecutive chunks.
    prev_overlap_audio = None

    async def process_transcription(audio_int16: np.ndarray):
        nonlocal prev_overlap_audio, last_transcription_time
        with transcribe_lock:
            now = time.time()
            if now - last_transcription_time < cooldown_period_s:
                return
            if len(audio_int16) < 160:  # Skip if audio is too short (e.g. <10ms)
                return

            # Normalize and enhance the audio
            audio_float = normalize_audio(audio_int16)
            # if not is_voice(audio_float, sample_rate):
            #     logging.info("No voice detected in chunk.")
            #     return

            enhanced_audio = enhance_audio(audio_float, sample_rate)

            # Align current chunk with previous overlap (if available) to avoid duplicate words.
            if prev_overlap_audio is not None and len(prev_overlap_audio) > 0:
                aligned_audio = align_and_trim_chunk(prev_overlap_audio, enhanced_audio, sample_rate)
            else:
                aligned_audio = enhanced_audio

            # Update the overlap buffer (use last PREV_OVERLAP_SEC seconds)
            overlap_samples = int(PREV_OVERLAP_SEC * sample_rate)
            prev_overlap_audio = enhanced_audio[-overlap_samples:] if len(enhanced_audio) >= overlap_samples else enhanced_audio.copy()

            transcription = transcribe_audio(aligned_audio, whisper_prompt, language, whisper_model_name, triton_client)
            if transcription:
                clean_transcription = transcription.strip()
                logging.info(f"Raw transcription: {clean_transcription}")
                await redis.publish(transcription_channel, json.dumps({"transcription": clean_transcription}))
                last_transcription_time = now

    def finalize_chunk():
        nonlocal buffer_bytes
        if len(buffer_bytes) == 0:
            return
        audio_int16 = np.frombuffer(buffer_bytes, dtype=np.int16)
        # Schedule asynchronous transcription processing
        asyncio.run_coroutine_threadsafe(process_transcription(audio_int16), loop)
        buffer_bytes = b''

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, buffer_bytes, buffer_start_time
        if status:
            logging.warning(f"Audio status: {status}")

        audio_frame = indata.flatten().tobytes()
        is_speech = vad.is_speech(audio_frame, sample_rate)

        if not triggered:
            if is_speech:
                triggered = True
                buffer_bytes = audio_frame
                buffer_start_time = time.time()
                audio_callback.silence_counter = 0
        else:
            buffer_bytes += audio_frame
            duration = time.time() - buffer_start_time

            # Finalize chunk if maximum duration is reached
            if duration >= max_audio_duration_s:
                triggered = False
                finalize_chunk()
            elif not is_speech:
                if not hasattr(audio_callback, "silence_counter"):
                    audio_callback.silence_counter = 0
                audio_callback.silence_counter += 1
                if audio_callback.silence_counter >= 3:
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
                logging.info("Recording audio with advanced alignment and noise reduction...")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"InputStream error: {e}")

    asyncio.run_coroutine_threadsafe(record(), loop)

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
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
