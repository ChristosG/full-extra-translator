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
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

# For spectrogram alignment
import librosa
from scipy.signal import correlate2d

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

######################################################################
# CONFIG
######################################################################
WHISPER_PROMPT = "<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"
LANGUAGE_CODE = "de"
WHISPER_MODEL_NAME = "whisper"
TRITON_SERVER_URL = "localhost:8001"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSCRIPTION_CHANNEL = "transcriptions"

SAMPLE_RATE = 16000
FRAME_DURATION = 30     # ms per frame
VAD_AGGRESSIVENESS = 1  # 0–3

MAX_AUDIO_DURATION = 3.0         # Max chunk length in seconds
COOLDOWN_PERIOD = 0.2           # Post-transcription cooldown

# Hangover: require 0.3s of consecutive silence before finalizing
HANGOVER_SEC = 0.3

# Overlap alignment
PREV_OVERLAP_SEC = 0.5
CURR_OVERLAP_SEC = 1.0
MATCH_THRESHOLD = 0.7  # correlation threshold

# Phrases to skip
SKIP_PHRASES = ["Vielen Dank.", "Tschüss", "Bis zum nächsten Mal"]

######################################################################
# SETUP REDIS AND TRITON
######################################################################
redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
    if not triton_client.is_server_ready():
        raise RuntimeError("Triton server not ready.")
except Exception as e:
    logging.error(f"Error creating Triton client: {e}")
    triton_client = None

######################################################################
# HELPER FUNCTIONS
######################################################################
def compute_mel_spectrogram(audio_float32: np.ndarray, sr: int) -> np.ndarray:
    """
    Convert 1D float32 audio to a log-mel-spectrogram using librosa.
    We'll do a standard n_fft=512, hop_length=128, n_mels=40 approach.
    """
    import librosa
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
    2D cross-correlation in 'valid' mode (sliding window).
    Return the best normalized correlation [0..1].
    """
    corr = correlate2d(ref_spec, tgt_spec, mode='valid')
    max_corr = np.max(corr)

    ref_power = np.sum(ref_spec * ref_spec)
    tgt_power = np.sum(tgt_spec * tgt_spec)
    if ref_power <= 1e-8 or tgt_power <= 1e-8:
        return 0.0

    norm_factor = np.sqrt(ref_power * tgt_power)
    normalized_corr = max_corr / (norm_factor + 1e-10)
    return float(np.clip(normalized_corr, -1.0, 1.0))

def align_and_trim_chunk(
    prev_overlap_audio: np.ndarray,
    current_chunk: np.ndarray,
    sr: int,
    threshold: float
) -> np.ndarray:
    """
    Compare last PREV_OVERLAP_SEC of `prev_overlap_audio` to the first
    CURR_OVERLAP_SEC of `current_chunk`. If correlation is above 'threshold',
    remove the overlap portion from current_chunk.
    """
    overlap_len_prev = int(PREV_OVERLAP_SEC * sr)
    if len(prev_overlap_audio) <= overlap_len_prev:
        ref_part = prev_overlap_audio
    else:
        ref_part = prev_overlap_audio[-overlap_len_prev:]

    overlap_len_curr = int(CURR_OVERLAP_SEC * sr)
    if len(current_chunk) <= overlap_len_curr:
        target_part = current_chunk
    else:
        target_part = current_chunk[:overlap_len_curr]

    # Edge case: too small to do correlation
    if len(ref_part) < 256 or len(target_part) < 256:
        return current_chunk

    # Convert to float mel-spec
    ref_float = ref_part.astype(np.float32) / 32768.0
    tgt_float = target_part.astype(np.float32) / 32768.0
    ref_spec = compute_mel_spectrogram(ref_float, sr)
    tgt_spec = compute_mel_spectrogram(tgt_float, sr)

    corr_score = sliding_correlation_score(ref_spec, tgt_spec)
    logging.debug(f"[align_and_trim_chunk] correlation={corr_score:.3f}")

    trimmed_chunk = current_chunk
    if corr_score >= threshold:
        # Do a 'full' cross-correlation to find best offset
        from scipy.signal import correlate2d
        full_corr = correlate2d(ref_spec, tgt_spec, mode='full')
        max_y, max_x = np.unravel_index(np.argmax(full_corr), full_corr.shape)
        center_x = ref_spec.shape[1] - 1
        time_shift = max_x - center_x

        hop_length = 128
        shift_in_samples = time_shift * hop_length
        logging.debug(f"[align_and_trim_chunk] best offset shift_in_samples={shift_in_samples}")

        if shift_in_samples < 0:
            remove_count = abs(int(shift_in_samples))
            if remove_count < len(current_chunk):
                trimmed_chunk = current_chunk[remove_count:]
            else:
                trimmed_chunk = np.array([], dtype=current_chunk.dtype)

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
# RECORDING WITH HANGOVER APPROACH
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
    """
    Record audio with VAD + 'hangover' approach, then do overlap alignment
    between consecutive chunks, transcribe each chunk once, skip phrases, etc.
    """

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    channels = 1
    dtype = 'int16'
    frame_size = int(sample_rate * frame_duration_ms / 1000)

    triggered = False
    buffer_bytes = b''
    buffer_start_time = time.time()

    # Hangover ring buffer for silent frames
    hangover_frames = int(HANGOVER_SEC / (frame_duration_ms / 1000.0))
    silence_buffer = collections.deque(maxlen=hangover_frames)

    last_transcription_time = 0.0
    transcribe_lock = threading.Lock()

    # Overlap logic
    prev_overlap_audio: Optional[np.ndarray] = None

    async def process_transcription(final_chunk: bytes):
        nonlocal prev_overlap_audio, last_transcription_time
        with transcribe_lock:
            now = time.time()
            if (now - last_transcription_time) < cooldown_period_s:
                logging.debug("Cooldown active. Skipping transcription.")
                return

            audio_int16 = np.frombuffer(final_chunk, dtype=np.int16)
            if len(audio_int16) == 0:
                return

            audio_float = normalize_audio(audio_int16)
            transcription = transcribe_audio(
                audio_data=audio_float,
                whisper_prompt=whisper_prompt,
                language=language,
                model_name=whisper_model_name,
                triton_client=triton_client
            )
            if transcription and len(transcription.strip()) > 2:
                # Skip certain phrases
                if any(sp in transcription for sp in SKIP_PHRASES):
                    logging.info(f"Skipping transcription containing skip phrase: {transcription}")
                else:
                    logging.info(f"Transcription: {transcription}")
                    msg = {"transcription": transcription}
                    asyncio.run_coroutine_threadsafe(
                        publish_message(msg, transcription_channel),
                        loop
                    )
                last_transcription_time = now

            # Update prev_overlap_audio for next chunk
            overlap_len_samples = int(PREV_OVERLAP_SEC * sample_rate)
            if len(audio_float) > overlap_len_samples:
                prev_overlap_audio = audio_float[-overlap_len_samples:].copy()
            else:
                prev_overlap_audio = audio_float.copy()

    def finalize_chunk():
        nonlocal buffer_bytes
        chunk_int16 = np.frombuffer(buffer_bytes, dtype=np.int16)

        # Overlap alignment
        if prev_overlap_audio is not None and len(prev_overlap_audio) > 0:
            aligned_chunk = align_and_trim_chunk(
                prev_overlap_audio,
                chunk_int16,
                sr=sample_rate,
                threshold=MATCH_THRESHOLD
            )
            trimmed_bytes = aligned_chunk.tobytes()
        else:
            trimmed_bytes = buffer_bytes

        # Trigger transcription
        asyncio.run_coroutine_threadsafe(
            process_transcription(trimmed_bytes),
            loop
        )

        buffer_bytes = b''

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, buffer_bytes, buffer_start_time, silence_buffer

        if status:
            logging.warning(f"Audio status: {status}")

        audio_frame = indata.flatten().tobytes()
        is_speech = vad.is_speech(audio_frame, sample_rate)

        if not triggered:
            # If we detect speech, start new chunk
            if is_speech:
                triggered = True
                buffer_bytes = audio_frame
                buffer_start_time = time.time()
                silence_buffer.clear()
        else:
            # Already triggered, keep collecting
            buffer_bytes += audio_frame
            duration = time.time() - buffer_start_time

            # If we exceed max chunk length, finalize
            if duration >= max_audio_duration_s:
                triggered = False
                finalize_chunk()
                return

            # Check speech or silence
            if is_speech:
                # reset silence buffer
                silence_buffer.clear()
            else:
                # add this frame to silence_buffer
                silence_buffer.append(audio_frame)
                # if we've hit hangover limit => finalize
                if len(silence_buffer) == silence_buffer.maxlen:
                    triggered = False
                    finalize_chunk()

    async def record():
        try:
            with sd.InputStream(
                channels=channels,
                samplerate=sample_rate,
                dtype=dtype,
                blocksize=frame_size,
                callback=audio_callback
            ):
                logging.info("Recording with hangover + overlap alignment... Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(0.05)
        except Exception as e:
            logging.error(f"InputStream error: {e}")

    asyncio.run_coroutine_threadsafe(record(), loop)

######################################################################
# MAIN
######################################################################
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
