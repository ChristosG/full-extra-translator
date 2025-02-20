#!/usr/bin/env python3
"""
Godlike Audio Preprocessing & Transcription Pipeline with Enhanced Overlap Handling

This script records audio (using a VAD trigger) from a virtual sink, applies advanced
preprocessing (adaptive denoising, spectral filtering, dynamic range compression, and
robust overlap alignment), and sends the processed audio to a Triton-deployed Whisper
model for transcription. Transcriptions are published to Redis.

Enhancements include:
  - A sliding window approach with a fixed audio overlap.
  - A short lookahead delay to avoid chopping off the last word.
  - Post‑processing stitching of overlapping transcriptions.
  - Dynamic cancellation of finalization if speech resumes in the lookahead period.

For testing, set TEST_BYPASS_VOICE_CHECK = True to force transcription.
"""

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
from scipy.signal import correlate2d
import difflib
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# For testing, you can bypass the voice quality check:
TEST_BYPASS_VOICE_CHECK = True
SKIP_PHRASES = ["Vielen Dank.", "Tschüss", "Untertitelung des ZDF, 2020"]

# Whisper/Triton/Redis settings
WHISPER_PROMPT = "<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"
LANGUAGE_CODE = "de"
WHISPER_MODEL_NAME = "whisper"
TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "triton_latest:8001")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = 6379
TRANSCRIPTION_CHANNEL = "transcriptions"

# Audio parameters
SAMPLE_RATE = 16000         # Hz
FRAME_DURATION = 30         # ms (block size for callback)
MAX_AUDIO_DURATION = 3.0    # seconds per chunk
COOLDOWN_PERIOD = 0.2       # seconds between transcriptions

# VAD parameters
VAD_AGGRESSIVENESS = 1      # integer 0 (least aggressive) to 3 (most aggressive)

# Overlap settings
PREV_OVERLAP_SEC = 0.5      # seconds from end of previous chunk
CURR_OVERLAP_SEC = 1.0      # seconds from start of current chunk

# Maximum trimming to avoid removing too many words (in seconds)
MAX_TRIM_SEC = 0.2

# Additional thresholds/configuration
MIN_CHUNK_LENGTH = 0.3      # seconds; if chunk is too short, skip it

# Lookahead delay (in seconds) before finalizing a chunk after silence
LOOKAHEAD_DELAY = 0.2

# ------------------------------------------------------------------------------
# Adaptive Speaker Statistics (for dynamic VAD)
# ------------------------------------------------------------------------------
speaker_stats = {"avg_rms": None, "count": 0}

def update_speaker_stats(rms_value):
    global speaker_stats
    if speaker_stats["avg_rms"] is None:
        speaker_stats["avg_rms"] = rms_value
        speaker_stats["count"] = 1
    else:
        count = speaker_stats["count"]
        speaker_stats["avg_rms"] = (speaker_stats["avg_rms"] * count + rms_value) / (count + 1)
        speaker_stats["count"] += 1

def is_valid_voice(audio: np.ndarray, sr: int) -> bool:
    rms = np.sqrt(np.mean(audio**2))
    threshold = 0.005
    if speaker_stats["avg_rms"] is not None:
        threshold = 0.3 * speaker_stats["avg_rms"]
    logging.info(f"Computed RMS: {rms:.5f}, threshold: {threshold:.5f}")
    if rms < threshold:
        logging.debug(f"RMS too low: {rms:.5f} vs threshold {threshold:.5f}")
        return False
    try:
        harmonic_ratio = np.mean(librosa.effects.harmonic(audio))
    except Exception as e:
        harmonic_ratio = 0
    if harmonic_ratio < 0.2:
        logging.debug(f"Harmonic ratio too low: {harmonic_ratio:.3f}")
        return False
    return True

# ------------------------------------------------------------------------------
# SETUP: REDIS & TRITON CLIENT
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
# NEW OVERLAP DETECTION (WAVEFORM CORRELATION)
# ------------------------------------------------------------------------------
def compute_overlap_offset(prev_chunk: np.ndarray, current_chunk: np.ndarray, sr: int) -> int:
    """
    Compute the number of samples to trim from the start of current_chunk based on the overlap
    between the tail of prev_chunk and the head of current_chunk. The overlap offset is capped
    by MAX_TRIM_SEC.
    """
    N_prev = int(PREV_OVERLAP_SEC * sr)
    N_curr = int(CURR_OVERLAP_SEC * sr)
    if len(prev_chunk) < N_prev or len(current_chunk) < N_curr:
        return 0
    prev_overlap = prev_chunk[-N_prev:]
    curr_overlap = current_chunk[:N_curr]
    # Normalize to zero mean, unit variance.
    prev_norm = (prev_overlap - np.mean(prev_overlap)) / (np.std(prev_overlap) + 1e-10)
    curr_norm = (curr_overlap - np.mean(curr_overlap)) / (np.std(curr_overlap) + 1e-10)
    corr_full = np.correlate(prev_norm, curr_norm, mode='full')
    lag = np.argmax(corr_full) - (len(curr_norm) - 1)
    if lag < 0:
        lag = 0
    max_corr = np.max(corr_full)
    norm_corr = max_corr / (len(curr_norm))
    if norm_corr < 0.3:
        return 0
    MAX_TRIM_SAMPLES = int(MAX_TRIM_SEC * sr)
    offset = min(lag, MAX_TRIM_SAMPLES)
    logging.info(f"Computed overlap offset: {offset} samples (norm_corr: {norm_corr:.3f})")
    return offset

# ------------------------------------------------------------------------------
# AUDIO PREPROCESSING FUNCTIONS
# ------------------------------------------------------------------------------
def normalize_audio_func(audio_int16: np.ndarray) -> np.ndarray:
    audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max
    mx = np.max(np.abs(audio_float32))
    if mx > 1e-8:
        audio_float32 /= mx
    return np.clip(audio_float32, -1.0, 1.0)

def adaptive_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 2048:
        return audio
    try:
        audio_float = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        D = librosa.stft(audio_float, n_fft=512, hop_length=160)
        mag = np.abs(D)
        noise_frames = min(5, mag.shape[1])
        noise_profile = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
        mask = np.clip(mag / (noise_profile + 1e-10), 0, 1)
        mask = np.where(mag > 2 * noise_profile, 1.0, mask)
        return librosa.istft(mag * mask * np.exp(1j * np.angle(D)), length=len(audio))
    except Exception as e:
        logging.error(f"Denoising failed: {str(e)}")
        return audio

def transcribe_audio(audio_data: np.ndarray, whisper_prompt: str, language: str, model_name: str,
                     triton_client: grpcclient.InferenceServerClient) -> str:
    if triton_client is None:
        logging.error("No Triton client.")
        return ""
    try:
        if not triton_client.is_model_ready(model_name):
            logging.error(f"Model {model_name} not ready on Triton server.")
            return ""
        audio_data = audio_data.astype(np.float32)
        audio_data = np.expand_dims(audio_data, axis=0)
        input_wav = grpcclient.InferInput("WAV", audio_data.shape, np_to_triton_dtype(audio_data.dtype))
        input_wav.set_data_from_numpy(audio_data)
        input_text = grpcclient.InferInput("TEXT_PREFIX", [1, 1], "BYTES")
        input_text.set_data_from_numpy(np.array([[whisper_prompt.encode()]], dtype=object))
        outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTS")]
        resp = triton_client.infer(model_name=model_name, inputs=[input_wav, input_text], outputs=outputs)
        transcription = resp.as_numpy("TRANSCRIPTS")[0]
        if isinstance(transcription, np.ndarray):
            transcription = b" ".join(transcription).decode("utf-8")
        else:
            transcription = transcription.decode("utf-8")
        return transcription
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return ""

async def publish_message(message: dict, channel: str):
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message: {message}")
    except Exception as e:
        logging.error(f"Redis error: {e}")

# ------------------------------------------------------------------------------
# UTILITY: Merge overlapping transcriptions (text stitching)
# ------------------------------------------------------------------------------
def merge_transcriptions(prev_text: str, current_text: str) -> str:
    """
    Attempt to merge two strings by detecting overlap in words.
    This simply searches for the longest common word sequence at the end of prev_text
    and at the beginning of current_text.
    """
    prev_words = prev_text.split()
    curr_words = current_text.split()
    max_overlap = min(len(prev_words), len(curr_words), 10)
    overlap_length = 0
    for i in range(max_overlap, 0, -1):
        if prev_words[-i:] == curr_words[:i]:
            overlap_length = i
            break
    if overlap_length > 0:
        merged = prev_text + " " + " ".join(curr_words[overlap_length:])
    else:
        merged = prev_text + " " + current_text
    return merged.strip()

# ------------------------------------------------------------------------------
# MAIN RECORDING/VAD LOGIC
# ------------------------------------------------------------------------------
def record_audio_and_publish(whisper_prompt: str, language: str, whisper_model_name: str,
                             transcription_channel: str, sample_rate: int, frame_duration_ms: int,
                             max_audio_duration_s: float, cooldown_period_s: float,
                             loop: asyncio.AbstractEventLoop):
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    channels = 1
    dtype = 'int16'
    frame_size = int(sample_rate * frame_duration_ms / 1000)

    triggered = False
    buffer_bytes = b''
    buffer_start_time = time.time()
    last_transcription_time = 0.0
    transcribe_lock = threading.Lock()
    prev_overlap_audio = None  # stored as a float32 array
    previous_transcription = ""  # for text stitching

    # This will hold a handle to a scheduled finalization task (for lookahead)
    finalization_timer = None

    async def process_transcription(final_chunk: bytes):
        nonlocal prev_overlap_audio, last_transcription_time, previous_transcription
        with transcribe_lock:
            try:
                now = time.time()
                if now - last_transcription_time < cooldown_period_s:
                    return
                audio_int16 = np.frombuffer(final_chunk, dtype=np.int16)
                if len(audio_int16) < 160:
                    return
                audio_float = normalize_audio_func(audio_int16)
                audio_float = adaptive_denoise(audio_float, sample_rate)
                if prev_overlap_audio is not None and len(prev_overlap_audio) > 0:
                    offset = compute_overlap_offset(prev_overlap_audio, audio_float, sample_rate)
                    aligned_audio = audio_float[offset:]
                else:
                    aligned_audio = audio_float
                if len(aligned_audio) / sample_rate < MIN_CHUNK_LENGTH:
                    logging.debug("Chunk too short: skipping transcription")
                    return
                if not TEST_BYPASS_VOICE_CHECK and not is_valid_voice(aligned_audio, sample_rate):
                    logging.debug("Voice check failed: skipping transcription")
                    return
                update_speaker_stats(np.sqrt(np.mean(aligned_audio**2)))
                transcription = transcribe_audio(aligned_audio, whisper_prompt, language, whisper_model_name, triton_client)
                if transcription:
                    clean_transcription = transcription.strip()
                    logging.info(f"Raw transcription: {clean_transcription}")
                    if clean_transcription not in SKIP_PHRASES:
                        # Merge with previous transcription if available.
                        if previous_transcription:
                            merged_text = merge_transcriptions(previous_transcription, clean_transcription)
                        else:
                            merged_text = clean_transcription
                        previous_transcription = merged_text  # update stored text
                        await redis.publish(transcription_channel, json.dumps({"transcription": merged_text}))
                        last_transcription_time = now
                else:
                    logging.info("No transcription produced.")
                # Update previous overlap buffer (store last PREV_OVERLAP_SEC seconds)
                overlap_samples = int(PREV_OVERLAP_SEC * sample_rate)
                if len(audio_float) >= overlap_samples:
                    prev_overlap_audio = audio_float[-overlap_samples:].copy()
                else:
                    prev_overlap_audio = audio_float.copy()
            except Exception as e:
                logging.error(f"Processing error: {str(e)}")

    def finalize_chunk():
        nonlocal buffer_bytes, finalization_timer
        try:
            chunk_int16 = np.frombuffer(buffer_bytes, dtype=np.int16)
            if prev_overlap_audio is not None and len(prev_overlap_audio) > 0:
                current_float = chunk_int16.astype(np.float32) / np.iinfo(np.int16).max
                offset = compute_overlap_offset(prev_overlap_audio, current_float, sample_rate)
                trimmed_bytes = (current_float[offset:] * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            else:
                trimmed_bytes = buffer_bytes
            # Schedule transcription without blocking the audio callback.
            future = asyncio.run_coroutine_threadsafe(process_transcription(trimmed_bytes), loop)
            # Add a callback to log any exceptions from the transcription task.
            future.add_done_callback(
                lambda fut: logging.error(f"Transcription task error: {fut.exception()}") if fut.exception() else None
            )
        finally:
            buffer_bytes = b''
            cancel_finalization_timer()

    def cancel_finalization_timer():
        nonlocal finalization_timer
        if finalization_timer is not None:
            finalization_timer.cancel()
            finalization_timer = None

    def schedule_finalization():
        nonlocal finalization_timer
        if finalization_timer is None:
            finalization_timer = loop.call_later(LOOKAHEAD_DELAY, finalize_chunk)

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, buffer_bytes, buffer_start_time, finalization_timer
        if status:
            logging.warning(f"Audio status: {status}")
        audio_frame = indata.flatten().tobytes()
        is_speech = vad.is_speech(audio_frame, sample_rate)
        if not triggered:
            if is_speech:
                triggered = True
                buffer_bytes = audio_frame
                buffer_start_time = time.time()
                cancel_finalization_timer()  # in case a finalization was pending
                logging.debug("Speech trigger activated")
        else:
            # Append the incoming frame.
            buffer_bytes += audio_frame
            duration = time.time() - buffer_start_time
            if duration >= max_audio_duration_s:
                logging.debug("Max duration reached, finalizing")
                triggered = False
                cancel_finalization_timer()
                finalize_chunk()
            elif not is_speech:
                # Count consecutive non-speech frames (roughly 3 frames for ~90ms)
                if not hasattr(audio_callback, "silence_counter"):
                    audio_callback.silence_counter = 0
                audio_callback.silence_counter += 1
                if audio_callback.silence_counter >= 3:
                    # Instead of finalizing immediately, schedule a lookahead delay.
                    logging.debug("Silence detected, scheduling finalization")
                    triggered = False
                    audio_callback.silence_counter = 0
                    schedule_finalization()
            else:
                # If speech resumes and a finalization was scheduled, cancel it.
                if finalization_timer is not None:
                    cancel_finalization_timer()
                audio_callback.silence_counter = 0

    async def record():
        try:
            with sd.InputStream(channels=channels, samplerate=sample_rate, dtype=dtype,
                                blocksize=frame_size, callback=audio_callback):
                logging.info("Recording with adaptive VAD, lookahead and overlap alignment...")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"InputStream error: {e}")

    # Heartbeat: log every 60 seconds that recording is active.
    async def heartbeat():
        while True:
            logging.info("Heartbeat: recording still active.")
            await asyncio.sleep(60)

    asyncio.run_coroutine_threadsafe(record(), loop)
    asyncio.run_coroutine_threadsafe(heartbeat(), loop)

# ------------------------------------------------------------------------------
# MAIN ENTRY POINT
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
