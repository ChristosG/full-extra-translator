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
from pydantic import BaseModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

WHISPER_PROMPT = "<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"
LANGUAGE_CODE = "de"
WHISPER_MODEL_NAME = "whisper"
TRITON_SERVER_URL = "localhost:8001"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSCRIPTION_CHANNEL = "transcriptions"
SAMPLE_RATE = 16000  # Hz
FRAME_DURATION = 30  # ms
COOLDOWN_PERIOD = 0.2  # sec
MAX_AUDIO_DURATION = 3.8  # sec
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# NEW: "hangover" frames after speech ends, and "lookback" frames
# we'll store ~300ms before start, 300ms after end
HANGOVER_FRAMES = int(300 / FRAME_DURATION)  # e.g. 300ms
LOOKBACK_FRAMES = int(300 / FRAME_DURATION)  # e.g. 300ms

# Redis init
redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

# ------------------------------
# Create a single Triton gRPC client up-front so we reuse
# ------------------------------
try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
    # Optional: check server readiness once
    if not triton_client.is_server_ready():
        raise ValueError("Triton server not ready.")
except Exception as e:
    logging.error(f"Failed to create Triton client: {e}")
    triton_client = None

class TritonLLM(LLM):
    llm_url: str = f"http://{TRITON_SERVER_URL}/v2/models/llama3.1/generate"

    class Config:
        extra = 'forbid'

    @property
    def _llm_type(self) -> str:
        return "Triton LLM"

    def _call(
        self,
        prompt: str,
        temperature: float,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": 100,
                "temperature": temperature,
                "top_k": 50,
            }
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.llm_url, json=payload, headers=headers)
            response.raise_for_status()
            translation = response.json().get('text_output', '')
            if not translation:
                raise ValueError("No 'text_output' field in the response.")
            return translation
        except requests.exceptions.RequestException as e:
            logging.error(f"LLM request failed: {e}")
            return ""
        except ValueError as ve:
            logging.error(f"LLM response error: {ve}")
            return ""

    @property
    def _identifying_params(self) -> dict:
        return {"llmUrl": self.llm_url}

llm = TritonLLM()

try:
    tokenizer = AutoTokenizer.from_pretrained("/mnt/nvme512/engines/Meta-Llama-3.1-8B-Instruct")
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)

def normalize_audio(audio_int16: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalizes audio data to [-1.0, 1.0] and clamps values.
    """
    try:
        audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max
        max_val = np.max(np.abs(audio_float32))
        if max_val > 0:
            audio_normalized = audio_float32 / max_val
            audio_normalized = np.clip(audio_normalized, -1.0, 1.0)
        else:
            audio_normalized = audio_float32

        if not np.all(np.abs(audio_normalized) <= 1.0):
            logging.error("Audio normalization failed to clamp values within [-1.0, 1.0].")
            return None
        return audio_normalized
    except Exception as e:
        logging.error(f"Audio normalization error: {e}")
        return None

def transcribe_audio(
    audio_data: np.ndarray,
    whisper_prompt: str,
    language: str,
    model_name: str = "whisper-large-v3",
    triton_client: Optional[grpcclient.InferenceServerClient] = None
) -> Optional[str]:
    """
    Sends audio data to the Triton server via gRPC for transcription.
    Reuses an existing triton_client if provided.
    """
    if triton_client is None:
        logging.error("No valid Triton client available.")
        return None

    try:
        if not triton_client.is_model_ready(model_name):
            logging.error(f"Model {model_name} is not ready on Triton server.")
            return None

        samples = audio_data.astype(np.float32)
        samples = np.expand_dims(samples, axis=0)  # [1, audio_length]

        # Prepare inputs
        input_wav = grpcclient.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype))
        input_wav.set_data_from_numpy(samples)

        input_text = grpcclient.InferInput("TEXT_PREFIX", [1, 1], "BYTES")
        input_text.set_data_from_numpy(np.array([[whisper_prompt.encode()]], dtype=object))

        # We want only the "TRANSCRIPTS" output
        outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTS")]

        # Send request
        response = triton_client.infer(
            model_name=model_name,
            inputs=[input_wav, input_text],
            outputs=outputs
        )

        transcription = response.as_numpy("TRANSCRIPTS")[0]
        if isinstance(transcription, np.ndarray):
            transcription = b" ".join(transcription).decode("utf-8")
        else:
            transcription = transcription.decode("utf-8")

        logging.debug(f"Raw Transcription: {transcription}")
        return transcription

    except Exception as e:
        logging.error(f"Transcription Error: {e}")
        return None

async def publish_message(message: dict, channel: str):
    """
    Publishes a JSON message to the specified Redis channel.
    """
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message to {channel}: {message}")
    except Exception as e:
        logging.error(f"Redis Publish Error: {e}")

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
    Records audio, applies VAD to detect speech segments, transcribes using Triton,
    and publishes transcriptions to Redis. 
    This version uses hangover/ look-back frames to reduce cut-off issues.
    """

    vad = webrtcvad.Vad(2)  # Aggressiveness from 0 to 3. Tune to taste.
    channels = 1
    dtype = 'int16'
    frame_size = int(sample_rate * frame_duration_ms / 1000)

    # We'll store raw frames in a ring buffer and in active speech buffer
    ring_buffer = collections.deque(maxlen=LOOKBACK_FRAMES)  # for look-back
    triggered = False

    # This will collect raw audio data (bytes) for the "active" chunk
    active_frames = []
    chunk_start_time = 0.0
    last_transcription_time = 0

    transcribe_lock = threading.Lock()

    async def process_transcription(final_audio_bytes: bytes):
        """
        Normalize, run inference, and publish result.
        Uses a lock to ensure only one transcription runs at a time.
        """
        nonlocal last_transcription_time
        with transcribe_lock:
            current_time = time.time()
            if current_time - last_transcription_time < cooldown_period_s:
                logging.debug("Cooldown period active. Skipping transcription.")
                return

            audio_int16 = np.frombuffer(final_audio_bytes, dtype=np.int16)
            audio_normalized = normalize_audio(audio_int16)
            if audio_normalized is None:
                return

            transcription = transcribe_audio(
                audio_data=audio_normalized,
                whisper_prompt=whisper_prompt,
                language=language,
                model_name=whisper_model_name,
                triton_client=triton_client
            )

            if transcription and len(transcription.strip()) > 2:
                # Example checks to skip certain phrases
                skip_phrases = ["Vielen Dank.", "Tschüss"]
                if any(sp in transcription for sp in skip_phrases):
                    logging.info(f"Skipping transcription. Contains skip phrase: {transcription}")
                    return

                logging.info("=== New Transcription ===")
                logging.info(f"Transcription (German): {transcription}")

                message = {'transcription': transcription}
                asyncio.run_coroutine_threadsafe(
                    publish_message(message, transcription_channel),
                    loop
                )
                last_transcription_time = current_time
            else:
                logging.debug(f"Ignoring empty or short transcription: {transcription}")

    def finalize_chunk():
        """
        Called when we decide we've truly reached the end of speech.
        We also do a 300ms (HANGOVER_FRAMES) look-back to avoid cutting final words too early.
        """
        nonlocal active_frames, triggered

        # We already have active_frames with speech frames. We add a little ring_buffer tail
        final_audio_bytes = b''.join(active_frames)

        # Reset
        active_frames = []
        triggered = False

        # Kick off transcription
        asyncio.run_coroutine_threadsafe(
            process_transcription(final_audio_bytes),
            loop
        )

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, active_frames, chunk_start_time

        if status:
            logging.warning(f"Recording status: {status}")

        # Flatten the chunk into raw bytes
        frame_bytes = indata.flatten().tobytes()
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if not triggered:
            # Accumulate frames in ring_buffer
            ring_buffer.append(frame_bytes)
            if is_speech:
                # Once triggered, prepend the look-back frames
                triggered = True
                # Add everything in ring_buffer to the active_frames:
                active_frames = list(ring_buffer)
                ring_buffer.clear()
                active_frames.append(frame_bytes)
                chunk_start_time = time.time()
                logging.info("Speech detected. Start collecting chunk.")
        else:
            # Already triggered: keep collecting frames
            active_frames.append(frame_bytes)

            chunk_duration = time.time() - chunk_start_time
            if chunk_duration >= max_audio_duration_s:
                logging.info("Max chunk duration reached -> finalize now.")
                finalize_chunk()
                return

            if is_speech:
                # If it's speech, reset the "hangover" in ring_buffer
                ring_buffer.clear()
            else:
                # If it's silence, we add frame to ring_buffer
                ring_buffer.append(frame_bytes)
                # If ring_buffer is full (>= HANGOVER_FRAMES), we finalize
                if len(ring_buffer) >= HANGOVER_FRAMES:
                    logging.info("End of speech detected -> finalize chunk.")
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
                logging.info("Recording with VAD... Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error initializing audio stream: {e}")

    asyncio.run_coroutine_threadsafe(record(), loop)


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        recording_thread = threading.Thread(
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
            )
        )
        recording_thread.start()
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("Recording stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        loop.stop()
        recording_thread.join()
