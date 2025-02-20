import asyncio
import json
import logging
import base64
import numpy as np
import torch
import aioredis
import os

# If you're using TTS from tts-org:
from TTS.api import TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################
# CONFIG
########################################
# REDIS_HOST = "localhost"
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = 6379

BETTER_TRANSLATION_CHANNEL = "better_translations"
TTS_CHANNEL = "tts"

TTS_MODEL_NAME = "tts_models/en/ljspeech/vits--neon"
TTS_SAMPLE_RATE = 22050

tts_model = None


########################################
# TTS MODEL INIT
########################################
def init_tts_model():
    global tts_model
    if tts_model is not None:
        return
    logger.info(f"Loading English TTS model: {TTS_MODEL_NAME}")
    tts_model_local = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model_local.to(device)
    tts_model = tts_model_local
    logger.info("TTS model loaded successfully.")


########################################
# TTS INFERENCE
########################################
def run_tts(text: str):
    text = text.strip()
    if not text or not tts_model:
        return None

    audio_samples = tts_model.tts(text)
    audio_samples = np.array(audio_samples, dtype=np.float32)

    raw_bytes = audio_samples.tobytes()
    b64_audio = base64.b64encode(raw_bytes).decode("utf-8")
    return {
        "base64_audio": b64_audio,
        "sample_rate": TTS_SAMPLE_RATE
    }

########################################
# MAIN SUBSCRIBER LOOP
########################################
async def auto_tts_subscriber():
    """
    Subscribes to the BETTER_TRANSLATION_CHANNEL.
    Whenever a new translation arrives, run TTS -> publish to TTS_CHANNEL.
    """
    redis_conn = aioredis.from_url(
        f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
        decode_responses=True,
        max_connections=10
    )

    pubsub = redis_conn.pubsub()
    await pubsub.subscribe(BETTER_TRANSLATION_CHANNEL)
    logger.info(f"Subscribed to channel: {BETTER_TRANSLATION_CHANNEL}")

    try:
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                data_str = msg["data"]
                # Attempt to parse JSON
                try:
                    data_obj = json.loads(data_str)
                except json.JSONDecodeError:
                    data_obj = {}

                text = data_obj.get("translation", "").strip()
                if text:
                    # Run TTS
                    audio_info = run_tts(text)
                    if audio_info:
                        out_str = json.dumps(audio_info)
                        # Publish the base64 audio to TTS_CHANNEL
                        await redis_conn.publish(TTS_CHANNEL, out_str)
                        logger.info(f"Published TTS audio to '{TTS_CHANNEL}' for text: {text[:50]}")
    except asyncio.CancelledError:
        logger.info("auto_tts_subscriber cancelled.")
    except Exception as e:
        logger.error(f"auto_tts_subscriber error: {e}", exc_info=True)
    finally:
        await pubsub.unsubscribe(BETTER_TRANSLATION_CHANNEL)
        await redis_conn.close()

########################################
# ENTRY POINT
########################################
async def main():
    init_tts_model()
    logger.info("Starting TTS subscriber loop ...")
    await auto_tts_subscriber()

if __name__ == "__main__":
    asyncio.run(main())
