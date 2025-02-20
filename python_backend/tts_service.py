# tts_service.py
import logging
import base64
import numpy as np
import torch
from TTS.api import TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# You can change the model name to any English TTS model you'd like, e.g.:
TTS_MODEL_NAME = 'tts_models/de/thorsten/tacotron2-DDC' # "tts_models/en/vctk/vits" # "tts_models/en/ljspeech/tacotron2-DDC"
SAMPLE_RATE = 22050

tts_model = None

def init_tts():
    """
    Loads the TTS model on startup (once).
    """
    global tts_model
    logger.info(f"Loading English TTS model: {TTS_MODEL_NAME}")
    tts_model = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model.to(device)
    logger.info("TTS model loaded successfully.")

def synthesize_tts(text: str) -> str:
    """
    Takes a text string, returns base64-encoded raw float32 PCM audio.
    The front-end can decode and play it.
    """
    if not tts_model:
        raise RuntimeError("TTS model not initialized. Call init_tts() first.")

    text = text.strip()
    if not text:
        return ""

    # Synthesize (returns a list or np array of samples)
    audio_samples = tts_model.tts(text)
    audio_samples = np.array(audio_samples, dtype=np.float32)  # ensure float32

    # Convert to raw bytes
    raw_bytes = audio_samples.tobytes()
    # Base64-encode
    b64_audio = base64.b64encode(raw_bytes).decode("utf-8")
    return b64_audio
