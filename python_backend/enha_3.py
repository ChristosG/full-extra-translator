import asyncio
import json
import logging
import time
from typing import Optional, List

import aioredis
from transformers import AutoTokenizer
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
from difflib import SequenceMatcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

################################################################
# CONFIG
################################################################
LLAMA_MODEL_NAME = "ensemble"
TRITON_SERVER_URL = "localhost:8000"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSLATION_CHANNEL = "better_translations"
TRANSCRIPTION_CHANNEL = "transcriptions"

MAX_BUFFER_SIZE = 5
MIN_TRANSLATION_INTERVAL = 1.0
SIMILARITY_THRESHOLD = 0.95

redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

################################################################
# LLM CLASS
################################################################
class TritonLLM(LLM):
    llm_url: str = f"http://{TRITON_SERVER_URL}/v2/models/{LLAMA_MODEL_NAME}/generate"

    @property
    def _llm_type(self) -> str:
        return "Triton LLM"

    def _call(
        self,
        prompt: str,
        temperature: float,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": 750,
                "temperature": temperature,
                "top_k": 50,
            }
        }

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.llm_url, json=payload, headers=headers)
            response.raise_for_status()
            text_out = response.json().get('text_output', '')
            return text_out or ""
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

################################################################
# GLOBALS
################################################################
transcription_buffer = []
buffer_lock = asyncio.Lock()

translation_history = []
last_translation_time = time.time()

################################################################
# UTILITY FUNCTIONS
################################################################
def sentences_similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()

def unify_german_texts(old_text: str, new_text: str) -> str:
    """
    Your naive text-based unify function (partial overlap logic).
    ...
    [ same code as before, omitted for brevity ]
    """
    # For demonstration, we assume you already implemented this or something similar
    pass

################################################################
# NEW: GERMAN TEXT CORRECTION STEP
################################################################
def refine_german_text(german_text: str) -> str:
    """
    Let the LLM fix obvious transcription errors or partial words
    in the German text. We give it a context instruction to correct 
    confusion like 'Aufmerksamkeit' -> 'Einladung' if it sees that 
    is more logical in context.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI that corrects partial or misheard German text. "
                "Given some raw German text from a speech transcription, if any words "
                "seem obviously incorrect in context, replace them with the correct word. "
                "Focus on standardizing the text, removing repeated partial words, or "
                "fixing short mistakes. Output only the corrected German text. "
            )
        },
        {"role": "user", "content": german_text.strip()},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    corrected = llm(prompt=prompt, temperature=0.0).strip()
    return corrected

def translate_text(german_text: str) -> str:
    """
    Translates the refined/corrected German text to English
    """
    def _single_pass(txt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI translator working for the German company P&I. "
                    "Your task is to translate the following text from German to English. "
                    "Just provide the English translation without additional explanation."
                )
            },
            {"role": "user", "content": txt.strip()},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return llm(prompt=prompt, temperature=0.0).strip()

    # Attempt #1
    translation = _single_pass(german_text)
    if translation:
        return translation

    # Minimal fallback
    logging.warning("Empty translation, retrying with fallback prompt.")
    fallback_prompt = german_text + "\nPlease provide the English translation now."
    return _single_pass(fallback_prompt)

################################################################
# ASYNC REDIS PUBLISH / SUBSCRIBE
################################################################
async def publish_message(message: dict, channel: str):
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message to {channel}: {message}")
    except Exception as e:
        logging.error(f"Redis Publish Error: {e}")

async def process_buffer():
    global transcription_buffer, translation_history, last_translation_time

    async with buffer_lock:
        if not transcription_buffer:
            return

        # Unify all partial pieces first
        current_text = transcription_buffer[0]
        for i in range(1, len(transcription_buffer)):
            current_text = unify_german_texts(current_text, transcription_buffer[i])

        current_text = current_text.strip()

        # We can do your "is_semantically_complete" check or rely on buffer size/time
        # For brevity, let's assume we do a final translation if buffer is large or enough time passed
        buffer_is_full = (len(transcription_buffer) >= MAX_BUFFER_SIZE)
        time_since_last = time.time() - last_translation_time

        if buffer_is_full or time_since_last >= MIN_TRANSLATION_INTERVAL:
            # 1) Let the LLM correct the German text
            corrected_german = refine_german_text(current_text)
            logging.info(f"Refined German text: {corrected_german}")

            # 2) Translate corrected text
            english_translation = translate_text(corrected_german)
            logging.info(f"Translation: {english_translation}")

            if english_translation:
                # Compare to last translation to avoid duplicates
                if translation_history:
                    sim = sentences_similarity(english_translation, translation_history[-1]['translation'])
                else:
                    sim = 0.0

                if sim < SIMILARITY_THRESHOLD:
                    translated_message = {
                        'translation': english_translation,
                        'finalized': buffer_is_full  # or some logic
                    }
                    await publish_message(translated_message, TRANSLATION_CHANNEL)
                    translation_history.append({'translation': english_translation, 'finalized': buffer_is_full})
                else:
                    logging.info("Translation is too similar to last. Skipping publish.")

            # Reset or partially reset buffer
            if buffer_is_full:
                transcription_buffer.clear()

            last_translation_time = time.time()

async def translate_and_publish(message: dict):
    transcription = message.get('transcription')
    if not transcription:
        logging.warning("Message has no 'transcription'.")
        return

    async with buffer_lock:
        transcription_buffer.append(transcription.strip())

    await process_buffer()

async def listen_to_redis():
    try:
        pubsub = redis.pubsub()
        await pubsub.subscribe(TRANSCRIPTION_CHANNEL)
        logging.info(f"Subscribed to Redis channel: {TRANSCRIPTION_CHANNEL}")

        async for raw_msg in pubsub.listen():
            if raw_msg['type'] == 'message':
                data = raw_msg['data']
                try:
                    msg = json.loads(data)
                    logging.info(f"Received transcription: {msg}")
                    await translate_and_publish(msg)
                except json.JSONDecodeError:
                    logging.error("JSON decode error on message.")
    except asyncio.CancelledError:
        logging.info("Redis listener task cancelled.")
    except Exception as e:
        logging.error(f"Error in Redis listener: {e}")

async def main():
    await listen_to_redis()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Translation service stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
