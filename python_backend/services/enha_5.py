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
import os
################################################################
# LOGGING
################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

################################################################
# CONFIG
################################################################
LLAMA_MODEL_NAME = "ensemble"
# TRITON_SERVER_URL = "localhost:8000"
TRITON_SERVER_URL= 'triton_latest:8000'
# REDIS_HOST = "localhost"
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = 6379
TRANSLATION_CHANNEL = "better_translations"
TRANSCRIPTION_CHANNEL = "transcriptions"

# The buffer collects multiple partial transcriptions
MAX_BUFFER_SIZE = 5
MIN_TRANSLATION_INTERVAL = 1.0  # seconds

# If the new translation is >= 80% similar to the last one, we skip
SIMILARITY_THRESHOLD = 0.85  

################################################################
# REDIS SETUP
################################################################
redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

################################################################
# LLM WRAPPER (TRITON)
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
        """
        Sends a request to the Triton endpoint with a JSON payload.
        """
        payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": 1500,
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

################################################################
# INITIALIZE LLM & TOKENIZER
################################################################
llm = TritonLLM()

try:
    tokenizer = AutoTokenizer.from_pretrained("/engines/Meta-Llama-3.1-8B-Instruct")
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)

################################################################
# GLOBAL BUFFERS & STATE
################################################################
transcription_buffer = []
buffer_lock = asyncio.Lock()

translation_history = []  # track previous translations
last_translation_time = time.time()

################################################################
# HELPER FUNCTIONS
################################################################
def sentences_similarity(s1: str, s2: str) -> float:
    """
    Calculate string similarity ratio between two sentences.
    """
    return SequenceMatcher(None, s1, s2).ratio()

def unify_german_texts(old_text: str, new_text: str) -> str:
    """
    Simple attempt to unify partial overlaps. 
    1) We'll look at the last 25 chars of old_text and see if they appear 
       at the start of new_text. 
    2) If there's a match of length > 5, we unify to avoid duplication. 
    3) Otherwise, we just concatenate them with a space.
    """
    old_text = old_text.strip()
    new_text = new_text.strip()
    if not old_text:
        return new_text
    if not new_text:
        return old_text

    overlap_window = 25
    end_substring = old_text[-overlap_window:] if len(old_text) > overlap_window else old_text
    start_substring = new_text[:overlap_window] if len(new_text) > overlap_window else new_text

    seq = SequenceMatcher(None, end_substring, start_substring)
    match = seq.find_longest_match(0, len(end_substring), 0, len(start_substring))

    if match.size > 5:
        overlap_str = end_substring[match.a: match.a + match.size]
        logging.debug(f"Found overlap: '{overlap_str}' size {match.size}")
        new_start = match.b + match.size
        unified = old_text + new_text[new_start:]
        return unified
    else:
        return old_text + " " + new_text

def refine_german_text(german_text: str) -> str:
    """
    OPTIONAL: Let the LLM fix obvious transcription errors 
    or partial words in the German text.
    """
    if not german_text.strip():
        return german_text

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that corrects German text. "
                "You receive raw German speech transcripts that might contain partial or misheard words. "
                "Your job is to fix those obvious mistakes in context or any inconsistency with the context. "
                "Output ONLY the corrected German text — do not add explanations, disclaimers, or any extra commentary."
            )
        },
        {"role": "user", "content": german_text.strip()},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    corrected = llm(prompt=prompt, temperature=0.0).strip()
    return corrected or german_text

def translate_german_to_english(german_text: str) -> str:
    """
    Translate the given German text to English using the LLM.
    """
    def _single_pass(txt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI translator working for the German company P&I. "
                    "You receive German text, and must provide an English translation. "
                    "IMPORTANT: Output ONLY the English translation, no disclaimers or extra text."
                )
            },
            {"role": "user", "content": txt.strip()},
        ]
        p = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return llm(prompt=p, temperature=0.0).strip()

    if not german_text.strip():
        return ""

    attempt = _single_pass(german_text)
    if attempt:
        return attempt

    # minimal fallback
    fallback = german_text + "\nPlease translate this to English now."
    return _single_pass(fallback)

################################################################
# ASYNC REDIS PUBLISH
################################################################
async def publish_message(message: dict, channel: str):
    """
    Publishes a JSON message to the specified Redis channel.
    """
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message to {channel}: {message}")
    except Exception as e:
        logging.error(f"Redis Publish Error: {e}")

################################################################
# CORE LOGIC
################################################################
async def process_buffer():
    """
    Processes the transcription_buffer to unify partial texts, 
    optionally refine them with LLM, then translate.
    """
    global transcription_buffer, translation_history, last_translation_time

    async with buffer_lock:
        if not transcription_buffer:
            return

        # 1) Unify partial text in the buffer
        current_text = transcription_buffer[0]
        for i in range(1, len(transcription_buffer)):
            current_text = unify_german_texts(current_text, transcription_buffer[i])
        current_text = current_text.strip()

        # 2) Check if we should produce a translation now:
        #    We do so if the buffer is full or enough time has passed
        buffer_is_full = (len(transcription_buffer) >= MAX_BUFFER_SIZE)
        time_since_last = time.time() - last_translation_time

        # If we never set this to True, we won't finalize -> no lines in the "full container"
        should_finalize = buffer_is_full or (time_since_last >= MIN_TRANSLATION_INTERVAL)

        if should_finalize:
            # 2a) Optionally refine
            refined_german = refine_german_text(current_text)
            if refined_german != current_text:
                logging.info(f"Refined text:\nFrom: {current_text}\nTo:   {refined_german}")
            else:
                logging.debug(f"No refinement changes made.")

            # 2b) Translate
            english_translation = translate_german_to_english(refined_german)
            logging.info(f"Translation: {english_translation}")

            if english_translation:
                # 2c) Compare to last to avoid duplicates
                if translation_history:
                    sim = sentences_similarity(english_translation, translation_history[-1]['translation'])
                else:
                    sim = 0.0

                if sim < SIMILARITY_THRESHOLD:
                    # 2d) Publish final or partial update
                    # But since we decided "should_finalize", let's just call this final.
                    translated_message = {
                        "translation": english_translation,
                        "finalized": True  # we treat everything that passes this check as final
                    }
                    await publish_message(translated_message, TRANSLATION_CHANNEL)

                    translation_history.append({
                        "translation": english_translation,
                        "finalized": True
                    })
                else:
                    logging.info("Translation is too similar to last, skipping publish.")
            else:
                logging.warning("Empty translation, skipping publish.")

            # 2e) If we treat this chunk as final, reset the buffer
            transcription_buffer.clear()
            last_translation_time = time.time()

async def translate_and_publish(message: dict):
    """
    Called whenever we get a new transcription from Redis.
    """
    transcription = message.get('transcription')
    if not transcription:
        logging.warning("Received message with no 'transcription'.")
        return

    # Add to our buffer
    async with buffer_lock:
        transcription_buffer.append(transcription.strip())

    # Try to process
    await process_buffer()

async def listen_to_redis():
    """
    Subscribes to 'transcriptions' channel and awaits messages.
    """
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
