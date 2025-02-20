import asyncio
import json
import logging
import time
from typing import Optional, List

import aioredis
from transformers import AutoTokenizer
from langchain.llms.base import LLM
from pydantic import BaseModel, Extra
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
SIMILARITY_THRESHOLD = 0.95  # how close we consider two translations to be "the same"

# Example skip phrases you might also re-add from your code:
# SKIP_PHRASES = ["Vielen Dank.", "Tschüss"]  # If needed

################################################################
# REDIS SETUP
################################################################
redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

################################################################
# LLM CLASS
################################################################
class TritonLLM(LLM):
    """
    A simple wrapper around your Triton-based LLM endpoint.
    """
    llm_url: str = f"http://{TRITON_SERVER_URL}/v2/models/{LLAMA_MODEL_NAME}/generate"

    class Config:
        extra = 'forbid'

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
        # Potentially bump max_tokens from 500 to 750 or more
        payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": 750,
                "temperature": temperature,
                "top_k": 50,
                # You can try adding "min_new_tokens": 50, etc. for more forced output
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

################################################################
# GLOBALS
################################################################
transcription_buffer = []
buffer_lock = asyncio.Lock()

translation_history = []  # Keep track of recent translations
last_translation_time = time.time()

################################################################
# FUNCTIONS
################################################################
def sentences_similarity(s1: str, s2: str) -> float:
    """
    Calculates the string similarity ratio between two sentences.
    """
    return SequenceMatcher(None, s1, s2).ratio()

def unify_german_texts(old_text: str, new_text: str) -> str:
    """
    Simple text-based approach to unify partial overlaps between `old_text`
    and `new_text` in German. We look for overlapping substrings near the boundary
    to avoid repeating words like "eingeladen habt" / "eingeschaltet habt".
    
    This is a naive approach. If you need something more robust, consider:
      - advanced NLP alignment,
      - or an LLM-based unifier with a custom prompt.
    """
    old_text = old_text.strip()
    new_text = new_text.strip()
    if not old_text:
        return new_text
    if not new_text:
        return old_text

    # We'll do a simple substring search from the end of old_text
    # and the start of new_text. We'll try to find the largest common substring
    # near the boundary. You can use difflib or a rolling approach.

    # For demonstration, let's do a small overlap window of ~30 chars near the
    # end of old_text, see if it appears at the start of new_text.
    max_overlap_window = 30
    overlap_window = old_text[-max_overlap_window:]
    idx = new_text.lower().find(overlap_window.lower())
    if idx == 0:
        # Means the new_text starts with that overlap
        # so we can unify them without repeating overlap_window
        # e.g., old_text="heute wieder eingeladen habt."
        # new_text="eingeschaltet habt, um mit uns..."
        # If there's no direct substring match, we skip this logic.
        unified = old_text + new_text[len(overlap_window):]
        return unified
    else:
        # fallback: do partial difflib approach
        seqMatch = SequenceMatcher(None, overlap_window.lower(), new_text.lower())
        match = seqMatch.find_longest_match(0, len(overlap_window), 0, len(new_text))
        # if match.size is big enough, do some trimming
        if match.size > 5:
            # e.g. we found a decent overlap of length > 5
            overlap_str = overlap_window[match.a: match.a + match.size]
            # see where it is in new_text
            new_start_idx = match.b + match.size
            # unify
            unified = old_text + new_text[new_start_idx:]
            return unified
        else:
            # no overlap, just combine with a space
            return old_text + " " + new_text

def is_semantically_complete(text: str) -> bool:
    """
    Uses the LLM to determine if the text is semantically complete in German.
    You already have this logic, just reusing it. 
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Determine whether the following German text is a complete sentence "
                "or expresses a complete thought. Answer 'Yes' or 'No' without any additional text."
            )
        },
        {"role": "user", "content": text.strip()},
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    response = llm(prompt=prompt, temperature=0.0).strip().lower()

    logging.info(f"Semantic completeness check response: {response}")
    return response.startswith('yes')

def translate_text(transcription: str) -> Optional[str]:
    """
    Translates the transcription from German to English using the LLM.
    Includes a minimal retry if the translation returns empty.
    """
    def _single_pass(txt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI translator working for the German company P&I, which builds the LOGA platform. "
                    "Your task is to translate the following text from German to English. "
                    "Just provide the English translation without any additional explanation or comments."
                )
            },
            {"role": "user", "content": txt.strip()},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return llm(prompt=prompt, temperature=0.0).strip()

    # First attempt
    translation = _single_pass(transcription)
    if translation:
        return translation

    logging.warning("First translation attempt returned empty. Retrying once with a slightly different prompt...")

    # Simple fallback approach: you can also tweak temperature or max_tokens here
    fallback_prompt = transcription + "\nPlease provide an English translation now."
    translation_retry = _single_pass(fallback_prompt)
    return translation_retry or None

async def publish_message(message: dict, channel: str):
    """
    Publishes a JSON message to the specified Redis channel.
    """
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message to {channel}: {message}")
    except Exception as e:
        logging.error(f"Redis Publish Error: {e}")

async def process_buffer():
    """
    Processes the transcription buffer to decide if we should unify & translate.
    """
    global transcription_buffer, translation_history, last_translation_time

    async with buffer_lock:
        if not transcription_buffer:
            return

        # For demonstration, unify all partial pieces in transcription_buffer
        # so we get one "combined" German text. If there's overlap between consecutive lines,
        # we do a unify step. This helps fix boundary issues like "eingeladen habt." / "eingeschaltet habt."
        current_text = transcription_buffer[0]
        for i in range(1, len(transcription_buffer)):
            current_text = unify_german_texts(current_text, transcription_buffer[i])

        current_text = current_text.strip()
        buffer_is_full = (len(transcription_buffer) >= MAX_BUFFER_SIZE)
        semantically_complete = is_semantically_complete(current_text)

        time_since_last_translation = time.time() - last_translation_time

        # If the text is complete or we hit max buffer, or enough time passed
        if semantically_complete or buffer_is_full or time_since_last_translation >= MIN_TRANSLATION_INTERVAL:
            translation = translate_text(current_text)
            if translation:
                logging.info(f"Translation (English): {translation}")

                # Compare to the last known translation to avoid duplicates
                if not translation_history:
                    similarity_to_last = 0.0
                else:
                    similarity_to_last = sentences_similarity(translation, translation_history[-1]['translation'])

                if similarity_to_last < SIMILARITY_THRESHOLD:
                    should_finalize = semantically_complete or buffer_is_full

                    translated_message = {
                        'translation': translation,
                        'finalized': should_finalize
                    }
                    await publish_message(translated_message, TRANSLATION_CHANNEL)
                    translation_history.append({'translation': translation, 'finalized': should_finalize})
                else:
                    logging.info("Translation is very similar to the previous one. Skipping publication.")

                last_translation_time = time.time()

                # If we consider this chunk final or buffer is full, we clear the buffer.
                if semantically_complete or buffer_is_full:
                    transcription_buffer.clear()
            else:
                logging.warning("Translation failed or returned empty. No publication.")
                # Optionally keep the buffer or clear it, depending on your preference.

async def translate_and_publish(message: dict):
    """
    Adds the new transcription chunk to the buffer, then tries to process it.
    """
    transcription = message.get('transcription')
    if not transcription:
        logging.warning("Received message without 'transcription' field.")
        return

    async with buffer_lock:
        transcription_buffer.append(transcription.strip())

    await process_buffer()

async def listen_to_redis():
    """
    Listens to the transcriptions Redis channel and processes translations.
    """
    try:
        pubsub = redis.pubsub()
        await pubsub.subscribe(TRANSCRIPTION_CHANNEL)
        logging.info(f"Subscribed to Redis channel: {TRANSCRIPTION_CHANNEL}")

        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = message['data']
                try:
                    msg = json.loads(data)
                    logging.info(f"Received transcription: {msg}")
                    await translate_and_publish(msg)
                except json.JSONDecodeError:
                    logging.error("Failed to decode JSON message.")
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
