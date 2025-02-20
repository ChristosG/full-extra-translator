import asyncio
import json
import logging
import re
import time
from typing import Optional, List, Tuple

import aioredis
from sentence_splitter import SentenceSplitter
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
TRITON_SERVER_URL= 'trtllm24:5991'
TRITON_SERVER_URL_4060='trtllm24:5991'
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = 6379
TRANSLATION_CHANNEL = "enhanced_translations"
TRANSCRIPTION_CHANNEL = "transcriptions"

MAX_BUFFER_LENGTH = 200  # Characters
MIN_TRANSLATION_INTERVAL = 1.0  # seconds
SIMILARITY_THRESHOLD = 0.85
CONTEXT_WINDOW_SIZE = 2  # Previous sentences to maintain as context

################################################################
# REDIS SETUP
################################################################
redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

################################################################
# LLM WRAPPER (TRITON)
################################################################
class TritonLLM(LLM):
    llm_url: str = f"http://{TRITON_SERVER_URL_4060}/v2/models/{LLAMA_MODEL_NAME}/generate"

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
# INITIALIZE COMPONENTS
################################################################
llm = TritonLLM()
splitter = SentenceSplitter(language='de')

try:
    tokenizer = AutoTokenizer.from_pretrained("/engines/Meta-Llama-3.1-8B-Instruct")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)

################################################################
# GLOBAL STATE
################################################################
transcription_buffer = []
translation_history = []
german_context = []
last_translation_time = time.time()
buffer_lock = asyncio.Lock()

################################################################
# HELPER FUNCTIONS
################################################################
def sentence_ended(text: str) -> bool:
    return re.search(r'[.!?](?:\s*|$)', text[-10:] if len(text) > 10 else text) is not None

def split_sentences(text: str) -> List[str]:
    try:
        return splitter.split(text)
    except Exception as e:
        logging.error(f"Sentence splitting error: {e}")
        return [text]

def text_similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()

def format_translation_prompt(context: List[str], current_text: str) -> str:
    context_str = " ".join(context[-CONTEXT_WINDOW_SIZE:])
    return (
        f"Translate the following German text to English, considering this context:\n"
        f"Context: {context_str}\n"
        f"Text to translate: {current_text}\n"
        f"Translation:"
    )

async def process_transcription_chunk(chunk: str) -> Tuple[str, str]:
    """
    Process a single transcription chunk through refinement and translation
    """
    refined = await refine_german_text(chunk)
    context = german_context[-CONTEXT_WINDOW_SIZE:]
    
    messages = [{
        "role": "system",
        "content": "You are a professional German-to-English translator. Maintain context coherence and fix any partial words naturally."
    }, {
        "role": "user",
        "content": format_translation_prompt(context, refined)
    }]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    translation = llm(prompt=prompt, temperature=0.1).strip()
    
    # Post-process translation
    translation = re.sub(r'^Translation:\s*', '', translation, flags=re.IGNORECASE)
    return refined, translation

async def refine_german_text(text: str) -> str:
    """
    Specialized refinement focusing on partial endings
    """
    if len(text) < 10 or text[-1] in '.!?':
        return text

    messages = [{
        "role": "system",
        "content": (
            "Correct the German text focusing on the ending. "
            "Complete or fix partial words at the end. "
            "Output ONLY the corrected text."
        )
    }, {
        "role": "user",
        "content": f"Text: {text}"
    }]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    refined = llm(prompt=prompt, temperature=0.0).strip()
    return refined if len(refined) > len(text)//2 else text

################################################################
# CORE PROCESSING LOGIC
################################################################
async def process_buffer():
    global transcription_buffer, german_context, last_translation_time

    async with buffer_lock:
        if not transcription_buffer:
            return

        current_text = ' '.join(transcription_buffer).strip()
        sentences = split_sentences(current_text)
        
        # Find complete sentences
        complete, remaining = [], []
        for sent in sentences:
            if sentence_ended(sent):
                complete.append(sent)
            else:
                remaining.append(sent)
                break
        remaining = ' '.join(remaining + sentences[len(complete)+1:])

        # Process if complete sentences found or timeout
        if not complete:
            if time.time() - last_translation_time >= MIN_TRANSLATION_INTERVAL:
                complete = [current_text]
                remaining = ''
            else:
                return

        complete_text = ' '.join(complete).strip()
        refined, translation = await process_transcription_chunk(complete_text)

        # Similarity check
        prev_text = translation_history[-1]['german'] if translation_history else ''
        similarity = text_similarity(refined, prev_text)
        
        if similarity < SIMILARITY_THRESHOLD:
            await redis.publish(TRANSLATION_CHANNEL, json.dumps({
                "german": refined,
                "translation": translation,
                "finalized": True
            }))
            translation_history.append({"german": refined, "translation": translation})
            german_context.extend(split_sentences(refined))

        # Update buffer with remaining text
        transcription_buffer = [remaining.strip()] if remaining.strip() else []
        last_translation_time = time.time()

################################################################
# MAIN LOOP
################################################################
async def handle_transcription(message: dict):
    text = message.get('transcription', '').strip()
    if not text:
        return

    async with buffer_lock:
        if transcription_buffer:
            # Merge with previous text using overlap detection
            prev_text = transcription_buffer[-1]
            overlap = min(20, len(prev_text), len(text))
            if prev_text[-overlap:] == text[:overlap]:
                transcription_buffer[-1] += text[overlap:]
            else:
                transcription_buffer.append(text)
        else:
            transcription_buffer.append(text)

    await process_buffer()

async def redis_listener():
    pubsub = redis.pubsub()
    await pubsub.subscribe(TRANSCRIPTION_CHANNEL)
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                data = json.loads(message['data'])
                await handle_transcription(data)
            except Exception as e:
                logging.error(f"Processing error: {e}")

async def main():
    await redis_listener()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Service stopped")
