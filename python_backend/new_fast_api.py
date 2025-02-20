# fast_api.py

import asyncio
import json
import logging
import base64
import numpy as np
import torch
import uvicorn
import aioredis

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from transformers import AutoTokenizer
import requests
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

import os

########################################
#
########################################

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://zelime.duckdns.org",
        "https://zelime.duckdns.org/tts",
        "localhost:3000/tts",
        "https://zelime.duckdns.org/summaries",
        "localhost:3000/summaries",
                "https://zelime.duckdns.org/ws-tts",
        "localhost:3000/ws-tts",
                "https://zelime.duckdns.org/ws",
        "localhost:3000/ws",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# REDIS_HOST = "localhost"
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = 6379
TRANSLATION_CHANNEL = "translations"
TRANSCRIPTION_CHANNEL = "transcriptions"
BETTER_TRANSLATION_CHANNEL = "better_translations"
SUMMARIES_CHANNEL = "summaries"

TTS_CHANNEL = "tts"

# TRITON config
LLAMA_MODEL_NAME = "ensemble"
# TRITON_SERVER_URL = "localhost:5991"
TRITON_SERVER_URL_4060 = 'trtllm24:5991'


########################################
# MAIN Connection Manager for text
########################################
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        async with self.lock:
            # Copy to avoid mutation while iterating
            for conn in self.active_connections.copy():
                try:
                    await conn.send_text(message)
                except:
                    await self.disconnect(conn)

manager = ConnectionManager()

########################################
# CONNECTION MANAGER FOR TTS
########################################
class TTSConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        async with self.lock:
            for conn in self.active_connections.copy():
                try:
                    await conn.send_text(message)
                except:
                    await self.disconnect(conn)

tts_manager = TTSConnectionManager()

########################################
# Your existing /ws for text
########################################
@app.websocket("/ws")
async def websocket_text(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
            # no read from client for now
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"websocket_text error: {e}", exc_info=True)
        await manager.disconnect(websocket)

########################################
# NEW /ws-tts for TTS audio only
########################################
@app.websocket("/ws-tts")
async def websocket_tts(websocket: WebSocket):
    """
    A separate WS that only broadcasts 'tts' channel data.
    """
    await tts_manager.connect(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
        
    except WebSocketDisconnect:
        await tts_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"websocket_tts error: {e}", exc_info=True)
        await tts_manager.disconnect(websocket)

########################################
# Summaries 
########################################

class SummaryRequest(BaseModel):
    text: str  

class SummaryResponse(BaseModel):
    summary: str

class TritonLLM(LLM):
    llm_url: str = f"http://{TRITON_SERVER_URL_4060}/v2/models/{LLAMA_MODEL_NAME}/generate"

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
                "max_tokens": 200,
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
    tokenizer = AutoTokenizer.from_pretrained("/engines/Meta-Llama-3.1-8B-Instruct")
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)

@app.post("/summaries", response_model=SummaryResponse)
async def summarize_text_endpoint(req: SummaryRequest):
    """
    Takes the text from the front-end, calls Triton or any summarization model,
    returns bulletpoint summary directly.
    """
    # 1) get the text
    raw_text = req.text.strip()
    if not raw_text:
        return {"summary": "No text to summarize."}

    # 2) call your summarization function or direct Triton endpoint
    summary = do_bulletpoint_summary(raw_text)

    return {"summary": summary}

def do_bulletpoint_summary(text: str) -> str:
    """
    Example: calls a local Triton endpoint for summarization.
    Adapt to your actual model route / logic.
    """


    messages = [
        {"role": "system", "content": "You are an AI that summarizes the given text which is from a presentation. Summary the text with no extra disclaimers:\n"},
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    summry = llm(prompt=prompt, temperature=0.0)
    return summry.strip() if summry else "failed."



########################################
# Startup & Shutdown
########################################
@app.on_event("startup")
async def startup_event():
    try:
        app.state.redis = aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
            decode_responses=True,
            max_connections=10
        )
        app.state.pubsub = app.state.redis.pubsub()

        await app.state.pubsub.subscribe(
            TRANSLATION_CHANNEL,
            TRANSCRIPTION_CHANNEL,
            BETTER_TRANSLATION_CHANNEL,
            SUMMARIES_CHANNEL,
            TTS_CHANNEL
        )

        logger.info(f"Subscribed to channels: translations, transcriptions, better_translations, summaries, tts")

        # init TTS

        

        # main text subscription
        app.state.listener_task = asyncio.create_task(listen_to_redis(app.state.pubsub))
    except Exception as e:
        logger.error(f"startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    # close tasks
    app.state.listener_task.cancel()
    try:
        await app.state.listener_task
    except:
        pass

    await app.state.pubsub.unsubscribe(
        TRANSLATION_CHANNEL,
        TRANSCRIPTION_CHANNEL,
        BETTER_TRANSLATION_CHANNEL,
        SUMMARIES_CHANNEL,
        TTS_CHANNEL
    )
    await app.state.redis.close()
    logger.info("Shutdown complete.")

########################################
# The main subscription for text
########################################
async def listen_to_redis(pubsub):
    logger.info("Listening to Redis channels for text + TTS broadcasting ...")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"]
                raw_str = message["data"]

                # Attempt to parse JSON
                try:
                    data_obj = json.loads(raw_str)
                except json.JSONDecodeError:
                    data_obj = {"text": raw_str}

                if channel == TTS_CHANNEL:
                    # If it's TTS audio data, broadcast only to TTS websockets
                    output = {
                        "channel": channel,
                        "data": data_obj
                    }
                    await tts_manager.broadcast(json.dumps(output))
                else:
                    # For other channels, broadcast to regular text websockets
                    output = {
                        "channel": channel,
                        "data": data_obj
                    }
                    await manager.broadcast(json.dumps(output))
    except asyncio.CancelledError:
        logger.info("listen_to_redis cancelled.")
    except Exception as e:
        logger.error(f"listen_to_redis error: {e}")



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=False)
