# app.py

import asyncio
import json
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aioredis
import logging
from pydantic import BaseModel
import requests
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional
from transformers import AutoTokenizer

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://zelime.duckdns.org", "https://zelime.duckdns.org/summaries", "localhost:3000/summaries"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSLATION_CHANNEL = "translations"
TRANSCRIPTION_CHANNEL = "transcriptions"
BETTER_TRANSLATION_CHANNEL = "better_translations"  
LLAMA_MODEL_NAME = "ensemble"
TRITON_SERVER_URL = "localhost:5991"



class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
        logger.info(f"Client connected: {websocket.client}")

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"Client disconnected: {websocket.client}")

    async def broadcast(self, message: str):
        async with self.lock:
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to {connection.client}: {e}")
                    await self.disconnect(connection)

manager = ConnectionManager()


class SummaryRequest(BaseModel):
    text: str  # The full text user is seeing

class SummaryResponse(BaseModel):
    summary: str

class TritonLLM(LLM):
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
    tokenizer = AutoTokenizer.from_pretrained("/mnt/nvme512/engines/Meta-Llama-3.1-8B-Instruct")
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            #handle data from client? maybe TODO their mic ? or possibly their v_sink?
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    try:
        app.state.redis = aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
            decode_responses=True,
            max_connections=10  
        )
        app.state.pubsub = app.state.redis.pubsub()

        # ADD "summaries" TO THE LIST HERE
        await app.state.pubsub.subscribe(
            TRANSLATION_CHANNEL,
            TRANSCRIPTION_CHANNEL,
            BETTER_TRANSLATION_CHANNEL,
            "summaries"
        )

        logger.info(f"Subscribed to Redis channels: {TRANSLATION_CHANNEL}, {TRANSCRIPTION_CHANNEL}, {BETTER_TRANSLATION_CHANNEL}, summaries")
        
        app.state.listener_task = asyncio.create_task(listen_to_redis(app.state.pubsub))
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    await manager.broadcast("Server is shutting down.")
    
    app.state.listener_task.cancel()
    try:
        await app.state.listener_task
    except asyncio.CancelledError:
        logger.info("Redis listener task cancelled.")
    
    await app.state.pubsub.unsubscribe(TRANSLATION_CHANNEL, TRANSCRIPTION_CHANNEL, BETTER_TRANSLATION_CHANNEL, "summaries")
    await app.state.redis.close()
    logger.info("Redis connection closed.")

async def listen_to_redis(pubsub):
    logger.info("Listening to Redis channels for messages...")
    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel']
                raw_str = message['data']  # e.g. "{"translation": "..."}" or a raw string

                try:
                    # parse the data if it's valid JSON
                    obj = json.loads(raw_str)
                except json.JSONDecodeError:
                    # fallback if it's not JSON
                    obj = {"text": raw_str}

                # now we always wrap it consistently
                output = {
                    "channel": channel,
                    "data": obj
                }
                await manager.broadcast(json.dumps(output))
    except asyncio.CancelledError:
        logger.info("Redis listener task cancelled.")
    except Exception as e:
        logger.error(f"Error in Redis listener: {e}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=False, workers=1)
