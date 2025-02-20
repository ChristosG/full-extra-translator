# summary_service.py

import asyncio
import json
import time
from typing import List, Optional

import uvicorn
import aioredis
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis Config
REDIS_HOST = "localhost"
REDIS_PORT = 6379

BETTER_TRANSLATIONS_CHANNEL = "better_translations"
SUMMARIES_CHANNEL = "summaries"

# Speaker logic
SPEAKER_PAUSE_THRESHOLD = 20.0  # seconds of inactivity => new speaker

# We'll store lines in memory
transcript_store = []  # list of dicts: {speaker_id, timestamp, text}
last_timestamp = 0.0
current_speaker = 1

# Create FastAPI app for the /summaries endpoint
app = FastAPI()

# We'll create a single Redis connection
redis = None

#######################################################
# LLM Summarizer
#######################################################
# Very simple: calls an example local Triton or any summarization endpoint
def summarize_text(text: str) -> str:
    """Call an LLM to produce bulletpoint summary."""
    if not text.strip():
        return ""

    # EXAMPLE: local endpoint
    # Replace with your actual summarization model or logic
    url = "http://localhost:5991/v2/models/ensemble/generate"
    payload = {
        "text_input": (
            "You are an AI that summarizes text into bullet points. "
            "Output ONLY bullet points, no disclaimers.\n"
            + text
        ),
        "parameters": {
            "max_tokens": 32256,
            "temperature": 0.0,
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        summary = data.get("text_output", "")
        return summary
    except Exception as e:
        logger.error(f"Failed to summarize: {e}")
        return ""

#######################################################
# Summaries Request Model
#######################################################
class SummaryRequest(BaseModel):
    speaker_ids: Optional[List[int]] = Field(default=None)
    all_speakers: bool = Field(default=False)

#######################################################
# POST /summaries Endpoint
#######################################################
@app.post("/summaries")
async def create_summary(req: SummaryRequest):
    """
    Summarize lines from the requested speakers 
    up to now (in transcript_store).
    """
    if req.all_speakers:
        # gather all lines
        relevant = transcript_store
    else:
        # gather only from the specified speaker_ids
        # if speaker_ids is None or empty, pick everything
        speaker_ids = req.speaker_ids or []
        if not speaker_ids:
            # means no speakers specified => everything
            relevant = transcript_store
        else:
            relevant = [x for x in transcript_store if x["speaker_id"] in speaker_ids]

    # Join the text
    joined_text = "\n".join(x["text"] for x in relevant)
    summary = summarize_text(joined_text)
    if not summary.strip():
        summary = "No summary available (possibly empty text)."

    # We publish to the "summaries" channel so the front-end can pick it up
    message = {
        "speaker_ids": req.speaker_ids or [],
        "bulletpoints": summary
    }
    await redis.publish(SUMMARIES_CHANNEL, json.dumps(message))
    return {"status": "ok", "summary": summary}

#######################################################
# Listen to better_translations
#######################################################
async def listen_better_translations():
    """
    Subscribes to the better_translations channel, 
    identifies speaker changes by time gaps, 
    and stores final lines in transcript_store.
    """
    global last_timestamp, current_speaker
    pubsub = redis.pubsub()
    await pubsub.subscribe(BETTER_TRANSLATIONS_CHANNEL)
    logger.info(f"Subscribed to {BETTER_TRANSLATIONS_CHANNEL} for lines...")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"]
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    obj = {"text": data}

                # expecting {"translation": "...", "finalized": true} from your pipeline
                text = obj.get("translation", "")
                finalized = obj.get("finalized", False)
                if text and finalized:
                    now = time.time()
                    if (now - last_timestamp) > SPEAKER_PAUSE_THRESHOLD:
                        current_speaker += 1
                        logger.info(f"Detected new speaker: {current_speaker}")
                    transcript_store.append({
                        "speaker_id": current_speaker,
                        "timestamp": now,
                        "text": text
                    })
                    last_timestamp = now
                    logger.info(f"Added line for speaker {current_speaker}: {text}")
    except asyncio.CancelledError:
        logger.info("Listener cancelled.")
    except Exception as e:
        logger.error(f"Error in better_translations listener: {e}")

#######################################################
# FastAPI Startup & Shutdown
#######################################################
@app.on_event("startup")
async def startup_event():
    global redis
    redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0", decode_responses=True)
    asyncio.create_task(listen_better_translations())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down summary_service, closing Redis.")
    await redis.close()

#######################################################
# RUN
#######################################################
if __name__ == "__main__":
    uvicorn.run("summary_service:app", host="0.0.0.0", port=8001, reload=False)
