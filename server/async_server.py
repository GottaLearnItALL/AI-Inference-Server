import asyncio
import sqlite3
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from contextlib import asynccontextmanager
from server.openai_api import api_router
from server.engine import engine_loop, tokenizer, Request, request_queue
from triage.db import DB_PATH, init_db
from server.jobs_api import job_api



@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(DB_PATH)
    task = asyncio.create_task(engine_loop())
    yield
    task.cancel()




app = FastAPI(title='Qwen Inference Server', lifespan=lifespan)
app.include_router(api_router)
app.include_router(job_api)


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

# Endpoint Async

@app.post("/generate/stream")
async def stream_endpoint(req: PromptRequest):
    async def event_generator():
        messages = [{"role": "user", "content": req.prompt}]
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        request = Request(prompt_tokens=prompt_tokens, max_tokens=req.max_tokens)
        await request_queue.put(request)
        while True:
            item = await request.out_queue.get()
            yield f"data:{json.dumps(item)}\n\n"
            if item['type'] == "done":
                break
    return StreamingResponse(event_generator(), media_type="text/event-stream")

