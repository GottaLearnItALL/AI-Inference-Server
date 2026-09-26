import asyncio
from dataclasses import dataclass, field
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from mlx_lm import load
import json
import time
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(engine_loop())
    yield
    task.cancel()




app = FastAPI(title='Qwen Inference Server', lifespan=lifespan)
model, tokenizer = load('mlx-community/Qwen3-4B-Instruct-2507-4bit')

MAX_BATCH = 8

request_queue: asyncio.Queue = asyncio.Queue()

@dataclass
class Request:
    prompt_tokens: list[int]
    max_tokens: int
    out_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    generated: list[int] = field(default_factory=list)


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


# THE ENGINE
async def engine_loop():
    active = []
    while True:
        # Admit
        if not active:
            active.append(await request_queue.get())
        while len(active) < MAX_BATCH:
            try:
                active.append(request_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        # STEP (fake for now)
        # for each req in active: pretend we produced one token
        for req in active:
            req.generated.append(0)
            req.out_queue.put_nowait({"type": "token", "text": "x"})
        time.sleep(0.02)

        # Eviction
        still_active = []
        for req in active:
            if len(req.generated) >= req.max_tokens:
                req.out_queue.put_nowait({
                    "type": "done", # Nothing leaves active without a done marker.
                    "generation_tokens": len(req.generated),
                    "finish_reason": "length",
                })
            else:
                still_active.append(req)
        active = still_active
        await asyncio.sleep(0)
