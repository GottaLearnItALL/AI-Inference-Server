import asyncio
from dataclasses import dataclass, field
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import mlx_lm
from pydantic import BaseModel
from mlx_lm import load
import json
import time
from contextlib import asynccontextmanager
from typing import Any
from mlx_lm.models.cache import KVCache, make_prompt_cache
import mlx.core as mx




@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(engine_loop())
    yield
    task.cancel()




app = FastAPI(title='Qwen Inference Server', lifespan=lifespan)
model, tokenizer = load('mlx-community/Qwen3-4B-Instruct-2507-4bit')

MAX_BATCH = 64

request_queue: asyncio.Queue = asyncio.Queue()

@dataclass
class Request:
    prompt_tokens: list[int]
    max_tokens: int
    out_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    generated: list[int] = field(default_factory=list)
    cache: Any = None


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


# THE CONTINUOUS BATCHING ENGINE!.
async def engine_loop():
    active = [] # requests curently in the batch
    batch_cache = None # One Batched Cache, persists across iterations

    while True:
        # ---------------------------------------------------------------
        # 1. ADMIT - pull waiting requests off the shared queue.
        new_requests = []
        if not active:
            new_requests.append(await request_queue.get())
        while len(new_requests) + len(active) < MAX_BATCH:
            try:
                new_requests.append(request_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # ---------------------------------------------------------------
        # 2. Prefill -> new arrivals only.
        # Each arrival gets its own cache
        # Runs irs entire prompt and generates the firs token.
        # The only place where req.cache is used.

        for req in new_requests:
            req.cache = make_prompt_cache(model)
            input_ids = mx.array([req.prompt_tokens])
            logits = model(input_ids, cache=req.cache)
            next_logits = logits[:,-1,:]
            token = mx.argmax(next_logits, axis=-1).item()
            req.generated.append(token)
            req.out_queue.put_nowait({
                "type":"token",
                "text": tokenizer.decode([token])
            })
        
        # ---------------------------------------------------------------
        # 3. JOIN - fold the new sequences caches into the batched one.
        # merge() stacks them, extend() appends them to what's already there,
        # handling the left-padding alignment.
        # active and cache rows must change together, right here.

        if new_requests:
            new_cache = [KVCache.merge(layers) for layers in zip(*[r.cache for r in new_requests])]
            if batch_cache is None:
                batch_cache = new_cache
            else:
                for layer, new_layer in zip(batch_cache,new_cache):
                    layer.extend(new_layer)
            active.extend(new_requests)
        
        # ---------------------------------------------------------------
        # 4. DECODE -> One Forward Pass for the whole batch
        # Row i of the output belongs to active[i]. That ordering is the 
        # invariant everything else protects

        if active:
            input_ids = mx.array([[r.generated[-1]] for r in active])
            logits = model(input_ids, cache=batch_cache)
            tokens = mx.argmax(logits[:,-1,:],axis=-1)
            for req, token in zip(active, tokens):
                token = token.item()
                req.generated.append(token)
                if token in tokenizer.eos_token_ids:
                    continue
                req.out_queue.put_nowait({
                    "type":"token",
                    "text": tokenizer.decode([token])
                })
        
        # --------------------------------------------------------------- 
        # 5. Evict - Finished sequences leave. Collect survivor indices
        # in the same pass, then filter() the cache to match. Nothing 
        # leaves active without a done marker first.

        still_active, survivors = [], []
        for i, req in enumerate(active):
            hit_eos = req.generated and req.generated[-1] in tokenizer.eos_token_ids
            hit_limit = len(req.generated) >= req.max_tokens
            if hit_eos or hit_limit:
                req.out_queue.put_nowait({
                    "type":"done",
                    "generation_tokens": len(req.generated),
                    "finish_reason": "stop" if hit_eos else "length"
                })
            else:
                still_active.append(req)
                survivors.append(i)
        if len(still_active) != len(active):
            active = still_active
            if active:
                for layer in batch_cache:
                    layer.filter(survivors)
            else:
                batch_cache = None

        await asyncio.sleep(0)


# THE ENGINE
# async def engine_loop():
#     active = []
#     while True:
#         # Admit
#         if not active:
#             active.append(await request_queue.get())
#         while len(active) < MAX_BATCH:
#             try:
#                 active.append(request_queue.get_nowait())
#             except asyncio.QueueEmpty:
#                 break
        
#         # Active has 2 kinds of sequences: NEW, ONGOING
#         # NEW needs prefill
#         # ONGOING needs decode
#         # Can't do both in one single pass
#         to_prefill = [r for r in active if not r.generated]
#         to_decode = [r for r in active if r.generated]

#         # Prefill Section
#         for req in to_prefill:
#                 req.cache = make_prompt_cache(model)
#                 input_ids = mx.array([req.prompt_tokens]) # shape [1, prompt_len]
#                 logits = model(input_ids, cache=req.cache)
#                 c = req.cache[0]
#                 print(c.keys.shape, c.offset)
#                 print([m for m in dir(c) if not m.startswith('_')])
#                 new_token_logits = logits[:,-1,:] #[batch_size, sequence_length, vocab_size], we get last token
#                 token = mx.argmax(new_token_logits, axis=-1).item()
#                 req.generated.append(token)
#                 req.out_queue.put_nowait({
#                     "type": "token",
#                     "text": tokenizer.decode([token]),
#                 })
        
#         if to_decode:
#             input_ids = mx.array([[r.generated[-1]] for r in to_decode])
#             print(input_ids.shape)
#             batch_cache = KVCache.merge([r.cache for r in to_decode])
#             logits = model(input_ids,cache=batch_cache) # [batch,seq_len,vocab]
#             nxt_logits = logits[:,-1,:] #[batch,vocab]
#             tokens = mx.argmax(nxt_logits,axis=-1)

#             # Sending the tokens to respective requests
#             for req,token in zip(to_decode,tokens):
#                 tok = token.item()
#                 req.generated.append(tok)
#                 if tok in tokenizer.eos_token_ids:
#                     continue
#                 req.out_queue.put_nowait({
#                     "type": "token",
#                     "text": tokenizer.decode([tok])
#                 })


#         # Eviction
#         still_active = []
#         for req in active:
#             hit_eos = req.generated and req.generated[-1] in tokenizer._eos_token_ids
#             hit_limit = len(req.generated) >= req.max_tokens
#             if hit_eos or hit_limit:
#                 req.out_queue.put_nowait({
#                     "type": "done", # Nothing leaves active without a done marker.
#                     "generation_tokens": len(req.generated),
#                     "finish_reason": "stop" if hit_eos else "length",
#                 })
#             else:
#                 still_active.append(req)
#         active = still_active
        # await asyncio.sleep(0)

