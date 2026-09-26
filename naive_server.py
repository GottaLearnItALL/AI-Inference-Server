from fastapi import FastAPI, HTTPException
from fastapi.routing import StreamingResponse
from pydantic import BaseModel
import uvicorn
from mlx_lm import load,generate, stream_generate
import time
import json
import asyncio

model, tokenizer = load('mlx-community/Qwen3-4B-Instruct-2507-4bit')


app = FastAPI(title='Qwen FastAPI Wrapper')

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200



@app.post("/generate")
def generate_endpoint(request: PromptRequest):
    server_start = time.perf_counter()
    messages = [{"role": "user", "content": request.prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    start_time = time.perf_counter()
    output = generate(model,tokenizer,prompt=prompt, max_tokens = request.max_tokens)
    end_time = time.perf_counter()

    prompt_tokens = len(prompt)
    output_tokens = len(tokenizer.encode(output))
    

    elapsed_time = end_time - start_time
    server_end  = time.perf_counter()
    server_time = server_end - server_start
    server_overhead = server_time - elapsed_time
    print(f"Elapsed wall-clock time: {elapsed_time:.6f} seconds")
    print(f"Served Overhead wall-clock time: {server_overhead:.6f} seconds")
    return {
        "text": output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "generate_seconds": elapsed_time,
        "server_overhead_seconds": server_overhead,
    }


@app.post('/generate/stream')
def stream_endpoint(request: PromptRequest):
    def event_generator():
        last = None
        messages = [{"role": "user", "content": request.prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        for response in stream_generate(model, tokenizer, prompt, max_tokens=request.max_tokens):
            payload = {"type":"token", "text": response.text, "finish_reason": response.finish_reason}
            yield f"data:{json.dumps(payload)}\n\n"
            last = response
        

        payload = {"type": "done", "prompt_tokens": last.prompt_tokens, "generation_tokens": last.generation_tokens,  "finish_reason": response.finish_reason}
        yield f"data:{json.dumps(payload)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




