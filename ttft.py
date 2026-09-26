import time,httpx

URL = "http://127.0.0.1:8000/generate/stream"
BODY = {"prompt": "Explain what a race condition is.", "max_tokens": 200}

start = time.perf_counter()
ttft = None
with httpx.stream("POST", URL, json=BODY, timeout=None) as r:
    for line in r.iter_lines():
        if not line.startswith("data:"):
            continue
        if ttft is None:
            ttft = time.perf_counter() - start
total = time.perf_counter() - start

print(f"TTFT:  {ttft:.4f}s")
print(f"Total: {total:.4f}s")

