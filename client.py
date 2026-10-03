import time, asyncio, httpx, json

URL = "http://127.0.0.1:8000/generate/stream"
PROMPT = "What is a quantum computer ?"

async def one_request(client, prompt, max_tokens) -> dict:
    start = time.perf_counter()
    generation_tokens = None
    finish_reason = None
    ttft = None
    async with client.stream("POST", URL, json={"prompt": prompt, "max_tokens": max_tokens}) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            
            if ttft is None:
                ttft = time.perf_counter() - start
            payload = json.loads(line[len("data:"):])
            if payload["type"] == "done":
                generation_tokens = payload['generation_tokens']
                finish_reason = payload['finish_reason']
            
    return { "ttft": ttft,  
            "total": time.perf_counter() - start, 
            "tokens": generation_tokens, 
            "finish_reason": finish_reason
    }


async def run_concurrency(n, prompt, max_tokens) -> list[dict]:
    async with httpx.AsyncClient(timeout=None) as client:
        start = time.perf_counter()
        results = await asyncio.gather(*[one_request(client, prompt, max_tokens) for i in range(n)])
        batch_wall = time.perf_counter() - start
    return results, batch_wall

def pct(values, p):
    s = sorted(values)
    idx = min(int(p/100 * len(s)), len(s)-1)
    return s[idx]


def stats(n,results, batch_wall):
    ttfts = [r["ttft"] for r in results]
    totals = [r["total"] for r in results]
    tokens = [r["tokens"] for r in results]
    ttft_p50 = pct(ttfts, 50)
    ttft_p95 = pct(ttfts, 95)
    td_p50 = pct(totals, 50)
    td_p95 = pct(totals, 95)
    per_req = [r["tokens"] / r["total"] for r in results]
    mean_prt = sum(per_req) / len(per_req)
    aggregate_tps = sum(tokens) / batch_wall

    return {
        "n":n,
        "ttft_p50":ttft_p50,
        "ttft_p95":ttft_p95,
        "total_p50":td_p50, 
        "total_p95":td_p95, 
        "mean_prt":mean_prt, 
        "aggregate_tps":aggregate_tps}


if __name__ == "__main__":
    
    rows = []
    
    for i in [30,40,50,55,60]:
        results, batch_wall = asyncio.run(run_concurrency(i, PROMPT, 200))
        rows.append(stats(i,results, batch_wall))
        time.sleep(2)
    

    for row in rows:
        print(row)


    
