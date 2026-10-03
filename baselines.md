# LLM Inference Server — Baselines

**Model:** mlx-community/Qwen3-4B-Instruct-2507-4bit
**Machine:** Mac Mini M4 (~120 GB/s unified memory bandwidth)
**Prompt:** "Explain what a race condition is."
**max_tokens:** 200 (all runs; every result hits `finish_reason: length`)
**Measurement:** async client, 1 run per config, 2s sleep between configs
**Date:** 2026-09-25

---

## Definitions

- **TTFT** — seconds from request sent to first token received.
- **total** — seconds from request sent to last token received.
- **per-req tok/s** — 200 / total. What one user feels.
- **aggregate tok/s** — (n × 200) / batch wall clock. What the machine delivers.
Aggregate ÷ n = per-req.
- **generation_tokens** — tokens the model produced, including EOS;
not necessarily what the user saw.
- **p95 at n=1 and n=2** is the max of 1 or 2 samples, not a tail estimate.

---



## Config A — mlx-lm `stream_generate`, one loop per request

Threadpool concurrency, no scheduler.


| n   | ttft p50 | total p50 | per-req tok/s | aggregate tok/s |
| --- | -------- | --------- | ------------- | --------------- |
| 1   | 0.80     | 5.60      | 35.7          | 35.7            |
| 2   | 0.49     | 10.32     | 27.6          | 38.7            |
| 5   | 0.75     | 23.00     | 8.7           | 42.6            |
| 10  | 1.49     | 46.76     | 4.3           | 42.6            |


**Finding:** aggregate saturates at ~42 tok/s. Concurrency divides a fixed
pie — n=5 to n=10 gained 0.2% throughput and doubled latency.

**Streaming:** TTFT 0.66s vs 5.15s for the non-streaming endpoint — 7.8×
better perceived latency, same total work.

**Why it saturates:** decode is memory-bound. ~2 GB of weights read per
forward pass, 42 tok/s → ~84 GB/s against ~120 GB/s theoretical ≈ 70% MBU.
MFU is a few percent — compute sits idle. That idle compute is what
batching claims.

Prefill vs decode on the same run: prompt_tps 58 / generation_tps 39.
Prefill does 15 positions per weight read; decode does 1.

---



## Config B — own scheduler, unbatched execution

Continuous batching scheduler, but one model call per request per step.
Control row: isolates scheduler cost from batching benefit. MAX_BATCH=15.


| n   | ttft p50 | total p50 | per-req tok/s | aggregate tok/s |
| --- | -------- | --------- | ------------- | --------------- |
| 1   | 0.41     | 10.42     | 19.2          | 19.2            |
| 2   | 0.26     | 15.30     | 13.1          | 26.1            |
| 5   | 0.38     | 30.91     | 6.5           | 32.3            |
| 10  | 1.14     | 56.18     | 3.6           | 35.6            |


**Finding:** slower than Config A everywhere. Per-token Python overhead —
one model call, argmax, `.item()` sync, and dict construction per sequence
per step. The cost of owning the loop.

**Bug found at MAX_BATCH=8:** n=10 showed ttft p95 of 45.8s against a p50 of
0.70s. Two requests couldn't be admitted and waited for a slot. Raising the
cap to 15 collapsed p95 to 1.14s. The tail was admission, not execution.

---



## Config C — own scheduler, batched execution

One forward pass per step for the whole batch. Persistent `BatchKVCache`,
left-padded and masked per sequence; `extend` on admission, `filter` on
eviction.


| n   | ttft p50 | ttft p95 | total p50 | per-req tok/s | aggregate tok/s |
| --- | -------- | -------- | --------- | ------------- | --------------- |
| 1   | 0.36     | 0.36     | 5.31      | 37.6          | 37.6            |
| 2   | 0.25     | 0.25     | 5.42      | 37.0          | 73.8            |
| 5   | 0.44     | 0.58     | 8.03      | 25.0          | 124.0           |
| 10  | 0.68     | 1.11     | 15.06     | 13.3          | 132.4           |


---



## Result


| n   | A: mlx-lm | B: unbatched | C: batched | C vs A    |
| --- | --------- | ------------ | ---------- | --------- |
| 1   | 35.7      | 19.2         | 37.6       | 1.05×     |
| 2   | 38.7      | 26.1         | 73.8       | 1.91×     |
| 5   | 42.6      | 32.3         | 124.0      | 2.91×     |
| 10  | 42.6      | 35.6         | 132.4      | **3.11×** |


**Headline:** 3.1× aggregate throughput at concurrency 10.

**Latency, not just throughput:** at n=2, per-request total went 5.31s → 5.42s.
A second concurrent user costs ~2% latency instead of doubling it.

**Scheduler is free at n=1:** 37.6 vs mlx-lm's 35.7. Batching recovered the
overhead Config B introduced.

**New ceiling:** n=5 → n=10 gains only 7% aggregate while per-request halves.
Saturation again, now at ~132 instead of ~42. Not yet established whether
the new bound is bandwidth or Python-per-step overhead.

---

---
## Results Batch Size = 64

n	| agg tok/s	| total p50     | per-req tok/s |
----|-----------|---------------|---------------|
25	| 229.8	    |   21.76	    |    9.3        |
30	| 267.8	    |    22.40	    |    9.0        |
40	| 203.8	    |    39.08	    |    5.1        |
50	| 238.1	    |    41.86	    |    4.8        |
55	| 249.5	    |    43.92	    |    4.6        |
60	| 259.2	    |    46.17	    |    4.4        |




## Known gaps

- One run per config; no median-of-3.
- Client disconnect is not handled — the engine keeps generating into a
queue nobody reads.
- Greedy decoding only; no sampling or temperature.
- No KV cache memory ceiling. MAX_BATCH=15 is arbitrary, not derived from
available memory. (Phase 5.)
- Prefill is unbatched — one pass per new request.

