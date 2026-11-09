# Qwen Inference Server

MLX-backed Qwen3 inference with continuous batching, plus a GitHub issue triage workflow.

## Layout

```
server/          Inference server (FastAPI + continuous batching engine)
triage/          GitHub issue fetch + LLM triage
benchmarks/      Load tests, TTFT probes, throughput plots
docs/            Notes and baselines
data/            SQLite databases (gitignored)
scratch/         One-off experiments
```

## Run

From the project root:

```bash
uv run uvicorn server.async_server:app --host 127.0.0.1 --port 8000
```

Naive (non-batched) baseline:

```bash
uv run uvicorn server.naive_server:app --host 127.0.0.1 --port 8000
```

## Triage

```bash
uv run python -m triage.db         # create/refresh tables in data/git_triage.db
uv run python -m triage.fetch      # pull issues into data/git_triage.db
uv run python -m triage.git_triage # classify via local OpenAI-compatible API
```

## Benchmarks

```bash
uv run python benchmarks/client.py
uv run python benchmarks/ttft.py
uv run python benchmarks/script.py
```
