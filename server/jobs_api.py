from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from uuid import uuid4
import sqlite3
from triage.db import DB_PATH
from datetime import datetime,timezone
import asyncio
from triage.git_triage import build_text
from openai import AsyncOpenAI
import json
from typing import Literal

MAX_ATTEMPTS = 3
ALLOWED_COMPONENTS = ["langchain-core", "langchain", "langchain-openai",
                        "langchain-anthropic", "langchain-text-splitters",
                        "mcp", "other"]
SYS_PROMPT =  """  You are triaging Github issues for the langchainai/langchain repository.
                
                Return ONLY in JSON object with these exact field:
                
                "type":      one of ["bug", "feature request", "documentation"]
                "component": one of ["langchain-core", "langchain", "langchain-openai",
                        "langchain-anthropic", "langchain-text-splitters",
                        "mcp", "other"]
                "severity":  one of ["low", "medium", "high"]
                "summary":   one sentence describing the issue

                No markdown, no code fences, no explanation. JSON only.

                Choose the component from the package in the import paths and file paths,
    not from words that appear in error messages.

    Here are more details about each components:

    langchain-core: base abstractions — Runnable, messages, prompts, callbacks, tracers
    langchain: agents, middleware, chains, high-level APIs built on core
    langchain-openai / langchain-anthropic: provider chat model integrations
    langchain-text-splitters: document splitting utilities
    """
client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key='x')

job_api = APIRouter()

class JobRequest(BaseModel):
    issue_numbers: list[int] | None = None
    config: str | None = None

class TriageResult(BaseModel):
    type: Literal["bug", "feature request", "documentation"]
    component: Literal["langchain-core", "langchain", "langchain-openai",
                        "langchain-anthropic", "langchain-text-splitters",
                        "mcp", "other"]
    severity: Literal["low", "medium", "high"]
    summary: str


@job_api.get("/v1/jobs/{job_id}")
async def get_job(job_id:str):
    with sqlite3.connect(DB_PATH) as conn:
        fetch_job = "SELECT total FROM jobs WHERE id = ?"
        row = conn.execute(fetch_job,(job_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Job Does Not Exist")
        fetch_items = """SELECT status, COUNT(*) FROM items 
        WHERE job_id=? GROUP BY status"""
        total = row[0]
        count = conn.execute(fetch_items, (job_id,)).fetchall()
        conn.commit()
        count = dict(count)
    return {
        "job_id": job_id,
        "total": total,
        "counts": 
            {s: count.get(s,0) for s in ("pending", "running", "done","failed")}
    }


@job_api.get("/v1/jobs/{job_id}/results")
async def job_results(job_id:str):
    with sqlite3.connect(DB_PATH) as conn:
        fetch_job = "SELECT 1 FROM jobs WHERE id=?"
        row = conn.execute(fetch_job, (job_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Job Does Not Exist")
        fetch_result = "SELECT issue_number,result FROM items WHERE job_id=? and status='done'"
        rows = conn.execute(fetch_result, (job_id,)).fetchall()
    return [{"issue_number":n, **json.loads(result)} for n,result in rows]

async def run_job(job_id,n_workers=10):
    await asyncio.gather(*[worker(job_id) for _ in range(n_workers)])

@job_api.post("/v1/jobs")
async def queue_jobs(data: JobRequest):
    job_id = "job_" + uuid4().hex[:12]
    with sqlite3.connect(DB_PATH) as conn:
        if not data.issue_numbers:
            numbers_query = "SELECT number FROM issues"
            numbers = [r[0] for r in conn.execute(numbers_query).fetchall()]
            if not numbers:
                raise HTTPException(422, "no issues table in database run uv run python -m triage.fetch")
        else:
            numbers = data.issue_numbers
            known = {r[0] for r in conn.execute("SELECT number FROM issues")}
            missing = [n for n in numbers if n not in known]
            if missing:
                raise HTTPException(status_code=422, detail=f"unknown issues: {missing[:10]}")
        insert_query = "INSERT INTO jobs (id, created_at, total, config) VALUES (?,?,?,?)"
        created_at = datetime.now(timezone.utc).isoformat()
        total = len(numbers)
        config = data.config
        conn.execute(insert_query, (job_id, created_at, total, config))

        items_insert_query = "INSERT INTO items (job_id, issue_number) VALUES (?,?)"
        conn.executemany(items_insert_query, [(job_id,n) for n in numbers])        
        conn.commit()
    asyncio.create_task(run_job(job_id))
    return {"job_id":job_id, "status":"queued","total":total}


async def worker(job_id):
    conn = sqlite3.connect(DB_PATH)
    
    while True:
        # 1. CLAIM: find one pending issue_number for this job    
        fetch_issue_number = "SELECT issue_number FROM items WHERE status='pending' AND job_id=? LIMIT 1"
        row = conn.execute(fetch_issue_number, (job_id,)).fetchone()
        if row is None:
            break
        issue_number = row[0]

        conn.execute(
        "UPDATE items SET status='running', started_at=:t WHERE job_id=:j AND issue_number=:n",
        {"t": datetime.now(timezone.utc).isoformat(), "j": job_id, "n": issue_number},
)
        conn.commit()
    
        # 2. LOOK UP: title, body FROM issues WHERE number=?
        fetch_issue_query = "SELECT title, body FROM issues WHERE number=?"
        issue = conn.execute(fetch_issue_query, (issue_number,)).fetchone()
        if issue is None:
            conn.execute(
                "UPDATE items SET status='failed', error=:e, finished_at=:t "
                "WHERE job_id=:j AND issue_number=:n",
                {
                    "e": f"issue {issue_number} not found in issues table — run: uv run python -m triage.fetch",
                    "t": datetime.now(timezone.utc).isoformat(),
                    "j": job_id,
                    "n": issue_number,
                },
            )
            conn.commit()
            continue

        title, body = issue
        raw = None
        try:
            # 3. PROMPT: text = build_text(title, body)
            text = build_text(title, body)

            # 4. CALL: r = await client.chat.completions.create(...)
            r = await client.chat.completions.create(
                model="Qwen",
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            raw = r.choices[0].message.content
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            parsed = json.loads(raw)
            if parsed.get("component") not in ALLOWED_COMPONENTS:
                parsed["component"] = "other"
            TriageResult.model_validate(parsed)
            # 5. WRITE BACK: UPDATE items SET status='done', result=?, finished_at=?
            conn.execute(
                "UPDATE items SET status='done', result=:r, finished_at=:t "
                "WHERE job_id=:j AND issue_number=:n",
                {
                    "r": json.dumps(parsed),
                    "t": datetime.now(timezone.utc).isoformat(),
                    "j": job_id,
                    "n": issue_number,
                },
            )
            conn.commit()
        except Exception as e:
            print(e)
            fetch_attempts = "SELECT attempts FROM items WHERE issue_number=? AND job_id=?"
            attempts = int(conn.execute(fetch_attempts, (issue_number,job_id)).fetchone()[0]) 
            attempts+=1
            if attempts < MAX_ATTEMPTS:
                conn.execute(
                    "UPDATE items SET status='pending',attempts=:a, error=:e "
                    "WHERE job_id=:j AND issue_number=:n",
                    {"a": attempts, "e": str(e), "j": job_id, "n": issue_number},
                )
            else:
                conn.execute("UPDATE items SET status='failed', result=:r, attempts=:a, error=:e "
                    "WHERE job_id=:j AND issue_number=:n",
                    {"r": raw, "a": attempts, "e": str(e), "j": job_id, "n": issue_number},)
            conn.commit()
    conn.close()

