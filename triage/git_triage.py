from openai import OpenAI
import json
from triage.db import DB_PATH
import sqlite3

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key='x')


def parse_sections(body):
    
    header_split = body.split('### ')[1:]

    parsed_dict = {}

    for section in header_split:
        header,_,content = section.partition("\n")
        parsed_dict[header.strip()] = content.strip()

    return parsed_dict

WANTED = ["Description", "Error Message and Stack Trace (if applicable)", "Feature Description", "Use Case","Proposed Solution"]
def build_text(title,body):

    parts = [title]
    sections = parse_sections(body)
    
    for name in WANTED:
        content = sections.get(name)
        if content and content != '_No response_':
            parts.append(content)
    if len(parts) == 1:
        parts.append(body[:2000])
    return "\n\n".join(parts)




TYPES = ["bug", "feature request", "documentation"]
COMPONENTS = ["core", "langchain", "openai", "anthropic",
              "text-splitters", "mcp"]


def true_value(labels,vocab):
    hits = [l for l in labels if l in vocab]
    if len(hits) == 1:
        return hits[0]
    return None


# GIT TRIAGE V1

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT number, title, body, labels FROM issues").fetchall()

    SYS_PROMPT = """  You are triaging Github issues for the langchainai/langchain repository.
                
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

    results = []
    COMPONENT_MAP = {
        "langchain-core": "core",
        "langchain": "langchain",
        "langchain-openai": "openai",
        "langchain-anthropic": "anthropic",
        "langchain-text-splitters": "text-splitters",
        "mcp": "mcp",
        "other": "other",
    }
    for number, title, body, labels_json in rows[:10]:
        labels = json.loads(labels_json)
        text = build_text(title, body)
        r = client.chat.completions.create(
            model="qwen",
            messages=[
                {"role":"system","content":SYS_PROMPT},
                {"role":"user","content":text}], max_tokens=200)

        pred = json.loads(r.choices[0].message.content)
        results.append({
            "number": number,
            "pred_type": pred["type"],
            "true_type": true_value(labels, TYPES),
            "pred_comp": COMPONENT_MAP.get(pred["component"],"other"),
            "true_comp": true_value(labels, COMPONENTS),
        })

    print(results)

