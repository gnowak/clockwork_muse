# src/clockwork_muse/tools/serper_tool.py
from __future__ import annotations
import os, json, time, re, requests
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, AliasChoices
from langchain_core.tools import StructuredTool

LOG_DIR = Path("logs/tools"); LOG_DIR.mkdir(parents=True, exist_ok=True)

class SerperArgs(BaseModel):
    # Accept search_query OR query OR q
    search_query: str = Field(
        ...,
        description="Search query string",
        validation_alias=AliasChoices("search_query", "query", "q"),
    )
    num: int = Field(5, ge=1, le=10, description="Number of results (1-10)")

def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)

def _log(text: str) -> None:
    with (LOG_DIR / "serper.log").open("a", encoding="utf-8") as fh:
        fh.write(text)

# tolerant env parsing in case someone wrote "30  # comment"
_num = lambda s, default: (m.group(0) if (m := re.search(r"[-+]?\d+", str(s or ""))) else default)

def serper_search(search_query: str, num: int = 5) -> str:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not set")

    endpoint = os.getenv("SERPER_ENDPOINT", "https://google.serper.dev/search").strip()
    timeout = float(re.search(r"[-+]?\d*\.?\d+", os.getenv("SERPER_TIMEOUT","30") or "30").group(0))
    retries = int(_num(os.getenv("SERPER_RETRIES"), 3))
    backoff = float(re.search(r"[-+]?\d*\.?\d+", os.getenv("SERPER_BACKOFF","1.5") or "1.5").group(0))
    num = max(1, min(10, int(num)))

    payload = {"q": search_query, "num": num}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    last_err = None
    delay = 0.0
    for attempt in range(1, retries + 1):
        try:
            if delay: time.sleep(delay)
            r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if 200 <= r.status_code < 300:
                data = r.json()
                items = data.get("organic", []) or data.get("results", [])
                lines = []
                for it in items[:num]:
                    title = it.get("title") or it.get("name") or it.get("link")
                    link = it.get("link") or it.get("url")
                    snippet = (it.get("snippet") or it.get("summary") or "").strip()
                    if title and link:
                        lines.append(f"- [{title}]({link}) — {snippet}")
                out = "\n".join(lines) or _pretty(data)
                _log(f"\n=== serper ({attempt}) ===\nIN : {_pretty(payload)}\nOUT: {out[:2000]}\n")
                return out
            # retry on 429/5xx
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            last_err = RuntimeError(f"{r.status_code} {r.text[:300]}")
        except Exception as e:
            last_err = e
        delay = (delay or 0.2) * backoff

    _log(f"\n=== serper ERROR ===\nIN : {_pretty(payload)}\nERR: {last_err}\n")
    raise last_err

def make_serper_tool():
    return StructuredTool.from_function(
        func=serper_search,
        name="serper_robust",
        description="Serper search with retries/backoff and logging. Arg: `search_query` (aliases: `query`, `q`).",
        args_schema=SerperArgs,
    )
