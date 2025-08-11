# src/clockwork_muse/tools/serper_retry.py
from __future__ import annotations
import os, json, time, requests
from pathlib import Path
from typing import Any, Dict, Optional, ClassVar
from pydantic import BaseModel, ConfigDict, Field, AliasChoices

from crewai_tools import SerperDevTool
import re

LOG_DIR = Path("logs/tools"); LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_EXCLUDE_TERMS = [
    "virtual tour", "360", "vr", "promo", "advertisement",
    "trailer", "4k", "walkthrough", "pov", "ride pov"
]
DEFAULT_EXCLUDE_DOMAINS = [
    "pinterest.com", "x.com", "twitter.com"
]

def _env_float(key, default): 
    v=os.getenv(key); 
    if v is None: return default
    m=re.search(r'[-+]?\d*\.?\d+', v); return float(m.group(0)) if m else default
def _env_int(key, default):
    v=os.getenv(key); 
    if v is None: return default
    m=re.search(r'[-+]?\d+', v); return int(m.group(0)) if m else default

def model_post_init(self, _):
    ...
    object.__setattr__(self, "timeout", _env_float("SERPER_TIMEOUT", self.timeout))
    object.__setattr__(self, "retries", _env_int("SERPER_RETRIES", self.retries))
    object.__setattr__(self, "backoff", _env_float("SERPER_BACKOFF", self.backoff))

def _build_filtered_query(q: str) -> str:
    # allow overrides via env, comma-separated
    extra_terms = [t.strip() for t in os.getenv("SEARCH_EXCLUDE_TERMS", "").split(",") if t.strip()]
    extra_domains = [d.strip() for d in os.getenv("SEARCH_EXCLUDE_DOMAINS", "").split(",") if d.strip()]

    excludes = DEFAULT_EXCLUDE_TERMS + extra_terms
    exclude_domains = DEFAULT_EXCLUDE_DOMAINS + extra_domains

    parts = [q]
    # prefer “tips/guide/how to” intent if not already there
    if not any(k in q.lower() for k in ("tip", "guide", "how to", "mistakes", "things to know")):
        parts.append('(tips OR guide OR "how to")')
    for t in excludes:
        parts.append(f'-"{t}"')
    for d in exclude_domains:
        parts.append(f"-site:{d}")
    return " ".join(parts)


class SerperRobustSchema(BaseModel):
    # Accept search_query, query, or q
    search_query: str = Field(
        ...,
        description="Search query string",
        validation_alias=AliasChoices("search_query", "query", "q"),
    )

class SerperDevToolRobust(SerperDevTool):
    """Serper tool with explicit timeouts, retries, and file logging.
    Accepts search_query/query/q as the input field.
    """
    name: str = "serper_robust"
    description: str = (
        "Serper search with retries/backoff and logging. "
        "Input arg: `search_query` (aliases: `query`, `q`)."
    )
    args_schema: ClassVar[type[BaseModel]] = SerperRobustSchema

    # Pydantic-safe fields
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    api_key: str | None = Field(default=None, description="Serper API key")
    timeout: float = Field(default=30.0, description="Per-call timeout in seconds")
    retries: int = Field(default=3, description="Number of attempts per request")
    backoff: float = Field(default=1.5, description="Backoff multiplier between retries")
    endpoint: str = Field(default="https://google.serper.dev/search", description="Serper endpoint")

    def model_post_init(self, __context):
        # Fill from env if not provided
        if not self.api_key:
            env_key = os.getenv("SERPER_API_KEY")
            if env_key:
                object.__setattr__(self, "api_key", env_key)
        t = os.getenv("SERPER_TIMEOUT");   r = os.getenv("SERPER_RETRIES")
        b = os.getenv("SERPER_BACKOFF");  e = os.getenv("SERPER_ENDPOINT")
        if t: object.__setattr__(self, "timeout", float(t))
        if r: object.__setattr__(self, "retries", int(r))
        if b: object.__setattr__(self, "backoff", float(b))
        if e: object.__setattr__(self, "endpoint", e.strip())
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY is not set")

    # ---------------- internal helpers ----------------
    def _log(self, text: str) -> None:
        with (LOG_DIR / "serper.log").open("a", encoding="utf-8") as fh:
            fh.write(text)

    def _http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        last_err: Optional[Exception] = None
        delay = 0.2
        for attempt in range(1, self.retries + 1):
            try:
                if delay:
                    time.sleep(delay)
                r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
                if 200 <= r.status_code < 300:
                    return r.json()
                # Retry on 429/5xx; otherwise raise immediately
                if r.status_code not in (429, 500, 502, 503, 504):
                    r.raise_for_status()
                last_err = requests.HTTPError(f"{r.status_code} {r.text[:300]}")
            except Exception as e:  # network timeouts etc.
                last_err = e
            delay = delay * self.backoff
        assert last_err is not None
        raise last_err

    def _normalize(self, params: Any) -> Dict[str, Any]:
        # Allow run({"q": "..."}) or run("...")
        if isinstance(params, str):
            return {"q": _build_filtered_query(params), "num": 5, "tbs": "qdr:y"}
        if isinstance(params, dict):
            p = {k: v for k, v in params.items()}
            p.setdefault("num", 5)
            return p
        raise ValueError(f"Unsupported params type for Serper: {type(params)}")

    def _pretty(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return str(obj)

    # ---------------- CrewAI entrypoint ----------------
    def _run(self, search_query: str, **_kwargs) -> str:
        q = search_query.strip()
        payload = self._normalize(q)

        t0 = time.perf_counter()
        try:
            data = self._http(payload)
            dt = time.perf_counter() - t0
            self._log(
                f"\n=== serper_robust ({dt:.2f}s) ===\nIN : {self._pretty(payload)}\nOUT: {self._pretty(data)[:2000]}\n"
            )
            items = data.get("organic", []) or data.get("results", [])
            lines = []
            for it in items[: payload.get("num", 5)]:
                title = it.get("title") or it.get("name") or it.get("link")
                link = it.get("link") or it.get("url")
                snippet = (it.get("snippet") or it.get("summary") or "").strip()
                if title and link:
                    lines.append(f"- [{title}]({link}) — {snippet}")
            return "\n".join(lines) or json.dumps(data)
        except Exception as e:
            dt = time.perf_counter() - t0
            self._log(
                f"\n=== serper_robust ERROR ({dt:.2f}s) ===\nIN : {self._pretty(payload)}\nERR: {e}\n"
            )
            raise