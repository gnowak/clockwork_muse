# src/clockwork_muse/tools/serper_retry.py
from __future__ import annotations
import os, json, time, requests
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import ConfigDict, Field
from crewai_tools import SerperDevTool

LOG_DIR = Path("logs/tools"); LOG_DIR.mkdir(parents=True, exist_ok=True)

class SerperDevToolRobust(SerperDevTool):
    """Serper tool with explicit timeouts, retries, and file logging.
    Pydantic‑safe: all attributes are declared as fields to avoid
    "object has no field" errors.
    Accepts either a string (query) or a dict payload compatible with Serper.
    """
    # BaseTool/Pydantic model fields
    name: str = "serper_robust"
    description: str = "Serper search with retries/backoff and logging"

    # NEW: declare these as fields so Pydantic allows them
    api_key: Optional[str] = Field(default=None, description="Serper API key")
    timeout: float = Field(default=30.0, description="Per-call timeout in seconds")
    retries: int = Field(default=3, description="Number of attempts per request")
    backoff: float = Field(default=1.5, description="Backoff multiplier between retries")
    endpoint: str = Field(default="https://google.serper.dev/search", description="Serper endpoint")

    # allow our extra fields even if parent forbids extras
    model_config = ConfigDict(extra="allow")

    def model_post_init(self, __context):  # pydantic v2 hook, runs after validation
        # Pull from env if missing (do not mutate unknown attributes)
        if not self.api_key:
            env_key = os.getenv("SERPER_API_KEY")
            if env_key:
                object.__setattr__(self, "api_key", env_key)
        # Fill tunables from env if present
        t = os.getenv("SERPER_TIMEOUT");   
        r = os.getenv("SERPER_RETRIES");   
        b = os.getenv("SERPER_BACKOFF");   
        e = os.getenv("SERPER_ENDPOINT")
        if t: object.__setattr__(self, "timeout", float(t))
        if r: object.__setattr__(self, "retries", int(r))
        if b: object.__setattr__(self, "backoff", float(b))
        if e: object.__setattr__(self, "endpoint", e.strip())
        # Final guard
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
            return {"q": params, "num": 5}
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
    # CrewAI calls .run() -> _run()
    def _run(self, *args, **kwargs) -> str:
        params = None
        if args and not kwargs:
            params = args[0]
        elif kwargs:
            params = kwargs
        else:
            raise ValueError("Serper tool requires a query")

        payload = self._normalize(params)
        t0 = time.perf_counter()
        try:
            data = self._http(payload)
            dt = time.perf_counter() - t0
            # Log inputs/outputs (truncated)
            self._log(f"\n=== serper_robust ({dt:.2f}s) ===\nIN : {self._pretty(payload)}\nOUT: {self._pretty(data)[:2000]}\n")
            # Return compact markdown the agents can consume
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
            self._log(f"\n=== serper_robust ERROR ({dt:.2f}s) ===\nIN : {self._pretty(payload)}\nERR: {e}\n")
            raise
