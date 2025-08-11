# src/tools/logging_wrappers.py
from __future__ import annotations
import time, json
from pathlib import Path
from typing import Any
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

LOG_DIR = Path("logs/tools")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_log(name: str, payload: str) -> None:
    (LOG_DIR / f"{name}.log").open("a", encoding="utf-8").write(payload)


class SerperDevToolLogged(SerperDevTool):
    """Serper with simple input/output logging."""
    name: str = "serper_logged"
    description: str = "Serper search with logging"

    def _run(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = super()._run(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            _append_log("serper", f"\n=== serper ({dt:.2f}s) ===\nIN : {args or kwargs}\n")
        # best-effort pretty print
        try:
            pretty = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        except Exception:
            pretty = str(out)
        _append_log("serper", f"OUT: {pretty[:1500]}\n")
        return out


class ScrapeWebsiteToolLogged(ScrapeWebsiteTool):
    """Scraper with simple input/output logging."""
    name: str = "scraper_logged"
    description: str = "Website scraper with logging"

    def _run(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super()._run(*args, **kwargs)
        dt = time.perf_counter() - t0
        # out is usually a big string; truncate
        preview = out[:1500] if isinstance(out, str) else str(out)[:1500]
        _append_log("scraper", f"\n=== scraper ({dt:.2f}s) ===\nIN : {args or kwargs}\nOUT: {preview}\n")
        return out
