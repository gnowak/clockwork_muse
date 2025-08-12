# src/clockwork_muse/main.py
from __future__ import annotations

author = "clockwork_muse"
"""
Minimal, reliable runner for the ContentCrew pipeline.
- Supports multiple topics (runs per topic, same run_id)
- Supports stages: all | research | outline | script | edit | assets
- Loads .env (OS env wins) and does a SAME-PROCESS preflight for Serper/Scraper

Usage (Windows cmd):
  .\.venv312\Scripts\activate
  python -m clockwork_muse.main --topics "top books" "travel hacks" --channels yt_shorts --stage research

If your tasks.yaml uses per-topic/per-run templated output paths, this runner will
populate `topic`, `topics`, `channels`, and `run_id` in inputs.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from clockwork_muse.crew import ContentCrew


# -------------------------
# Logging
# -------------------------

def setup_logging(verbose: bool = True, debug: bool = False, log_file: str | None = None) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# -------------------------
# Tool preflight (same process/env as crew)
# -------------------------

def preflight_tools() -> None:
    """Fail fast if Serper/Scraper are unusable in THIS process.

    Only call this if your stage includes research.
    """
    # Ensure .env is loaded in THIS process (OS env still wins)
    load_dotenv(find_dotenv(usecwd=True), override=False)

    import os
    from crewai_tools import SerperDevTool, ScrapeWebsiteTool

    if not os.getenv("SERPER_API_KEY"):
        raise SystemExit("[ABORT] SERPER_API_KEY not set in this process.")

    # Serper: deterministic probe
    s = SerperDevTool()
    r = s.run(search_query='site:wikipedia.org "open source"')
    if not r:
        raise SystemExit("[ABORT] Serper returned empty response.")

    # Scraper: some versions expect website_url, others url
    w = ScrapeWebsiteTool()
    try:
        out = w.run(website_url="https://example.com")
    except TypeError:
        out = w.run(url="https://example.com")
    if not isinstance(out, str) or len(out) < 100:
        raise SystemExit("[ABORT] Scraper returned too little content.")


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--topics", nargs="+", required=True, help="One or more topics (space-separated)")
    p.add_argument("--channels", nargs="+", default=["yt_shorts"], help="Output channels, e.g. yt_shorts instagram_reels")
    p.add_argument("--stage", choices=["all", "research", "outline", "script", "edit", "assets"], default="all")
    p.add_argument("--skip-preflight", action="store_true", help="Skip Serper/Scraper connectivity checks")
    p.add_argument("--log-file", default=None, help="Optional log file path")
    p.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")
    p.add_argument("--debug", action="store_true", help="Debug logging (chatty)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Initialize logging early
    setup_logging(verbose=not args.quiet, debug=args.debug, log_file=args.log_file)

    # Load .env for this process (so preflight + crew see the same values)
    load_dotenv(find_dotenv(usecwd=True), override=False)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    for t in args.topics:
        print(f"\n=== Running topic: {t} ===")

        # Compose inputs for this topic
        inputs = {
            "topic": t,
            "topics": [t],          # keep compatibility if YAML uses topics
            "channels": args.channels,
            "run_id": run_id,
        }

        # Preflight only for research-containing stages
        if args.stage in ("all", "research") and not args.skip_preflight:
            preflight_tools()

        # Build & run the crew for this topic
        cc = ContentCrew()
        crew = cc.build(inputs, stage=args.stage)
        try:
            result = crew.kickoff(inputs=inputs)
        except SystemExit:
            # bubble up preflight aborts cleanly
            raise
        except Exception as e:
            logging.getLogger("clockwork_muse.main").exception("Run failed for topic '%s': %s", t, e)
            return 2

        print(result)

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
