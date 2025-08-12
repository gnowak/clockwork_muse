# src/clockwork_muse/main.py
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from clockwork_muse.crew import ContentCrew

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def preflight_tools() -> None:
    # Runs in the SAME process/env as the crew.
    from crewai_tools import SerperDevTool, ScrapeWebsiteTool
    import os

    # Serper
    if not os.getenv("SERPER_API_KEY"):
        raise SystemExit("[ABORT] SERPER_API_KEY not set in this process.")
    s = SerperDevTool()
    q = "site:wikipedia.org test"
    r = s.run(search_query=q)
    if not r:
        raise SystemExit("[ABORT] Serper returned empty response.")

    # Scraper (be explicit about the param name)
    w = ScrapeWebsiteTool()
    out = None
    try:
        out = w.run(website_url="https://example.com")
    except TypeError:
        out = w.run(url="https://example.com")
    if not isinstance(out, str) or len(out) < 100:
        raise SystemExit("[ABORT] Scraper returned too little content.")


def setup_logging(verbose: bool, debug: bool, log_file: str | None) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    if debug:
        os.environ.setdefault("CREWAI_LOG_LEVEL", "DEBUG")
    elif verbose:
        os.environ.setdefault("CREWAI_LOG_LEVEL", "INFO")


parser = argparse.ArgumentParser()
parser.add_argument("--topics", nargs="+", required=True)
parser.add_argument("--channels", nargs="+", default=["yt_shorts"])
parser.add_argument("--stage", choices=["all","research","outline","script","edit","assets"], default="all")
args = parser.parse_args()


inputs = {"topics": args.topics, "channels": args.channels}

if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    for t in args.topics:
        inputs = {
            "topic": t,
            "topics": [t],          # keep compatibility if YAML uses topics
            "channels": args.channels,
            "run_id": run_id,       # optional; used in paths below
        }
        print(f"\n=== Running topic: {t} ===")
        preflight_tools()
        crew = ContentCrew().build(inputs, stage=args.stage)
        result = crew.kickoff(inputs=inputs)
        print(result)