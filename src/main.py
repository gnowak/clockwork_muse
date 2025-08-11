# src/main.py
import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from src.crew import ContentCrew


def setup_logging(verbose: bool, debug: bool, log_file: str | None) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Nudge underlying libs (LiteLLM / CrewAI) to be chatty too
    if debug:
        os.environ.setdefault("LITELLM_LOG", "DEBUG")
        os.environ.setdefault("CREWAI_LOG_LEVEL", "DEBUG")
    elif verbose:
        os.environ.setdefault("LITELLM_LOG", "INFO")
        os.environ.setdefault("CREWAI_LOG_LEVEL", "INFO")


parser = argparse.ArgumentParser()
parser.add_argument("--topics", nargs="+", required=True)
parser.add_argument("--channels", nargs="+", default=["yt_shorts"])
parser.add_argument("--stage", choices=["all","research","outline","script","edit"], default="all",
                    help="Run a single stage or the full pipeline")
parser.add_argument("--verbose", action="store_true", help="Chatty logs")
parser.add_argument("--debug", action="store_true", help="Very chatty logs")
parser.add_argument("--log-file", default=None, help="Write logs to this file (e.g., logs/run.log)")
parser.add_argument("--trace-prompts", action="store_true", help="Save rendered prompts/context to logs/prompts/")
args = parser.parse_args()

# Default log file if verbose/debug and none provided
if (args.verbose or args.debug) and not args.log_file:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.log_file = f"logs/run-{args.stage}-{stamp}.log"

setup_logging(args.verbose, args.debug, args.log_file)

# Surface prompt tracing to the crew via env
if args.trace_prompts:
    os.environ["TRACE_PROMPTS"] = "1"

LOG = logging.getLogger("clockwork_muse.main")
inputs = {"topics": args.topics, "channels": args.channels}

if __name__ == "__main__":
    t0 = time.perf_counter()
    LOG.info("Kickoff start stage=%s topics=%s channels=%s", args.stage, args.topics, args.channels)
    crew = ContentCrew().build(inputs, stage=args.stage)
    result = crew.kickoff(inputs=inputs)
    dur = time.perf_counter() - t0
    LOG.info("Kickoff finished stage=%s in %.2fs", args.stage, dur)
    print("=== DONE ===")
    print(result)