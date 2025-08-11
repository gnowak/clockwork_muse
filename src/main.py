# src/main.py
import argparse
from src.crew import ContentCrew

parser = argparse.ArgumentParser()
parser.add_argument("--topics", nargs="+", required=True)
parser.add_argument("--channels", nargs="+", default=["yt_shorts"])
args = parser.parse_args()

inputs = {"topics": args.topics, "channels": args.channels}

if __name__ == "__main__":
    crew = ContentCrew().build(inputs)
    result = crew.kickoff(inputs=inputs)
    print("\n\n=== DONE ===\n")
    print(result)
