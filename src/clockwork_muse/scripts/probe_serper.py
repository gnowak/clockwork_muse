from __future__ import annotations
import json, os
from dotenv import load_dotenv, find_dotenv
from crewai_tools import SerperDevTool

def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if not os.getenv("SERPER_API_KEY"):
        print("SERPER_API_KEY not set"); exit(2)
    t = SerperDevTool()
    q = 'site:wikipedia.org "open source"'
    r = t.run(search_query=q)
    if isinstance(r, str):
        print(r[:600])
    else:
        print(json.dumps(r, indent=2)[:600])

if __name__ == "__main__":
    main()
