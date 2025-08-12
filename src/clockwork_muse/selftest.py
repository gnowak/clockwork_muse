# src/clockwork_muse/selftest.py
from __future__ import annotations
import os, sys, socket, time, json
from typing import Tuple
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)

def _print(status: str, msg: str):
    print(f"[{status}] {msg}")

def check_dns_http(timeout=6) -> Tuple[bool, str]:
    try:
        socket.gethostbyname("google.com")
        import requests
        r = requests.head("https://example.com", timeout=timeout, allow_redirects=True)
        return (200 <= r.status_code < 400, f"HTTP {r.status_code}")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")

def check_serper(timeout=12) -> Tuple[bool, str]:
    key = os.getenv("SERPER_API_KEY")
    if not key:
        return (False, "SERPER_API_KEY not set")
    try:
        import requests
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": "site:wikipedia.org test", "num": 1},
            timeout=timeout,
        )
        if 200 <= r.status_code < 300:
            data = r.json()
            n = len(data.get("organic", []) or data.get("results", []) or [])
            return (n > 0, f"ok, {n} result(s)")
        return (False, f"HTTP {r.status_code}: {r.text[:160]}")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")

def check_scraper(timeout=12) -> tuple[bool, str]:
    try:
        from crewai_tools import ScrapeWebsiteTool
        tool = ScrapeWebsiteTool()
        url = "https://en.wikipedia.org/wiki/Disneyland_Paris"
        try:
            out = tool.run(url=url)  # many versions accept 'url='
        except Exception:
            out = tool.run(website_url=url)  # some expect 'website_url='
        ok = isinstance(out, str) and len(out) > 200
        return (ok, f"{'ok' if ok else 'too short'} ({len(out) if isinstance(out,str) else 'n/a'} chars)")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")

def check_ollama(timeout=6) -> Tuple[bool, str]:
    model = os.getenv("MODEL", "")
    if not model.startswith("ollama/"):
        return (True, "not using ollama")
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=timeout)
        return (200 <= r.status_code < 400, f"HTTP {r.status_code}")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")

def main():
    strict = "--strict" in sys.argv
    results = []

    ok, msg = check_dns_http()
    results.append(("Network", ok, msg)); _print("OK" if ok else "FAIL", f"Network: {msg}")

    ok, msg = check_serper()
    results.append(("Serper", ok, msg)); _print("OK" if ok else "FAIL", f"Serper: {msg}")

    ok, msg = check_scraper()
    results.append(("Scraper", ok, msg)); _print("OK" if ok else "FAIL", f"Scraper: {msg}")

    ok, msg = check_ollama()
    results.append(("Ollama", ok, msg)); _print("OK" if ok else "WARN", f"Ollama: {msg}")

    failed = [name for name, ok, _ in results if not ok]
    summary = {"passed": [n for n, ok, _ in results if ok], "failed": failed}
    _print("SUMMARY", json.dumps(summary))

    sys.exit(1 if strict and failed else 0)

if __name__ == "__main__":
    main()
