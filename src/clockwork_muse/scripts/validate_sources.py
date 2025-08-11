# src/clockwork_muse/scripts/validate_sources.py
from __future__ import annotations
import sys, json, re, requests, pathlib, datetime as dt

def load_json_block(path: str):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S|re.I)
    if not m:  # allow raw JSON too
        m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        raise SystemExit("No JSON block found")
    return json.loads(m.group(1))

def is_alive(url: str) -> tuple[bool,int|None]:
    try:
        r = requests.head(url, timeout=12, allow_redirects=True)
        if r.status_code in (405, 403):  # some sites block HEAD
            r = requests.get(url, timeout=18)
        return (200 <= r.status_code < 400, r.status_code)
    except Exception:
        return (False, None)

def main(path):
    data = load_json_block(path)
    good = []
    for s in data.get("sources", []):
        ok, code = is_alive(s.get("url",""))
        s["http_ok"] = ok
        s["status"]  = code
        if ok:
            good.append(s)
    out = {"topic": data.get("topic"), "sources": good, "dropped": len(data.get("sources", [])) - len(good)}
    p = pathlib.Path(path)
    p.with_suffix(".validated.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Validated: {len(good)} kept, wrote {p.with_suffix('.validated.json')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m clockwork_muse.scripts.validate_sources <path-to-research.md>")
        sys.exit(2)
    main(sys.argv[1])
