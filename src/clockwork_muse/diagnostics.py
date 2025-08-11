# src/clockwork_muse/diagnostics.py
import os, json, sys, requests

BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434").rstrip("/")
STYLE = os.getenv("LLM_API_STYLE", "auto").lower()
MODEL = os.getenv("MODEL", "qwen2.5:7b-instruct")
KEY = os.getenv("OPENAI_API_KEY", "")

def say_ok_openai():
    url = f"{BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "say ok"}], "max_tokens": 8}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    return r.status_code, r.text[:400]

def say_ok_ollama_chat():
    url = f"{BASE}/api/chat"
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "say ok"}], "stream": False}
    r = requests.post(url, json=payload, timeout=30)
    return r.status_code, r.text[:400]

def say_ok_ollama_gen():
    url = f"{BASE}/api/generate"
    payload = {"model": MODEL, "prompt": "USER:\nsay ok\n\nASSISTANT:\n", "stream": False}
    r = requests.post(url, json=payload, timeout=30)
    return r.status_code, r.text[:400]

print(f"BASE={BASE} STYLE={STYLE} MODEL={MODEL}")
# Probe in a safe order based on base URL
if BASE.endswith("/v1"):
    print("Trying OpenAI /v1...")
    print(say_ok_openai())
else:
    print("Trying Ollama /api/chat...")
    code, body = say_ok_ollama_chat()
    print(code, body)
    if code == 404:
        print("Falling back to /api/generate...")
        print(say_ok_ollama_gen())