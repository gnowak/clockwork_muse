# src/clockwork_muse/logging_llm.py
#
# Dual‑API LLM shim for CrewAI with:
# - OpenAI‑compatible (/v1) **and** native Ollama (/api) support
# - Fallback to /api/generate if /api/chat is 404 (older Ollama)
# - Verbose file logging + optional JSON payload capture
# - Accepts extra kwargs CrewAI may pass (from_task, max_tokens, etc.)
#
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
from crewai import BaseLLM


# -----------------
# helpers
# -----------------

def _pretty_llm_log(messages: List[Dict[str, Any]], text: str, seconds: float) -> str:
    lines: List[str] = []
    lines.append(f"elapsed: {seconds:.2f}s")
    lines.append("=== PROMPT ===")
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        lines.append(f"\n[{role}]\n{content}")
    lines.append("\n=== RESPONSE ===\n")
    lines.append(text)
    lines.append("\n=== END ===\n")
    return "\n".join(lines)


def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    # Build a simple chat transcript for /api/generate
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        if isinstance(content, list):
            # Best effort join for multimodal shapes
            content = "\n".join([str(c.get("text", c)) for c in content])
        parts.append(f"{role}:\n{content}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


# -----------------
# LLM
# -----------------
class LoggingLLM(BaseLLM):
    """OpenAI‑compatible / Ollama‑native chat client with verbose logging.

    Env toggles (recommended):
      OPENAI_API_BASE = http://localhost:11434        # native
      LLM_API_STYLE   = ollama                        # or 'openai'
      OPENAI_API_KEY  = ollama                        # only for openai style
      MODEL           = qwen2.5:7b-instruct
      LOG_LLM_JSON    = 0/1
      LOG_LLM_ECHO    = 0/1
      LLM_TIMEOUT     = 600..3600
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float | None = 0.2,
        timeout: float | None = 600.0,
        log_dir: str = "logs/llm",
        log_json: bool = False,
        echo_to_console: bool = False,
        max_echo_chars: int = 1200,
        api_style: str | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout or 600.0)

        style = (api_style or os.getenv("LLM_API_STYLE", "")).strip().lower()
        if not style:
            style = "openai" if self.base_url.endswith("/v1") else "ollama"
        if style not in ("openai", "ollama"):
            raise ValueError(f"Unsupported api_style: {style}")
        self.api_style = style

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_json = bool(log_json)
        self.echo_to_console = bool(echo_to_console)
        self.max_echo_chars = int(max_echo_chars)

    # --------------- HTTP ---------------
    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_style == "openai" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # --------------- CrewAI entrypoint ---------------
    def call(self, messages: List[Dict[str, Any]] | str, tools=None, callbacks=None, available_functions=None, **kwargs: Any) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        temperature = kwargs.get("temperature", self.temperature or 0.0)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        token = uuid.uuid4().hex[:8]

        # ---------- Native Ollama path ----------
        if self.api_style == "ollama":
            # First try /api/chat
            chat_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {},
            }
            if kwargs.get("max_tokens") is not None:
                chat_payload["options"]["num_predict"] = int(kwargs["max_tokens"])  # cap tokens
            if temperature is not None:
                chat_payload["options"]["temperature"] = float(temperature)
            if kwargs.get("stop") is not None:
                chat_payload["options"]["stop"] = kwargs["stop"]
            if kwargs.get("top_p") is not None:
                chat_payload["options"]["top_p"] = float(kwargs["top_p"])  # nucleus
            if kwargs.get("seed") is not None:
                chat_payload["options"]["seed"] = int(kwargs["seed"])  # reproducibility

            t0 = time.perf_counter()
            try:
                data = self._post_json(f"{self.base_url}/api/chat", chat_payload)
                dt = time.perf_counter() - t0
                text = (data.get("message", {}) or {}).get("content") or data.get("response") or ""
                prefix = f"{stamp}-{self.model.replace('/', '_')}-{token}"
                (self.log_dir / f"{prefix}.txt").write_text(_pretty_llm_log(messages, text, dt), encoding="utf-8")
                if self.log_json:
                    (self.log_dir / f"{prefix}.json").write_text(json.dumps({"request": chat_payload, "response": data}, ensure_ascii=False, indent=2), encoding="utf-8")
                if self.echo_to_console:
                    print(f"\n--- LLM [{self.model}] {prefix} ({dt:.2f}s) [ollama/chat] ---\n{text[:self.max_echo_chars]}\n--- /LLM ---\n")
                return text
            except requests.HTTPError as e:
                # Fallback for older Ollama without /api/chat
                if e.response is None or e.response.status_code != 404:
                    raise
                # Fall through to /api/generate

            # Fallback: /api/generate
            prompt = _messages_to_prompt(messages)
            gen_payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {},
            }
            if kwargs.get("max_tokens") is not None:
                gen_payload["options"]["num_predict"] = int(kwargs["max_tokens"])  # cap tokens
            if temperature is not None:
                gen_payload["options"]["temperature"] = float(temperature)
            if kwargs.get("stop") is not None:
                gen_payload["options"]["stop"] = kwargs["stop"]
            if kwargs.get("top_p") is not None:
                gen_payload["options"]["top_p"] = float(kwargs["top_p"])  # nucleus
            if kwargs.get("seed") is not None:
                gen_payload["options"]["seed"] = int(kwargs["seed"])  # reproducibility

            t1 = time.perf_counter()
            data = self._post_json(f"{self.base_url}/api/generate", gen_payload)
            dt = time.perf_counter() - t1
            text = data.get("response") or (data.get("message", {}) or {}).get("content") or ""

            prefix = f"{stamp}-{self.model.replace('/', '_')}-{token}"
            (self.log_dir / f"{prefix}.txt").write_text(_pretty_llm_log(messages, text, dt), encoding="utf-8")
            if self.log_json:
                (self.log_dir / f"{prefix}.json").write_text(json.dumps({"request": gen_payload, "response": data}, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.echo_to_console:
                print(f"\n--- LLM [{self.model}] {prefix} ({dt:.2f}s) [ollama/generate] ---\n{text[:self.max_echo_chars]}\n--- /LLM ---\n")
            return text

        # ---------- OpenAI‑compatible path ----------
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        for k in ("max_tokens", "stop", "top_p", "frequency_penalty", "presence_penalty", "n", "response_format", "tool_choice", "tools", "seed"):
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        t2 = time.perf_counter()
        data = self._post_json(f"{self.base_url}/chat/completions", payload)
        dt = time.perf_counter() - t2
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""

        prefix = f"{stamp}-{self.model.replace('/', '_')}-{token}"
        (self.log_dir / f"{prefix}.txt").write_text(_pretty_llm_log(messages, text, dt), encoding="utf-8")
        if self.log_json:
            (self.log_dir / f"{prefix}.json").write_text(json.dumps({"request": payload, "response": data}, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.echo_to_console:
            print(f"\n--- LLM [{self.model}] {prefix} ({dt:.2f}s) [openai] ---\n{text[:self.max_echo_chars]}\n--- /LLM ---\n")
        return text
