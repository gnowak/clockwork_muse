# src/logging_llm.py
from __future__ import annotations
import os, json, time, uuid, requests
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict
from crewai import BaseLLM

class LoggingLLM(BaseLLM):
    """OpenAI-compatible chat client with verbose logging of prompts/responses."""
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
    ):
        super().__init__(model=model, temperature=temperature)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.log_dir = Path(log_dir)
        self.log_json = log_json
        self.echo_to_console = echo_to_console
        self.max_echo_chars = max_echo_chars
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}{path}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def call(
        self,
        messages: List[Dict[str, Any]] | str,
        tools=None,
        callbacks=None,
        available_functions=None,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Prepare and send request
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature or 0.0,
        }
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        token = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        data = self._post("/chat/completions", payload)
        dt = time.perf_counter() - t0

        # Extract text
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""

        # Write logs
        prefix = f"{stamp}-{self.model.replace('/', '_')}-{token}"
        txt_path = self.log_dir / f"{prefix}.txt"
        txt_path.write_text(
            _pretty_llm_log(messages, text, dt),
            encoding="utf-8"
        )
        if self.log_json:
            (self.log_dir / f"{prefix}.json").write_text(
                json.dumps({"request": payload, "response": data}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if self.echo_to_console:
            preview = text[: self.max_echo_chars]
            print(f"\n--- LLM [{self.model}] {prefix} ({dt:.2f}s) ---\n{preview}\n--- /LLM ---\n")

        return text

def _pretty_llm_log(messages, text, seconds):
    """Human-friendly combined prompt/response log."""
    lines = []
    lines.append(f"elapsed: {seconds:.2f}s")
    lines.append("=== PROMPT ===")
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"\n[{role.upper()}]\n{content}")
    lines.append("\n=== RESPONSE ===\n")
    lines.append(text)
    lines.append("\n=== END ===\n")
    return "\n".join(lines)
