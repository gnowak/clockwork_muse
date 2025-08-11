# src/crew.py
import os
import time
import logging
import yaml

from datetime import datetime
from pathlib import Path

#Logging
from src.logging_llm import LoggingLLM
from src.tools.logging_wrappers import SerperDevToolLogged, ScrapeWebsiteToolLogged

from jinja2 import Template
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

LOG = logging.getLogger("clockwork_muse.crew")
MAX_CONTEXT_CHARS = 12000  # keep prompts manageable


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _render(template: str, ctx: dict) -> str:
    if not isinstance(template, str):
        return template
    try:
        return Template(template).render(**ctx)
    except Exception:
        return template


def _read(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    return txt[:MAX_CONTEXT_CHARS]


def _write(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class ContentCrew:
    def __init__(self):
        load_dotenv()
        self.agents_cfg = _load_yaml("src/config/agents.yaml")
        self.tasks_cfg = _load_yaml("src/config/tasks.yaml")

        # Tools
        self.serper = SerperDevToolLogged() if os.getenv("SERPER_API_KEY") else None
        self.scraper = ScrapeWebsiteToolLogged()
        # Wrap with BaseTool-compatible logger
        if self.serper:
            self.serper = LoggingTool(name="serper", description="Serper logger", tool=self.serper)
        if self.scraper:
            self.scraper = LoggingTool(name="scraper", description="Scraper logger", tool=self.scraper)

        # Models (allow overrides per role)
        self.model = os.getenv("MODEL", "qwen2.5:7b-instruct")
        self.writer_model = os.getenv("WRITER_MODEL", self.model)

        # LLM log toggles (envs)
        base = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
        key  = os.getenv("OPENAI_API_KEY", "ollama")
        log_json = os.getenv("LOG_LLM_JSON", "0").lower() in ("1","true","yes")
        echo = os.getenv("LOG_LLM_ECHO", "0").lower() in ("1","true","yes")
        timeout = float(os.getenv("LLM_TIMEOUT", "1200"))  # 20 min default

        self.llm_default = LoggingLLM(
            model=self.model, base_url=base, api_key=key,
            timeout=timeout, log_json=log_json, echo_to_console=echo
        )
        self.llm_writer = LoggingLLM(
            model=self.writer_model, base_url=base, api_key=key,
            timeout=timeout, log_json=log_json, echo_to_console=echo
        )

    def _with_model(self, cfg: dict, model: str) -> dict:
        c = dict(cfg)
        c.setdefault("model", model)
        # Allow long-running steps to cook if you’ve wired a custom LLM w/ long timeout
        if "max_execution_time" not in c:
            c["max_execution_time"] = 3600
        if "max_iter" not in c:
            c["max_iter"] = 1
        return c

    def _mk_task(self, key: str, agent: Agent, inputs: dict, extra_context: str = "") -> Task:
        cfg = self.tasks_cfg[key]
        desc = _render(cfg["description"], inputs)
        exp = _render(cfg.get("expected_output", ""), inputs)
        if extra_context:
            desc = f"{desc}\n\nContext:\n{extra_context}"
        LOG.info(
            "Prepared task=%s -> output_file=%s (desc=%d chars, ctx=%d chars)",
            key, cfg.get("output_file"), len(desc), len(extra_context)
        )
        if self.trace_prompts:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            _write(f"logs/prompts/{stamp}-{key}-desc.md", desc)
        return Task(
            description=desc,
            expected_output=exp,
            agent=agent,
            output_file=cfg.get("output_file"),
        )

    def build(self, inputs: dict, stage: str = "all") -> Crew:
        LOG.info(
            "Building crew stage=%s topics=%s channels=%s",
            stage, inputs.get("topics"), inputs.get("channels")
        )
        LOG.info("Models: default=%s, writer=%s", self.model, self.writer_model)
        LOG.info("Tools: serper=%s, scraper=%s", bool(self.serper), bool(self.scraper))

        # Agents
        trendhunter = Agent(**self._with_model(self.agents_cfg["trendhunter"], self.model), llm=self.llm_default)
        researcher  = Agent(**self._with_model(self.agents_cfg["researcher"],  self.model),
                            tools=[t for t in (self.serper, self.scraper) if t],
                            llm=self.llm_default)
        writer      = Agent(**self._with_model(self.agents_cfg["writer"],      self.writer_model),
                            llm=self.llm_writer)
        editor      = Agent(**self._with_model(self.agents_cfg["editor"],      self.writer_model),
                            llm=self.llm_writer)

        tasks = []
        plan = []

        if stage in ("all", "research"):
            tasks.append(self._mk_task("trend_scan", trendhunter, inputs))
            plan.append("trend_scan")
            tasks.append(self._mk_task("web_research", researcher, inputs))
            plan.append("web_research")

        if stage in ("all", "outline"):
            research_md = _read("outputs/research.md")
            tasks.append(self._mk_task("outline", writer, inputs, extra_context=research_md))
            plan.append("outline")

        if stage in ("all", "script"):
            # Provide both outline + research if available
            ctx = _read("outputs/outline.md")
            if not ctx:
                ctx = _read("outputs/research.md")
            else:
                ctx = f"Outline:\n{ctx}\n\nSources:\n{_read('outputs/research.md')}"
            tasks.append(self._mk_task("script", writer, inputs, extra_context=ctx))
            plan.append("script")

        if stage in ("all", "edit"):
            script_md = _read("outputs/script.md")
            if not script_md:
                script_md = _read("outputs/outline.md")
            tasks.append(self._mk_task("edit_pass", editor, inputs, extra_context=script_md))
            plan.append("edit_pass")

        LOG.info("Task plan: %s", plan)

        crew = Crew(
            agents=[trendhunter, researcher, writer, editor],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
        return crew
