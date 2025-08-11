# src/clockwork_muse/crew.py
import os
import logging
from datetime import datetime
from pathlib import Path
import yaml
from jinja2 import Template
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv, find_dotenv


LOG = logging.getLogger("clockwork_muse.crew")
MAX_CONTEXT_CHARS = 12000


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _render(s: str, ctx: dict) -> str:
    try:
        return Template(s).render(**ctx)
    except Exception:
        return s


def _read(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")[:MAX_CONTEXT_CHARS]


def _write(path: str, text: str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class ContentCrew:
    def __init__(self) -> None:
        load_dotenv(find_dotenv(usecwd=True), override=False)

        # Config (yaml)
        self.agents_cfg = _load_yaml("src/clockwork_muse/config/agents.yaml")
        self.tasks_cfg = _load_yaml("src/clockwork_muse/config/tasks.yaml")

        # Tools
        self.serper = SerperDevTool() if os.getenv("SERPER_API_KEY") else None
        self.scraper = ScrapeWebsiteTool()

        LOG = logging.getLogger("clockwork_muse.crew")

        LOG.info(
            "Serper wired=%s class=%s key_present=%s",
            bool(self.serper),
            type(self.serper).__name__ if self.serper else None,
            bool(os.getenv("SERPER_API_KEY"))
        )

        # Models (allow overrides per role)
        self.model = os.getenv("MODEL", "qwen2.5:7b-instruct")
        self.writer_model = os.getenv("WRITER_MODEL", self.model)

        # Prompt tracing toggle
        self.trace_prompts = str(os.getenv("TRACE_PROMPTS", "0")).lower() in ("1", "true", "yes")

    def _with_model(self, cfg: dict, model: str) -> dict:
        c = dict(cfg)
        c.setdefault("model", model)
        c.setdefault("max_execution_time", 3600)
        c.setdefault("max_iter", 1)
        return c

    def _task(self, key: str, agent: Agent, inputs: dict, extra_context: str = "") -> Task:
        cfg = self.tasks_cfg[key]
        desc = _render(cfg["description"], inputs)
        exp = _render(cfg.get("expected_output", ""), inputs)
        if extra_context:
            desc = f"{desc}\n\nContext:\n{extra_context}"
        if self.trace_prompts:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            _write(f"logs/prompts/{stamp}-{key}-desc.md", desc)
        return Task(description=desc, expected_output=exp, agent=agent, output_file=cfg.get("output_file"))

    def build(self, inputs: dict, stage: str = "all") -> Crew:
        # Agents
        trendhunter = Agent(**self._with_model(self.agents_cfg["trendhunter"], self.model))
        researcher  = Agent(**self._with_model(self.agents_cfg["researcher"],  self.model),
                            tools=[t for t in (self.serper, self.scraper) if t])
        writer      = Agent(**self._with_model(self.agents_cfg["writer"],      self.writer_model))
        editor      = Agent(**self._with_model(self.agents_cfg["editor"],      self.writer_model))

        tasks = []
        plan = []
        if stage in ("all", "research"):
            tasks.append(self._task("trend_scan", trendhunter, inputs)); plan.append("trend_scan")
            tasks.append(self._task("web_research", researcher, inputs)); plan.append("web_research")
        if stage in ("all", "outline"):
            research_md = _read("outputs/research.md")
            tasks.append(self._task("outline", writer, inputs, extra_context=research_md)); plan.append("outline")
        if stage in ("all", "script"):
            ctx = _read("outputs/outline.md") or _read("outputs/research.md")
            if _read("outputs/outline.md"):
                ctx = f"Outline:\n{_read('outputs/outline.md')}\n\nSources:\n{_read('outputs/research.md')}"
            tasks.append(self._task("script", writer, inputs, extra_context=ctx)); plan.append("script")
        if stage in ("all", "edit"):
            script_md = _read("outputs/script.md") or _read("outputs/outline.md")
            tasks.append(self._task("edit_pass", editor, inputs, extra_context=script_md)); plan.append("edit_pass")

        LOG.info("Task plan: %s", plan)
        return Crew(agents=[trendhunter, researcher, writer, editor], tasks=tasks, process=Process.sequential, verbose=True)