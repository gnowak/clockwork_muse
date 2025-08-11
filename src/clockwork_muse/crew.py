# src/clockwork_muse/crew.py
from __future__ import annotations
import os, yaml
from pathlib import Path
from jinja2 import Template
from dotenv import load_dotenv, find_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

CFG_AGENTS = "src/clockwork_muse/config/agents.yaml"
CFG_TASKS  = "src/clockwork_muse/config/tasks.yaml"

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _render(s: str, ctx: dict) -> str:
    if not isinstance(s, str):
        return s
    try:
        return Template(s).render(**ctx)
    except Exception:
        return s

class ContentCrew:
    def __init__(self):
        load_dotenv(find_dotenv(usecwd=True), override=False)
        self.agents_cfg = _load_yaml(CFG_AGENTS)
        self.tasks_cfg  = _load_yaml(CFG_TASKS)
        # Tools for researcher
        self.serper  = SerperDevTool() if os.getenv("SERPER_API_KEY") else None
        self.scraper = ScrapeWebsiteTool()

    def _make_agent(self, key: str) -> Agent:
        cfg = self.agents_cfg[key]
        tools = None
        if key == "researcher":
            tools = [t for t in (self.serper, self.scraper) if t]
        return Agent(config=cfg, tools=tools)

    def _make_task(self, key: str, agent: Agent, inputs: dict) -> Task:
        cfg = self.tasks_cfg[key]
        desc = _render(cfg["description"], inputs)
        exp  = _render(cfg.get("expected_output", ""), inputs)
        out  = _render(cfg.get("output_file",""), inputs) or None  # <-- render with inputs
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
        return Task(description=desc, expected_output=exp, agent=agent, output_file=out)

    def build(self, inputs: dict, stage: str = "all") -> Crew:
        agents = {
            "trendhunter": self._make_agent("trendhunter"),
            "researcher" : self._make_agent("researcher"),
            "writer"     : self._make_agent("writer"),
            "editor"     : self._make_agent("editor"),
        }
        stage_map = {
            "all":      ["trend_scan","web_research","outline","script","edit_pass"],
            "research": ["trend_scan","web_research"],
            "outline":  ["outline"],
            "script":   ["script"],
            "edit":     ["edit_pass"],
            "assets":   [],  # fill later
        }
        plan = stage_map.get(stage, stage_map["all"])
        tasks = []
        for name in plan:
            agent_key = self.tasks_cfg[name]["agent"]
            tasks.append(self._make_task(name, agents[agent_key], inputs))
        return Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential, verbose=True)
