# src/crew.py
import os
import yaml
from jinja2 import Template
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

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

class ContentCrew:
    def __init__(self):
        load_dotenv()
        self.agents_cfg = _load_yaml("src/config/agents.yaml")
        self.tasks_cfg = _load_yaml("src/config/tasks.yaml")
        self.serper = SerperDevTool() if os.getenv("SERPER_API_KEY") else None
        self.scraper = ScrapeWebsiteTool()

    def build(self, inputs: dict) -> Crew:
        # Agents
        trendhunter = Agent(**self.agents_cfg["trendhunter"])
        researcher = Agent(
            **self.agents_cfg["researcher"],
            tools=[t for t in (self.serper, self.scraper) if t],
        )
        writer = Agent(**self.agents_cfg["writer"])
        editor = Agent(**self.agents_cfg["editor"])

        # Tasks (render YAML strings with Jinja so {{ topics | join(', ') }} works)
        def mk_task(key: str, agent: Agent) -> Task:
            cfg = self.tasks_cfg[key]
            desc = _render(cfg["description"], inputs)
            exp = _render(cfg.get("expected_output", ""), inputs)
            return Task(
                description=desc,
                expected_output=exp,
                agent=agent,
                output_file=cfg.get("output_file"),
            )

        tasks = [
            mk_task("trend_scan", trendhunter),
            mk_task("web_research", researcher),
            mk_task("outline", writer),
            mk_task("script", writer),
            mk_task("edit_pass", editor),
        ]

        return Crew(
            agents=[trendhunter, researcher, writer, editor],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
