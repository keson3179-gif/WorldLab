"""First pilot: write settings.pilot_num_agents personas and stop.

This study does not start a world simulation.
"""

from __future__ import annotations

from datetime import datetime

from agents.persona_generator import PersonaGenerator, PersonaRunResult
from config.settings import Settings, ensure_python, get_settings
from llm.client import LLMClient, StructuredLLMClient


def run_pilot(
    run_settings: Settings | None = None,
    client: LLMClient | None = None,
    *,
    now: datetime | None = None,
) -> PersonaRunResult:
    ensure_python()
    settings = get_settings() if run_settings is None else run_settings
    llm = client if client is not None else StructuredLLMClient(settings)
    generator = PersonaGenerator(settings, llm)
    return generator.generate_run(num_agents=settings.pilot_num_agents, now=now)


def main() -> None:
    result = run_pilot()
    metadata = result.metadata
    print(f"persona pilot 完成：{metadata.completed_agents} agents")
    print(f"run_id：{metadata.run_id}")
    print(f"model：{metadata.model}")
    print(f"temperature：{metadata.model_parameters.temperature}")
    print(f"llm_seed：{metadata.model_parameters.seed}")
    print(f"reasoning_effort：{metadata.model_parameters.reasoning_effort}")
    print(f"master_seed：{metadata.master_seed}")
    print(f"prompt_version：{metadata.prompt_version}")
    print(f"personas：{result.run_dir}")
    print(f"audit：{result.audit_path}")


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m experiments.persona_pilot")
    main()
