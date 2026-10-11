from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ROOTS = ("config", "models", "agents", "llm", "experiments", "world")
SKIP_FILES = {"settings.py"}


def test_sampling_constants_stay_in_settings() -> None:
    offenders: list[str] = []
    for path in _app_files():
        text = path.read_text(encoding="utf-8")
        if "11434" in text or "localhost:11434" in text:
            offenders.append(f"{path.name}:endpoint")
        if re.search(r"""['"]ollama['"]""", text):
            offenders.append(f"{path.name}:api-key")
        if re.search(r"\btemperature\s*=\s*\d", text):
            offenders.append(f"{path.name}:temperature")
        if "OpenAI(" in text and path.name != "client.py":
            offenders.append(f"{path.name}:openai-constructor")
        for snippet in ("os.environ", "getenv", "subprocess", "exec(", "eval(", "while True"):
            if snippet in text:
                offenders.append(f"{path.name}:{snippet}")
    assert offenders == []


def test_only_the_persona_generator_draws_random() -> None:
    hits = []
    for path in _app_files(include_settings=True):
        text = path.read_text(encoding="utf-8")
        if "import random" in text or "random.Random(" in text:
            hits.append(path.relative_to(ROOT).as_posix())
    assert set(hits) == {
        "agents/persona_generator.py",
        "experiments/place_scenarios.py",
        "world/choice.py",
        "world/experiment_004.py",
        "world/experiment_005.py",
        "world/experiment_007.py",
        "world/experiment_008.py",
        "world/experiment_009.py",
        "world/experiment_010.py",
        "world/experiment_011.py",
        "world/history.py",
    }


def test_default_model_name_is_not_copied_into_other_modules() -> None:
    from config.settings import get_settings

    model_name = get_settings().model
    copied = [path.name for path in _app_files() if model_name in path.read_text(encoding="utf-8")]
    assert copied == []


def test_client_source_does_not_pass_tools() -> None:
    source = (ROOT / "llm" / "client.py").read_text(encoding="utf-8")
    assert '"tools"' not in source
    assert "'tools'" not in source
    assert '"functions"' not in source


def test_generator_requests_the_profile_schema_only() -> None:
    source = (ROOT / "agents" / "persona_generator.py").read_text(encoding="utf-8")
    assert source.count("response_model=") == 1
    assert "response_model=BehavioralProfile" in source
    assert "response_model=AgentTraits" not in source


def test_requirements_list_runtime_and_test_dependencies() -> None:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "openai" in text
    assert "pydantic" in text
    assert "pytest" in text


def test_data_directories_exist() -> None:
    assert (ROOT / "data" / "personas").is_dir()
    assert (ROOT / "data" / "logs").is_dir()


def _app_files(*, include_settings: bool = False):
    for folder in APP_ROOTS:
        for path in (ROOT / folder).rglob("*.py"):
            if path.name in SKIP_FILES and not include_settings:
                continue
            if path.name in SKIP_FILES and include_settings:
                yield path
                continue
            yield path
