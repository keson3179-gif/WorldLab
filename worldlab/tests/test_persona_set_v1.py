from __future__ import annotations

import json
from pathlib import Path

from agents.persona_generator import generate_traits
from experiments.freeze_persona_set import (
    CANONICALIZATION_RULE,
    PERSONA_SET_VERSION,
    freeze_persona_set,
    persona_content_payload,
    select_modal_profile,
)
from experiments.reproducibility_check import canonical_sha256
from tests.support import VALID_PROFILE


def test_modal_profile_uses_the_repeated_profile() -> None:
    first = {**VALID_PROFILE, "summary": "first profile"}
    second = {**VALID_PROFILE, "summary": "modal profile"}
    chosen, ambiguous, count = select_modal_profile([first, second, second])
    assert chosen["summary"] == "modal profile"
    assert ambiguous is False
    assert count == 2


def test_unanimous_profile_is_not_ambiguous() -> None:
    chosen, ambiguous, count = select_modal_profile([VALID_PROFILE, VALID_PROFILE, dict(VALID_PROFILE)])
    assert chosen == VALID_PROFILE
    assert ambiguous is False
    assert count == 3


def test_no_unique_mode_uses_the_first_profile() -> None:
    profiles = [{**VALID_PROFILE, "summary": label} for label in ("one", "two", "three")]
    chosen, ambiguous, count = select_modal_profile(profiles)
    assert chosen["summary"] == "one"
    assert ambiguous is True
    assert count == 1


def test_persona_content_hash_ignores_run_metadata() -> None:
    traits, _ = generate_traits(42, 0)
    trait_payload = traits.model_dump()
    content = persona_content_payload("agent_0000", trait_payload, VALID_PROFILE)
    assert set(content) == {"agent_id", "traits", "profile"}
    again = persona_content_payload("agent_0000", trait_payload, VALID_PROFILE)
    assert canonical_sha256(content) == canonical_sha256(again)
    changed = dict(content)
    changed["run_id"] = "should-not-be-part-of-the-contract"
    assert set(persona_content_payload("agent_0000", trait_payload, VALID_PROFILE)) == {"agent_id", "traits", "profile"}


def test_freeze_writes_manifest_and_python_traits(tmp_path: Path) -> None:
    run_dirs = [_write_source_run(tmp_path / name, name) for name in ("run-a", "run-b", "run-c")]
    output = freeze_persona_set(
        run_dirs,
        tmp_path / "persona_set_v1",
        master_seed=42,
        llm_seed=42,
        model="unit-test-model",
        prompt_version="persona_behavioral_profile_v1",
        source_run_ids=["run-a", "run-b", "run-c"],
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["persona_set_version"] == PERSONA_SET_VERSION
    assert manifest["canonicalization_rule"] == CANONICALIZATION_RULE
    assert manifest["source_run_ids"] == ["run-a", "run-b", "run-c"]
    assert manifest["master_seed"] == 42
    assert manifest["llm_seed"] == 42
    agent = json.loads((output / "agent_0002.json").read_text(encoding="utf-8"))
    expected, trait_seed = generate_traits(42, 2)
    assert agent["traits"] == expected.model_dump()
    assert agent["trait_seed"] == trait_seed
    assert agent["ambiguity_flag"] is False
    assert agent["modal_profile_count"] == 2
    assert agent["behavioral_profile"]["summary"] == "modal"
    assert "run_id" not in agent
    recorded = next(item for item in manifest["agents"] if item["agent_id"] == "agent_0002")
    content = persona_content_payload(agent["agent_id"], agent["traits"], agent["behavioral_profile"])
    assert recorded["persona_content_sha256"] == canonical_sha256(content)
    assert recorded["file_sha256"] != recorded["persona_content_sha256"]


def _write_source_run(run_dir: Path, run_id: str) -> Path:
    run_dir.mkdir()
    for index in range(3):
        traits, trait_seed = generate_traits(42, index)
        summary = "shared" if index != 2 else ("first" if run_id == "run-a" else "modal")
        profile = {**VALID_PROFILE, "summary": summary}
        payload = {
            "agent_id": f"agent_{index:04d}",
            "agent_index": index,
            "run_id": run_id,
            "trait_source": "python_rng",
            "trait_seed": trait_seed,
            "traits": traits.model_dump(),
            "profile_source": "llm",
            "behavioral_profile": profile,
            "prompt_version": "persona_behavioral_profile_v1",
        }
        (run_dir / f"agent_{index:04d}.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir
