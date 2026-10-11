"""Place-choice decision engine.

The engine receives numerical traits and place attributes only. It does not
read a behavioral profile or any persona-generated prose.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agents.persona_generator import TRAIT_DEFINITIONS
from models.schemas import TRAIT_NAMES, AgentTraits

PLACE_GUIDE = (
    "crowding: density of people at the place, from 0 low to 1 high\n"
    "visual_attractiveness: visual appeal of the place, from 0 low to 1 high\n"
    "distance: travel distance to the place, from 0 near to 1 far\n"
    "novelty: unfamiliarity of the place, from 0 familiar to 1 unfamiliar"
)


def traits_for_condition(traits: AgentTraits, crowd_aversion: float) -> AgentTraits:
    payload = {name: getattr(traits, name) for name in TRAIT_NAMES}
    payload["crowd_aversion"] = crowd_aversion
    return AgentTraits.model_validate(payload)


def build_decision_messages(
    traits: AgentTraits,
    place_a: dict[str, float],
    place_b: dict[str, float],
    *,
    decision_prompt_version: str,
    trait_definitions: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    source = TRAIT_DEFINITIONS if trait_definitions is None else trait_definitions
    missing = [name for name in TRAIT_NAMES if name not in source]
    if missing:
        raise ValueError(f"trait definitions missing {missing}")
    trait_payload = {name: getattr(traits, name) for name in TRAIT_NAMES}
    definitions = "\n".join(f"- {name}: {source[name]}" for name in TRAIT_NAMES)
    system = (
        "You choose one place for a simulated agent.\n"
        "Use only the numerical traits and the two place attribute lists.\n"
        "reason_codes may contain only crowding, visual_attractiveness, distance, novelty, or other.\n"
        "Return JSON only, matching the schema. No markdown.\n"
    )
    user = (
        f"prompt_version: {decision_prompt_version}\n\n"
        "Trait definitions:\n"
        f"{definitions}\n\n"
        "TRAITS_JSON\n"
        f"{_dump(trait_payload)}\n"
        "END_TRAITS_JSON\n\n"
        "Place attribute definitions:\n"
        f"{PLACE_GUIDE}\n\n"
        "PLACE_A\n"
        f"{_dump(place_a)}\n"
        "END_PLACE_A\n\n"
        "PLACE_B\n"
        f"{_dump(place_b)}\n"
        "END_PLACE_B\n\n"
        "Choose A or B.\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def chosen_lower_crowding(choice: str, place_a: dict[str, float], place_b: dict[str, float]) -> int:
    chosen = place_a if choice == "A" else place_b
    other = place_b if choice == "A" else place_a
    if chosen["crowding"] == other["crowding"]:
        raise ValueError("places have equal crowding")
    return int(chosen["crowding"] < other["crowding"])


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
