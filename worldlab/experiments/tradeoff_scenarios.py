"""Deterministic place pairs for the crowd-avoidance calibration.

Each pair keeps crowding at 0.25 versus 0.75. Difficulty raises the cost of
the less crowded place. Nothing here is drawn from a random generator.
"""

from __future__ import annotations

from typing import Any

from experiments.reproducibility_check import canonical_sha256
from models.schemas import PlaceAttributes

SCENARIO_ALGORITHM = "tradeoff_calibration.v1"
BLOCKS = ("A", "B", "C", "D")
DIFFICULTY_LEVELS = (1, 2, 3, 4, 5, 6)
SINGLE_GAPS = (5, 15, 30, 45, 60, 75)
COMBINED_GAPS = (2, 8, 14, 22, 30, 38)
LOW_CROWDING = 25
HIGH_CROWDING = 75
BASE_ATTRACTIVENESS = 80
BASE_DISTANCE = 20
BASE_NOVELTY = 80
PROMPT_FIELDS = ("crowding", "visual_attractiveness", "distance", "novelty")


def generate_scenario_matrix() -> dict[str, Any]:
    scenarios = [_scenario(block_index, block, difficulty_index) for block_index, block in enumerate(BLOCKS) for difficulty_index in range(len(DIFFICULTY_LEVELS))]
    payload = {
        "scenario_algorithm": SCENARIO_ALGORITHM,
        "crowding_low": _unit(LOW_CROWDING),
        "crowding_high": _unit(HIGH_CROWDING),
        "single_attribute_gaps": [_unit(gap) for gap in SINGLE_GAPS],
        "combined_gaps": [_unit(gap) for gap in COMBINED_GAPS],
        "scenarios": scenarios,
    }
    _validate(payload)
    payload["matrix_sha256"] = canonical_sha256(payload)
    return payload


def prompt_places(scenario: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Attributes sent to the model. Design labels stay out of the prompt."""

    return (
        {field: scenario["place_a"][field] for field in PROMPT_FIELDS},
        {field: scenario["place_b"][field] for field in PROMPT_FIELDS},
    )


def _scenario(block_index: int, block: str, difficulty_index: int) -> dict[str, Any]:
    gap = COMBINED_GAPS[difficulty_index] if block == "D" else SINGLE_GAPS[difficulty_index]
    low = {
        "crowding": _unit(LOW_CROWDING),
        "visual_attractiveness": _unit(BASE_ATTRACTIVENESS),
        "distance": _unit(BASE_DISTANCE),
        "novelty": _unit(BASE_NOVELTY),
    }
    high = {
        "crowding": _unit(HIGH_CROWDING),
        "visual_attractiveness": _unit(BASE_ATTRACTIVENESS),
        "distance": _unit(BASE_DISTANCE),
        "novelty": _unit(BASE_NOVELTY),
    }
    if block == "A":
        low["visual_attractiveness"] = _unit(BASE_ATTRACTIVENESS - gap)
    elif block == "B":
        low["distance"] = _unit(BASE_DISTANCE + gap)
    elif block == "C":
        low["novelty"] = _unit(BASE_NOVELTY - gap)
    else:
        low["visual_attractiveness"] = _unit(BASE_ATTRACTIVENESS - gap)
        low["distance"] = _unit(BASE_DISTANCE + gap)
        low["novelty"] = _unit(BASE_NOVELTY - gap)
    low_on_a = (block_index + difficulty_index) % 2 == 0
    if low_on_a:
        place_a, place_b, side = low, high, "A"
    else:
        place_a, place_b, side = high, low, "B"
    return {
        "scenario_id": f"item_{block_index * len(DIFFICULTY_LEVELS) + difficulty_index + 1:02d}",
        "tradeoff_block": block,
        "avoidance_cost_level": DIFFICULTY_LEVELS[difficulty_index],
        "sacrificed_gap": _unit(gap),
        "lower_crowding_side": side,
        "place_a": place_a,
        "place_b": place_b,
    }


def _unit(hundredths: int) -> float:
    if not 0 <= hundredths <= 100:
        raise ValueError("place attribute is outside 0 to 1")
    return float(f"{hundredths / 100:.2f}")


def _validate(payload: dict[str, Any]) -> None:
    scenarios = payload["scenarios"]
    if len(scenarios) != 24:
        raise ValueError("calibration matrix must contain 24 scenarios")
    ids = [scenario["scenario_id"] for scenario in scenarios]
    if ids != [f"item_{index:02d}" for index in range(1, 25)]:
        raise ValueError("scenario ids drifted")
    low_on_a = 0
    for scenario in scenarios:
        place_a = scenario["place_a"]
        place_b = scenario["place_b"]
        PlaceAttributes.model_validate(place_a)
        PlaceAttributes.model_validate(place_b)
        low, high = (place_a, place_b) if place_a["crowding"] < place_b["crowding"] else (place_b, place_a)
        if low["crowding"] != payload["crowding_low"] or high["crowding"] != payload["crowding_high"]:
            raise ValueError("crowding pair drifted")
        _validate_block(scenario["tradeoff_block"], low, high, scenario["sacrificed_gap"])
        if scenario["lower_crowding_side"] == "A":
            low_on_a += 1
            if place_a["crowding"] >= place_b["crowding"]:
                raise ValueError("side label does not match crowding")
        elif place_b["crowding"] >= place_a["crowding"]:
            raise ValueError("side label does not match crowding")
    if low_on_a != 12:
        raise ValueError("presentation side is not balanced")
    for block in BLOCKS:
        rows = [scenario for scenario in scenarios if scenario["tradeoff_block"] == block]
        gaps = [scenario["sacrificed_gap"] for scenario in rows]
        if gaps != sorted(gaps) or len(set(gaps)) != 6:
            raise ValueError(f"block {block} difficulty is not strictly increasing")
        sides = [scenario["lower_crowding_side"] for scenario in rows]
        if sides.count("A") != 3:
            raise ValueError(f"block {block} side is not balanced")
    for level in DIFFICULTY_LEVELS:
        rows = [scenario for scenario in scenarios if scenario["avoidance_cost_level"] == level]
        if [scenario["lower_crowding_side"] for scenario in rows].count("A") != 2:
            raise ValueError(f"difficulty {level} side is not balanced")
    single = payload["single_attribute_gaps"]
    combined = payload["combined_gaps"]
    if any(combined[index] >= single[index] for index in range(6)):
        raise ValueError("combined gaps must stay below the single-attribute gaps")


def _validate_block(block: str, low: dict[str, float], high: dict[str, float], gap: float) -> None:
    attractiveness_cost = round(high["visual_attractiveness"] - low["visual_attractiveness"], 2)
    distance_cost = round(low["distance"] - high["distance"], 2)
    novelty_cost = round(high["novelty"] - low["novelty"], 2)
    if block == "A" and not (attractiveness_cost == gap and distance_cost == 0 and novelty_cost == 0):
        raise ValueError("block A must sacrifice attractiveness only")
    if block == "B" and not (distance_cost == gap and attractiveness_cost == 0 and novelty_cost == 0):
        raise ValueError("block B must sacrifice distance only")
    if block == "C" and not (novelty_cost == gap and attractiveness_cost == 0 and distance_cost == 0):
        raise ValueError("block C must sacrifice novelty only")
    if block == "D" and not (attractiveness_cost == gap and distance_cost == gap and novelty_cost == gap):
        raise ValueError("block D must sacrifice all three attributes")
