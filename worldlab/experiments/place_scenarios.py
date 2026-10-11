"""Fixed-seed place-choice scenarios for Experiment 001.

The lower-crowding place is never better on every other attribute at once.
Which side is Place A is shuffled so the two orders are balanced.
"""

from __future__ import annotations

import random
from typing import Any

from experiments.reproducibility_check import canonical_sha256
from models.schemas import PlaceAttributes

SCENARIO_ALGORITHM = "place_scenarios.v1"
PLACE_FIELDS = ("crowding", "visual_attractiveness", "distance", "novelty")


def generate_scenario_matrix(
    *,
    scenario_seed: int,
    scenario_count: int,
    crowding_min_gap: float,
) -> dict[str, Any]:
    if scenario_count < 2 or scenario_count % 2 != 0:
        raise ValueError("scenario_count must be an even number of at least 2")
    rng = random.Random(scenario_seed)
    drafts = _draw_unordered_pairs(rng, scenario_count, crowding_min_gap)
    low_on_a = ["A"] * (scenario_count // 2) + ["B"] * (scenario_count // 2)
    rng.shuffle(low_on_a)
    scenarios = []
    for index, ((low, high), side) in enumerate(zip(drafts, low_on_a, strict=True)):
        if side == "A":
            place_a, place_b = low, high
        else:
            place_a, place_b = high, low
        scenarios.append(
            {
                "scenario_id": f"scenario_{index:03d}",
                "lower_crowding_side": side,
                "place_a": place_a,
                "place_b": place_b,
            }
        )
    payload = {
        "scenario_algorithm": SCENARIO_ALGORITHM,
        "scenario_seed": scenario_seed,
        "crowding_min_gap": crowding_min_gap,
        "scenarios": scenarios,
    }
    payload["matrix_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "matrix_sha256"}
    )
    return payload


def generate_scenario_matrix_from_settings(settings: Any) -> dict[str, Any]:
    return generate_scenario_matrix(
        scenario_seed=settings.scenario_seed,
        scenario_count=settings.scenario_count,
        crowding_min_gap=settings.crowding_min_gap,
    )


def _draw_unordered_pairs(
    rng: random.Random,
    scenario_count: int,
    crowding_min_gap: float,
) -> list[tuple[dict[str, float], dict[str, float]]]:
    drafts: list[tuple[dict[str, float], dict[str, float]]] = []
    attempts = 0
    while len(drafts) < scenario_count:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("scenario constraints were not met")
        left = _draw_place(rng)
        right = _draw_place(rng)
        if abs(left["crowding"] - right["crowding"]) < crowding_min_gap:
            continue
        low, high = sorted((left, right), key=lambda place: place["crowding"])
        if not _has_tradeoff(low, high):
            continue
        drafts.append((low, high))
    return drafts


def _draw_place(rng: random.Random) -> dict[str, float]:
    values = {field: rng.random() for field in PLACE_FIELDS}
    PlaceAttributes.model_validate(values)
    return values


def _has_tradeoff(low: dict[str, float], high: dict[str, float]) -> bool:
    """The less crowded place is worse on at least one other attribute."""

    less_attractive = low["visual_attractiveness"] < high["visual_attractiveness"]
    farther = low["distance"] > high["distance"]
    less_novel = low["novelty"] < high["novelty"]
    return less_attractive or farther or less_novel
