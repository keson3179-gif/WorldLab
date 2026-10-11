"""Python movement choice. The LLM does not select a place."""

from __future__ import annotations

import hashlib
import math
import random

PLACEMENT_ALGORITHM = "worldlab.world_placement.v1"
CHOICE_ALGORITHM = "worldlab.world_choice.v1"
RECONSIDER_ALGORITHM = "worldlab.reconsider.v1"


def initial_locations(agent_ids: list[str], place_ids: tuple[str, ...] | list[str], world_seed: int) -> dict[str, str]:
    rng = random.Random(world_seed)
    return {agent_id: place_ids[rng.randrange(len(place_ids))] for agent_id in sorted(agent_ids)}


def agent_reconsiders(choice_seed_value: int, timestep: int, agent_id: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    material = f"{RECONSIDER_ALGORITHM}|{choice_seed_value}|{timestep}|{agent_id}"
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed).random() < rate


def choice_seed(world_seed: int, timestep: int, agent_id: str) -> int:
    material = f"{CHOICE_ALGORITHM}|{world_seed}|{timestep}|{agent_id}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def softmax_choice(
    place_ids: list[str],
    utilities: list[int],
    temperature: float,
    seed: int,
) -> tuple[str, dict[str, float]]:
    if len(place_ids) != len(utilities) or not place_ids:
        raise ValueError("utilities must align with places")
    if temperature <= 0.0:
        raise ValueError("choice temperature must be positive")
    peak = max(utilities)
    weights = [math.exp((utility - peak) / temperature) for utility in utilities]
    total = math.fsum(weights)
    probabilities = [weight / total for weight in weights]
    draw = random.Random(seed).random()
    cursor = 0.0
    selected = place_ids[-1]
    for place_id, probability in zip(place_ids, probabilities, strict=True):
        cursor += probability
        if draw <= cursor:
            selected = place_id
            break
    return selected, {place_id: probability for place_id, probability in zip(place_ids, probabilities, strict=True)}
