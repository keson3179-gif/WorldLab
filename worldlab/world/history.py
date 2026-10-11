"""History state for World v0.2.

Persona traits are not modified. Familiarity is a fixed Python function of
visit counts. Place ids stay out of any model prompt because this layer does
not call a model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from world.geography import PLACE_IDS

FAVORITE_ALGORITHM = "worldlab.favorite_place.v1"
SHUFFLE_ALGORITHM = "worldlab.favorite_shuffle.v1"
FAMILIARITY_RULE = (
    "familiarity_raw = log1p(visit_count). "
    "familiarity_score = 0 when every raw value for that agent is 0; "
    "otherwise raw divided by the maximum raw value across the five places "
    "for that agent at the decision. Scores are in [0, 1]."
)
FAVORITE_SEED = 42
SHUFFLE_SEED = 43
FAVORITE_VISIT_COUNT = 20
FAVORITE_CUMULATIVE_DWELL = 40
FAVORITE_TIME_SINCE = 0
OTHER_VISIT_COUNT = 2
OTHER_CUMULATIVE_DWELL = 5
OTHER_TIME_SINCE = 10
MANIPULATION_NOTE = "Controlled experimental manipulation, not a human parameter."


@dataclass
class AgentHistory:
    agent_id: str
    traits: Any
    current_location: str
    previous_location: str | None
    visit_count: dict[str, int]
    cumulative_dwell: dict[str, int]
    time_since_last_visit: dict[str, int | None]


def assign_favorites(agent_ids: list[str], seed: int = FAVORITE_SEED) -> dict[str, str]:
    agents = sorted(agent_ids)
    if len(agents) % len(PLACE_IDS) != 0:
        raise RuntimeError("favorite assignment expects a balanced agent count")
    copies = len(agents) // len(PLACE_IDS)
    bag = [place_id for place_id in PLACE_IDS for _copy in range(copies)]
    random.Random(seed).shuffle(bag)
    return {agent_id: bag[index] for index, agent_id in enumerate(agents)}


def shuffle_favorites(favorites: dict[str, str], seed: int = SHUFFLE_SEED) -> dict[str, str]:
    agents = sorted(favorites)
    places = [favorites[agent_id] for agent_id in agents]
    random.Random(seed).shuffle(places)
    return {agent_id: places[index] for index, agent_id in enumerate(agents)}


def initial_history(
    agent_id: str,
    traits: Any,
    current_location: str,
    favorite_place: str,
) -> AgentHistory:
    visit = {place_id: OTHER_VISIT_COUNT for place_id in PLACE_IDS}
    dwell = {place_id: OTHER_CUMULATIVE_DWELL for place_id in PLACE_IDS}
    since: dict[str, int | None] = {place_id: OTHER_TIME_SINCE for place_id in PLACE_IDS}
    visit[favorite_place] = FAVORITE_VISIT_COUNT
    dwell[favorite_place] = FAVORITE_CUMULATIVE_DWELL
    since[favorite_place] = FAVORITE_TIME_SINCE
    return AgentHistory(
        agent_id=agent_id,
        traits=traits,
        current_location=current_location,
        previous_location=None,
        visit_count=visit,
        cumulative_dwell=dwell,
        time_since_last_visit=since,
    )


def blank_history(agent_id: str, traits: Any, current_location: str) -> AgentHistory:
    return AgentHistory(
        agent_id=agent_id,
        traits=traits,
        current_location=current_location,
        previous_location=None,
        visit_count={place_id: 0 for place_id in PLACE_IDS},
        cumulative_dwell={place_id: 0 for place_id in PLACE_IDS},
        time_since_last_visit={place_id: None for place_id in PLACE_IDS},
    )


def clone_population(states: dict[str, AgentHistory]) -> dict[str, AgentHistory]:
    return {agent_id: _clone(states[agent_id]) for agent_id in sorted(states)}


def commit_history(states: dict[str, AgentHistory], destinations: dict[str, str]) -> dict[str, AgentHistory]:
    updated: dict[str, AgentHistory] = {}
    for agent_id in sorted(states):
        state = states[agent_id]
        destination = destinations[agent_id]
        visit = dict(state.visit_count)
        dwell = dict(state.cumulative_dwell)
        since = dict(state.time_since_last_visit)
        if destination != state.current_location:
            visit[destination] = visit[destination] + 1
        dwell[destination] = dwell[destination] + 1
        for place_id in PLACE_IDS:
            if place_id == destination:
                since[place_id] = 0
            elif since[place_id] is None:
                since[place_id] = None
            else:
                since[place_id] = int(since[place_id]) + 1
        updated[agent_id] = AgentHistory(
            agent_id=agent_id,
            traits=state.traits,
            current_location=destination,
            previous_location=state.current_location,
            visit_count=visit,
            cumulative_dwell=dwell,
            time_since_last_visit=since,
        )
    return updated


def familiarity_scores(visit_count: dict[str, int]) -> dict[str, float]:
    raw = {place_id: math.log1p(visit_count[place_id]) for place_id in PLACE_IDS}
    peak = max(raw.values())
    if peak == 0.0:
        return {place_id: 0.0 for place_id in PLACE_IDS}
    return {place_id: raw[place_id] / peak for place_id in PLACE_IDS}


def decision_utility(base_utility: float, familiarity_score: float, familiarity_weight: float) -> float:
    return base_utility + familiarity_weight * familiarity_score


def _clone(state: AgentHistory) -> AgentHistory:
    return AgentHistory(
        agent_id=state.agent_id,
        traits=state.traits,
        current_location=state.current_location,
        previous_location=state.previous_location,
        visit_count=dict(state.visit_count),
        cumulative_dwell=dict(state.cumulative_dwell),
        time_since_last_visit=dict(state.time_since_last_visit),
    )
