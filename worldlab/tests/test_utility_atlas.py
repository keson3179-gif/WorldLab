from __future__ import annotations

import json

import pytest

from experiments.experiment_001 import load_base_traits
from config.settings import get_settings
from experiments.experiment_001e import PROMPT_VERSION, build_utility_messages, prompt_sha256
from llm.client import LLMCallResult
from world.choice import agent_reconsiders, initial_locations
from world.dynamics_calibration import (
    REGIME_NOTE,
    assign_regimes,
    load_initial_locations,
    measure_world,
    simulate_world,
)
from world.geography import PLACE_IDS, distance_matrix, place_index
from world.utility_atlas import (
    FrozenUtilityAtlas,
    build_atlas,
    crowding_level_ids,
    distance_levels,
    iter_slots,
    metric_key,
)


def test_reachable_grid_is_the_capacity_discrete_set() -> None:
    matrix = distance_matrix()
    per_place = []
    for place_id in PLACE_IDS:
        capacity = int(place_index()[place_id]["capacity"])
        per_place.append(len(distance_levels(matrix, place_id)) * len(crowding_level_ids(capacity)))
    assert per_place == [45, 55, 65, 85, 105]
    assert sum(per_place) * 20 == 7100
    bases = load_base_traits(get_settings().personas_dir / "persona_set_v1", 1)
    slots = iter_slots(bases)
    assert len(slots) == 355
    attributes = {
        "crowding": slots[0]["crowding"],
        "visual_attractiveness": slots[0]["visual_attractiveness"],
        "novelty": slots[0]["novelty"],
        "distance": slots[0]["distance"],
    }
    messages = build_utility_messages(bases[0]["traits"], attributes, prompt_version=PROMPT_VERSION)
    text = "\n".join(message["content"] for message in messages).lower()
    assert prompt_sha256(messages) == slots[0]["prompt_sha256"]
    assert "site_" not in text
    assert "place_id" not in text
    assert "origin" not in text


def test_atlas_reuses_cache_and_then_freezes(tmp_path) -> None:
    bases = load_base_traits(get_settings().personas_dir / "persona_set_v1", 1)
    slots = iter_slots(bases)[:2]
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        json.dumps(
            {
                "prompt_sha256": slots[0]["prompt_sha256"],
                "scores": {
                    "overall_utility": 10,
                    "crowding_fit": 10,
                    "visual_fit": 10,
                    "distance_fit": 10,
                    "novelty_fit": 10,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = _ScriptedUtility()
    destination = build_atlas(
        client=client,
        output_dir=tmp_path / "atlas",
        cache_path=cache,
        slots=slots,
        persona_dir=get_settings().personas_dir / "persona_set_v1",
    )
    manifest = json.loads((destination / "atlas_manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen"] is True
    assert manifest["theoretical_slot_count"] == 2
    assert manifest["unique_prompt_count"] == 2
    assert manifest["reused_cache_entries"] == 1
    assert manifest["new_llm_calls"] == 1
    assert manifest["sha256"]
    assert client.calls == 1
    again = _ScriptedUtility()
    build_atlas(client=again, output_dir=destination, cache_path=cache, slots=slots)
    assert again.calls == 0


def test_reconsideration_and_atlas_world_make_no_model_calls() -> None:
    assert agent_reconsiders(1, 0, "agent_0000", 1.0) is True
    assert agent_reconsiders(1, 0, "agent_0000", 0.0) is False
    assert agent_reconsiders(4, 2, "agent_0003", 0.25) == agent_reconsiders(4, 2, "agent_0003", 0.25)
    initial = load_initial_locations()
    assert initial == initial_locations(list(initial), PLACE_IDS, 42)
    atlas = _fake_atlas(initial)
    stayed = simulate_world(atlas, dict(reversed(list(initial.items()))), choice_temperature=50.0, reconsideration_rate=0.0, choice_seed_value=3, timesteps=4)
    assert stayed["movement_rate"] == 0.0
    assert stayed["llm_calls"] == 0
    forward = simulate_world(atlas, dict(initial), choice_temperature=50.0, reconsideration_rate=1.0, choice_seed_value=3, timesteps=4)
    backward = simulate_world(
        atlas,
        {agent_id: initial[agent_id] for agent_id in reversed(list(initial))},
        choice_temperature=50.0,
        reconsideration_rate=1.0,
        choice_seed_value=3,
        timesteps=4,
    )
    assert forward["occupancy"] == backward["occupancy"]
    with pytest.raises(KeyError):
        atlas.lookup("missing_agent", "site_01", 0.0, 0, 8)


def test_metric_definitions_and_regime_rule() -> None:
    series = []
    for step in range(6):
        place = "site_01" if step % 2 == 0 else "site_02"
        series.append({agent: place for agent in [f"agent_{index:04d}" for index in range(20)]})
    metrics = measure_world(series, moves=20 * 5, timesteps=5)
    assert metrics["lag1"]["site_01"] == pytest.approx(-1.0)
    assert metrics["site_01_site_02_alternation_rate"] == 1.0
    assert metrics["mean_total_variation"] == pytest.approx(1.0)
    cells = assign_regimes(
        [
            _cell(0.1, 8.0, 0.4, 0.9, 0.05, 0.05, 0.1),
            _cell(0.2, 6.0, 0.6, 0.4, 0.1, 0.2, 0.2),
            _cell(0.85, 1.1, 1.4, -0.4, 0.8, 0.05, 0.05),
            _cell(0.9, 1.0, 1.5, -0.5, 0.9, 0.04, 0.04),
        ]
    )
    assert cells[0]["regimes"] == ["relatively_persistent"]
    assert "highly_mobile" in cells[3]["regimes"]
    assert "diffuse" in cells[3]["regimes"]
    assert "highly_oscillatory" in cells[3]["regimes"]
    assert "human realism" in REGIME_NOTE
    source = ( __import__("pathlib").Path(__file__).resolve().parents[1] / "world" / "dynamics_calibration.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source


def _cell(movement: float, dwell: float, entropy: float, lag: float, alternation: float, saturation: float, empty: float) -> dict[str, object]:
    spread = {"mean": lag, "sd": 0.0, "n": 20}
    return {
        "movement_rate": {"mean": movement},
        "mean_dwell": {"mean": dwell},
        "occupancy_entropy": {"mean": entropy},
        "mean_lag1": lag,
        "saturation_frequency": {"mean": saturation},
        "empty_frequency": {"mean": empty},
        "site_01_site_02": {
            "lag1": {"site_01": spread, "site_02": spread},
            "alternation_rate": {"mean": alternation},
        },
    }


def _fake_atlas(locations: dict[str, str]) -> FrozenUtilityAtlas:
    matrix = distance_matrix()
    places = place_index()
    records = []
    for agent_id in locations:
        for place_id in PLACE_IDS:
            capacity = int(places[place_id]["capacity"])
            for distance in distance_levels(matrix, place_id):
                for level in crowding_level_ids(capacity):
                    records.append(
                        {
                            "agent_id": agent_id,
                            "place_id": place_id,
                            "distance_key": metric_key(distance),
                            "crowding_level_id": level,
                            "overall_utility": 1000 if place_id == "site_05" else 0,
                        }
                    )
    return FrozenUtilityAtlas(records)


class _ScriptedUtility:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, messages, response_model):
        del messages
        self.calls += 1
        parsed = response_model.model_validate(
            {name: 20 for name in ("overall_utility", "crowding_fit", "visual_fit", "distance_fit", "novelty_fit")}
        )
        return LLMCallResult(
            raw_text=parsed.model_dump_json(),
            parsed=parsed,
            model="scripted",
            temperature=0.0,
            llm_seed=42,
            reasoning_effort="none",
            api_endpoint="scripted",
        )
