"""World dynamics calibration 001.

Every utility comes from the frozen atlas. This module does not call a model.
Initial placement is the World v0.1 state and is not redrawn.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT
from world.choice import agent_reconsiders, choice_seed, softmax_choice
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index, travel_distance
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import FrozenUtilityAtlas, crowding_level_ids, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "dynamics_calibration_001"
CHOICE_TEMPERATURES = (50.0, 100.0, 200.0, 400.0)
RECONSIDERATION_RATES = (0.25, 0.5, 0.75, 1.0)
CHOICE_SEEDS = tuple(range(20))
TIMESTEPS = 100
REGIME_NOTE = (
    "Regime labels locate cells relative to this 16-cell grid. "
    "They are not a ranking of human realism. This design has no human calibration."
)
REGIME_RULE = (
    "highly_oscillatory: site_01 and site_02 lag-1 autocorrelations are both negative, "
    "and the site_01/site_02 alternation rate is at or above the grid's upper quartile. "
    "highly_mobile: movement rate is at or above the upper quartile and mean dwell is at or below the lower quartile. "
    "diffuse: occupancy-entropy mean is at or above the upper quartile, while saturation frequency and empty-place frequency "
    "are both at or below their medians. "
    "relatively_persistent: movement rate is at or below the lower quartile, mean dwell is at or above the upper quartile, "
    "and the mean lag-1 occupancy autocorrelation is at or above the median."
)


def load_initial_locations() -> dict[str, str]:
    payload = json.loads((WORLD_V01_DIR / "initial_state.json").read_text(encoding="utf-8"))
    locations = {str(agent_id): str(place_id) for agent_id, place_id in payload["locations"].items()}
    if len(locations) != 20 or sum(occupancy_of(locations).values()) != 20:
        raise RuntimeError("World v0.1 initial state is not a 20-agent placement")
    return locations


def simulate_world(
    atlas: FrozenUtilityAtlas,
    initial: dict[str, str],
    *,
    choice_temperature: float,
    reconsideration_rate: float,
    choice_seed_value: int,
    timesteps: int = TIMESTEPS,
) -> dict[str, Any]:
    matrix = distance_matrix()
    places = place_index()
    current = {agent_id: initial[agent_id] for agent_id in sorted(initial)}
    series = [dict(current)]
    moves = 0
    for timestep in range(timesteps):
        occupancy = occupancy_of(current)
        decisions: dict[str, str] = {}
        for agent_id in sorted(current):
            origin = current[agent_id]
            if not agent_reconsiders(choice_seed_value, timestep, agent_id, reconsideration_rate):
                decisions[agent_id] = origin
                continue
            utilities = []
            for place_id in PLACE_IDS:
                capacity = int(places[place_id]["capacity"])
                distance = travel_distance(matrix, origin, place_id)
                utilities.append(atlas.lookup(agent_id, place_id, distance, occupancy[place_id], capacity))
            destination, _probabilities = softmax_choice(
                list(PLACE_IDS),
                utilities,
                choice_temperature,
                choice_seed(choice_seed_value, timestep, agent_id),
            )
            decisions[agent_id] = destination
        updated = {agent_id: decisions[agent_id] for agent_id in sorted(current)}
        moves += sum(1 for agent_id in updated if updated[agent_id] != current[agent_id])
        current = updated
        series.append(dict(current))
    metrics = measure_world(series, moves=moves, timesteps=timesteps)
    metrics.update(
        {
            "choice_temperature": choice_temperature,
            "reconsideration_rate": reconsideration_rate,
            "choice_seed": choice_seed_value,
            "llm_calls": 0,
        }
    )
    return metrics


def measure_world(series: list[dict[str, str]], *, moves: int, timesteps: int) -> dict[str, Any]:
    occupancy = [occupancy_of(locations) for locations in series]
    entropies = [_entropy(item) for item in occupancy]
    spells = [
        length
        for agent_id in sorted(series[0])
        for length in _spell_lengths([step[agent_id] for step in series])
    ]
    capacities = {place_id: int(place_index()[place_id]["capacity"]) for place_id in PLACE_IDS}
    observations = len(occupancy) * len(PLACE_IDS)
    saturated = sum(1 for item in occupancy for place_id in PLACE_IDS if item[place_id] >= capacities[place_id])
    empty = sum(1 for item in occupancy for place_id in PLACE_IDS if item[place_id] == 0)
    site_01 = [item["site_01"] for item in occupancy]
    site_02 = [item["site_02"] for item in occupancy]
    transitions = len(occupancy) - 1
    opposite = 0
    for left, right in zip(occupancy, occupancy[1:]):
        change_01 = right["site_01"] - left["site_01"]
        change_02 = right["site_02"] - left["site_02"]
        if change_01 * change_02 < 0:
            opposite += 1
    return {
        "movement_rate": moves / (len(series[0]) * timesteps),
        "mean_dwell": statistics.fmean(spells),
        "entropy_mean": statistics.fmean(entropies),
        "entropy_sd": statistics.stdev(entropies) if len(entropies) > 1 else 0.0,
        "mean_total_variation": statistics.fmean(
            [_total_variation(left, right, len(series[0])) for left, right in zip(occupancy, occupancy[1:])]
        ),
        "lag1": {place_id: _autocorr([item[place_id] for item in occupancy], 1) for place_id in PLACE_IDS},
        "lag2": {place_id: _autocorr([item[place_id] for item in occupancy], 2) for place_id in PLACE_IDS},
        "saturation_frequency": saturated / observations,
        "empty_frequency": empty / observations,
        "saturation_by_place": {
            place_id: sum(1 for item in occupancy if item[place_id] >= capacities[place_id]) / len(occupancy)
            for place_id in PLACE_IDS
        },
        "empty_by_place": {
            place_id: sum(1 for item in occupancy if item[place_id] == 0) / len(occupancy) for place_id in PLACE_IDS
        },
        "mean_occupancy": {place_id: statistics.fmean([item[place_id] for item in occupancy]) for place_id in PLACE_IDS},
        "site_01_site_02_contemporaneous_correlation": _pearson(site_01, site_02),
        "site_01_site_02_alternation_rate": opposite / transitions,
        "occupancy": [{place_id: item[place_id] for place_id in PLACE_IDS} for item in occupancy],
    }


def assign_regimes(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    movement = [float(cell["movement_rate"]["mean"]) for cell in cells]
    dwell = [float(cell["mean_dwell"]["mean"]) for cell in cells]
    entropy = [float(cell["occupancy_entropy"]["mean"]) for cell in cells]
    alternation = [float(cell["site_01_site_02"]["alternation_rate"]["mean"]) for cell in cells]
    lag1 = [float(cell["mean_lag1"]) for cell in cells]
    saturation = [float(cell["saturation_frequency"]["mean"]) for cell in cells]
    empty = [float(cell["empty_frequency"]["mean"]) for cell in cells]
    move_low, move_high = _quartile_edges(movement)
    dwell_low, dwell_high = _quartile_edges(dwell)
    entropy_high = _quartile_edges(entropy)[1]
    alternation_high = _quartile_edges(alternation)[1]
    lag_median = statistics.median(lag1)
    saturation_median = statistics.median(saturation)
    empty_median = statistics.median(empty)
    labeled = []
    for cell in cells:
        labels = []
        pair = cell["site_01_site_02"]
        lag_01 = pair["lag1"]["site_01"]["mean"]
        lag_02 = pair["lag1"]["site_02"]["mean"]
        if (
            lag_01 is not None
            and lag_02 is not None
            and float(lag_01) < 0.0
            and float(lag_02) < 0.0
            and float(pair["alternation_rate"]["mean"]) >= alternation_high
        ):
            labels.append("highly_oscillatory")
        if float(cell["movement_rate"]["mean"]) >= move_high and float(cell["mean_dwell"]["mean"]) <= dwell_low:
            labels.append("highly_mobile")
        if (
            float(cell["occupancy_entropy"]["mean"]) >= entropy_high
            and float(cell["saturation_frequency"]["mean"]) <= saturation_median
            and float(cell["empty_frequency"]["mean"]) <= empty_median
        ):
            labels.append("diffuse")
        if (
            cell["mean_lag1"] is not None
            and float(cell["movement_rate"]["mean"]) <= move_low
            and float(cell["mean_dwell"]["mean"]) >= dwell_high
            and float(cell["mean_lag1"]) >= lag_median
        ):
            labels.append("relatively_persistent")
        labeled.append({**cell, "regimes": labels})
    return labeled


def run_calibration(
    atlas: FrozenUtilityAtlas | None = None,
    *,
    output_dir: Path | None = None,
    temperatures: tuple[float, ...] = CHOICE_TEMPERATURES,
    rates: tuple[float, ...] = RECONSIDERATION_RATES,
    seeds: tuple[int, ...] = CHOICE_SEEDS,
    timesteps: int = TIMESTEPS,
) -> Path:
    source = load_frozen_atlas() if atlas is None else atlas
    destination = output_dir or OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    initial = load_initial_locations()
    _require_coverage(source, initial)
    worlds_path = destination / "worlds.jsonl"
    if worlds_path.exists():
        worlds_path.unlink()
    worlds = []
    for temperature in temperatures:
        for rate in rates:
            for seed in seeds:
                metrics = simulate_world(
                    source,
                    initial,
                    choice_temperature=temperature,
                    reconsideration_rate=rate,
                    choice_seed_value=seed,
                    timesteps=timesteps,
                )
                if metrics["llm_calls"] != 0:
                    raise RuntimeError("dynamics calibration called a model")
                append_jsonl(worlds_path, metrics)
                worlds.append(metrics)
                print(
                    f"world temperature={temperature:g} rate={rate:.2f} seed={seed} "
                    f"movement={metrics['movement_rate']:.3f}",
                    flush=True,
                )
    summary = _summarize(worlds, temperatures, rates, seeds, timesteps)
    write_json(destination / "summary.json", summary)
    return destination


def _summarize(
    worlds: list[dict[str, Any]],
    temperatures: tuple[float, ...],
    rates: tuple[float, ...],
    seeds: tuple[int, ...],
    timesteps: int,
) -> dict[str, Any]:
    cells = []
    for temperature in temperatures:
        for rate in rates:
            selected = [
                world
                for world in worlds
                if world["choice_temperature"] == temperature and world["reconsideration_rate"] == rate
            ]
            cells.append(_cell(selected, temperature, rate))
    return {
        "experiment_id": "dynamics_calibration_001",
        "llm_calls": 0,
        "world_count": len(worlds),
        "timesteps": timesteps,
        "choice_seeds": list(seeds),
        "initial_placement": "world_v0_1 initial_state.json",
        "utility_source": "utility_atlas_v1",
        "decision_signal": "overall_utility",
        "regime_note": REGIME_NOTE,
        "regime_rule": REGIME_RULE,
        "cells": assign_regimes(cells),
    }


def _cell(worlds: list[dict[str, Any]], temperature: float, rate: float) -> dict[str, Any]:
    lag1 = {place_id: _mean_sd([world["lag1"][place_id] for world in worlds]) for place_id in PLACE_IDS}
    finite_lag1 = [float(item["mean"]) for item in lag1.values() if item["mean"] is not None]
    return {
        "choice_temperature": temperature,
        "reconsideration_rate": rate,
        "n_seeds": len(worlds),
        "llm_calls": 0,
        "movement_rate": _mean_sd([world["movement_rate"] for world in worlds]),
        "mean_dwell": _mean_sd([world["mean_dwell"] for world in worlds]),
        "occupancy_entropy": {
            "mean": statistics.fmean([world["entropy_mean"] for world in worlds]),
            "temporal_sd_mean": statistics.fmean([world["entropy_sd"] for world in worlds]),
            "between_seed_sd": _stdev([world["entropy_mean"] for world in worlds]),
        },
        "consecutive_state_total_variation": _mean_sd([world["mean_total_variation"] for world in worlds]),
        "lag1_autocorrelation": lag1,
        "lag2_autocorrelation": {place_id: _mean_sd([world["lag2"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "mean_lag1": statistics.fmean(finite_lag1) if finite_lag1 else None,
        "saturation_frequency": _mean_sd([world["saturation_frequency"] for world in worlds]),
        "empty_frequency": _mean_sd([world["empty_frequency"] for world in worlds]),
        "saturation_by_place": {
            place_id: _mean_sd([world["saturation_by_place"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "empty_by_place": {
            place_id: _mean_sd([world["empty_by_place"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "mean_occupancy": {
            place_id: _mean_sd([world["mean_occupancy"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "site_01_site_02": {
            "lag1": {"site_01": lag1["site_01"], "site_02": lag1["site_02"]},
            "lag2": {
                "site_01": _mean_sd([world["lag2"]["site_01"] for world in worlds]),
                "site_02": _mean_sd([world["lag2"]["site_02"] for world in worlds]),
            },
            "contemporaneous_correlation": _mean_sd(
                [world["site_01_site_02_contemporaneous_correlation"] for world in worlds]
            ),
            "alternation_rate": _mean_sd([world["site_01_site_02_alternation_rate"] for world in worlds]),
        },
        "between_seed_variability": {
            "movement_rate_sd": _stdev([world["movement_rate"] for world in worlds]),
            "mean_dwell_sd": _stdev([world["mean_dwell"] for world in worlds]),
            "entropy_mean_sd": _stdev([world["entropy_mean"] for world in worlds]),
            "total_variation_sd": _stdev([world["mean_total_variation"] for world in worlds]),
            "alternation_rate_sd": _stdev([world["site_01_site_02_alternation_rate"] for world in worlds]),
        },
    }


def _require_coverage(atlas: FrozenUtilityAtlas, locations: dict[str, str]) -> None:
    matrix = distance_matrix()
    places = place_index()
    for agent_id in locations:
        for place_id in PLACE_IDS:
            capacity = int(places[place_id]["capacity"])
            for origin in PLACE_IDS:
                distance = travel_distance(matrix, origin, place_id)
                for level in crowding_level_ids(capacity):
                    atlas.lookup(agent_id, place_id, distance, level, capacity)


def _entropy(occupancy: dict[str, int]) -> float:
    total = sum(occupancy.values())
    entropy = 0.0
    for count in occupancy.values():
        if count:
            share = count / total
            entropy -= share * math.log(share)
    return entropy


def _total_variation(left: dict[str, int], right: dict[str, int], population: int) -> float:
    change = sum(abs(right[place_id] - left[place_id]) for place_id in PLACE_IDS)
    return 0.5 * change / population


def _autocorr(values: list[int], lag: int) -> float | None:
    if len(values) <= lag:
        return None
    return _pearson([float(value) for value in values[:-lag]], [float(value) for value in values[lag:]])


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((value - mean_left) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - mean_right) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _spell_lengths(locations: list[str]) -> list[int]:
    lengths = []
    current = 1
    for previous, nxt in zip(locations, locations[1:]):
        if nxt == previous:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return lengths


def _mean_sd(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    return {
        "mean": None if not finite else statistics.fmean(finite),
        "sd": _stdev(finite),
        "n": len(finite),
    }


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _quartile_edges(values: list[float]) -> tuple[float, float]:
    low, _middle, high = statistics.quantiles(values, n=4, method="inclusive")
    return float(low), float(high)


def main() -> None:
    run_calibration()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.dynamics_calibration")
    main()
