"""Descriptive summaries of a finished world run.

These figures describe the recorded dynamics. They do not establish emergence.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from world.geography import PLACE_IDS

ANALYSIS_NOTE = "Descriptive dynamics only. This summary does not claim emergence."
DIAGNOSTIC_NOTE = "Component scores are diagnostic outputs, not independent construct measures."


def summarize_run(
    *,
    initial_locations: dict[str, str],
    snapshots: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    series = [dict(initial_locations)]
    series.extend(dict(snapshot["locations_after"]) for snapshot in snapshots)
    occupancy_series = [_counts(locations) for locations in series]
    moves = _moves(snapshots)
    spells = {agent_id: _spell_lengths([locations[agent_id] for locations in series]) for agent_id in sorted(initial_locations)}
    entropies = [_entropy(occupancy) for occupancy in occupancy_series]
    changes = sum(1 for previous, nxt in zip(occupancy_series, occupancy_series[1:]) if previous != nxt)
    crowding_series = _crowding_series(snapshots)
    return {
        "note": ANALYSIS_NOTE,
        "component_score_note": DIAGNOSTIC_NOTE,
        "timesteps": len(snapshots),
        "observation_count": len(series),
        "total_movements": sum(moves.values()),
        "final_occupancy": occupancy_series[-1],
        "occupancy_summary": _occupancy_summary(occupancy_series),
        "crowding_summary": _crowding_summary(crowding_series),
        "visits_per_place": {place_id: sum(occupancy[place_id] for occupancy in occupancy_series) for place_id in PLACE_IDS},
        "arrivals_per_place": _arrivals(snapshots),
        "place_concentration": _concentration(occupancy_series),
        "moves_per_agent": moves,
        "mobility": _mobility(moves, series),
        "dwell": {
            "definition": "consecutive location observations, including the initial location",
            "spell_count": sum(len(lengths) for lengths in spells.values()),
            "mean_spell_length": statistics.fmean([length for lengths in spells.values() for length in lengths]),
            "median_spell_length": float(statistics.median([length for lengths in spells.values() for length in lengths])),
            "per_agent_mean_spell_length": {agent_id: statistics.fmean(lengths) for agent_id, lengths in spells.items()},
        },
        "occupancy_entropy": {
            "unit": "nats",
            "definition": "Shannon entropy of the occupancy share at each observation",
            "initial": entropies[0],
            "final": entropies[-1],
            "mean": statistics.fmean(entropies),
            "min": min(entropies),
            "max": max(entropies),
            "trajectory": entropies,
        },
        "occupancy_state_changes": changes,
        "utility_distribution": _utility_distribution(evaluations),
        "occupancy_trajectory": occupancy_series,
        "crowding_trajectory": crowding_series,
    }


def _counts(locations: dict[str, str]) -> dict[str, int]:
    counts = {place_id: 0 for place_id in PLACE_IDS}
    for place_id in locations.values():
        counts[place_id] += 1
    return counts


def _moves(snapshots: list[dict[str, Any]]) -> dict[str, int]:
    totals = {agent_id: 0 for agent_id in snapshots[0]["locations_before"]} if snapshots else {}
    for snapshot in snapshots:
        for agent_id, origin in snapshot["locations_before"].items():
            if snapshot["locations_after"][agent_id] != origin:
                totals[agent_id] += 1
    return dict(sorted(totals.items()))


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


def _entropy(occupancy: dict[str, int]) -> float:
    total = sum(occupancy.values())
    entropy = 0.0
    for count in occupancy.values():
        if count:
            share = count / total
            entropy -= share * math.log(share)
    return entropy


def _occupancy_summary(series: list[dict[str, int]]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for place_id in PLACE_IDS:
        values = [occupancy[place_id] for occupancy in series]
        summary[place_id] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "final": values[-1],
        }
    return summary


def _crowding_series(snapshots: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not snapshots:
        return []
    series = [{place_id: float(snapshots[0]["crowding_before"][place_id]) for place_id in PLACE_IDS}]
    series.extend({place_id: float(snapshot["crowding_after"][place_id]) for place_id in PLACE_IDS} for snapshot in snapshots)
    return series


def _crowding_summary(series: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for place_id in PLACE_IDS:
        values = [crowding[place_id] for crowding in series]
        summary[place_id] = {"mean": statistics.fmean(values), "min": min(values), "max": max(values), "final": values[-1]}
    return summary


def _arrivals(snapshots: list[dict[str, Any]]) -> dict[str, int]:
    arrivals = {place_id: 0 for place_id in PLACE_IDS}
    for snapshot in snapshots:
        for agent_id, origin in snapshot["locations_before"].items():
            destination = snapshot["locations_after"][agent_id]
            if destination != origin:
                arrivals[destination] += 1
    return arrivals


def _concentration(series: list[dict[str, int]]) -> dict[str, float]:
    shares = []
    for place_id in PLACE_IDS:
        mean_count = statistics.fmean([occupancy[place_id] for occupancy in series])
        shares.append(mean_count / sum(series[0].values()))
    return {
        "herfindahl": sum(share * share for share in shares),
        "max_mean_share": max(shares),
    }


def _mobility(moves: dict[str, int], series: list[dict[str, str]]) -> dict[str, Any]:
    counts = list(moves.values())
    unique = {agent_id: len({locations[agent_id] for locations in series}) for agent_id in moves}
    return {
        "move_count": _spread(counts),
        "unique_places": _spread(list(unique.values())),
        "per_agent_unique_places": unique,
    }


def _spread(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "min": ordered[0],
        "p25": ordered[max(0, math.ceil(0.25 * (len(ordered) - 1)))],
        "median": float(statistics.median(values)),
        "p75": ordered[max(0, math.ceil(0.75 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def _utility_distribution(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    overall = [int(row["overall_utility"]) for row in evaluations]
    by_place: dict[str, list[int]] = {place_id: [] for place_id in PLACE_IDS}
    for row in evaluations:
        by_place[str(row["place_id"])].append(int(row["overall_utility"]))
    return {
        "signal": "overall_utility",
        "overall": _numeric(overall),
        "by_place": {place_id: _numeric(values) for place_id, values in by_place.items()},
        "diagnostic_components": {
            name: _numeric([int(row[name]) for row in evaluations])
            for name in ("crowding_fit", "visual_fit", "distance_fit", "novelty_fit")
        },
    }


def _numeric(values: list[int]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }
