"""Experiment 015: long-horizon dose fill-in on the Experiment 014 seed cohort.

The social grid is 0.40, 0.60, 0.80, and 1.00. Lambda 0.40 and 1.00, plus both
S0 anchors, must match Experiment 014 across t=0-2000 before any new lambda is
simulated. This module does not call a model.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from random import Random
from typing import Any, Iterator

import world.experiment_013 as experiment_013
import world.experiment_014 as experiment_014
from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import _tv, trajectory_hash
from world.experiment_005 import CENTERED_NORMALIZATION, FAMILIARITY_WEIGHT
from world.experiment_009 import classify_persistence
from world.experiment_012 import HISTORIES, REGIMES, _share_of
from world.experiment_013 import (
    BLEND_TOLERANCE,
    BOOTSTRAP_DRAWS,
    ENVIRONMENT_STEP,
    HELDOUT_SEEDS,
    OVERRIDE_COUNT,
    simulate_history,
)
from world.experiment_014 import (
    ACCUMULATOR_LIMIT,
    HORIZONS,
    _branch_safety,
    _dist,
    _empty_numeric,
    _file_sha256,
    _numeric_summary,
    _observe_memory,
    _observe_scores,
    _safety_pair,
    jaccard,
    load_intervention_registry,
    long_horizon_class,
    transition_counts,
)
from world.geography import PLACE_IDS
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_15"
WORLD_V14_DIR = experiment_014.OUTPUT_DIR
EXPECTED_ATLAS_SHA256 = experiment_014.EXPECTED_ATLAS_SHA256
TIMESTEPS = 2000
SOCIAL_LAMBDAS = (0.4, 0.6, 0.8, 1.0)
S0_LAMBDAS = (0.0, 1.0)
ANCHOR_LAMBDAS = {"S0": (1.0, 0.0), "RW50": (1.0, 0.4), "EXP99": (1.0, 0.4)}
NEW_LAMBDAS = (0.6, 0.8)
ANALYSIS_BOOTSTRAP_SEED = 20260927
PERSISTENCE_MIN = 0.025
ESTIMATED_OUTPUT_BYTES = 2_000_000_000
MINIMUM_FREE_BYTES = 40_000_000_000
WILSON_Z = 1.959963984540054
REPLAYS = (
    ("RW50", "PERTURBED", 0.4, 50),
    ("RW50", "CONTROL", 0.6, 75),
    ("EXP99", "PERTURBED", 0.8, 99),
    ("EXP99", "CONTROL", 1.0, 50),
    ("S0", "CONTROL", 0.0, 50),
    ("S0", "CONTROL", 1.0, 50),
)
ZERO_NOTE = (
    "The percentile interval collapses to [0, 0] because every observed seed value is zero. "
    "This does not show that the mean for unseen seeds is zero. "
    "A bootstrap cannot recover an unobserved tail from an all-zero sample."
)
SCOPE_SENTENCE = (
    "These 2000-step results are finite-horizon simulation results on the existing seeds 50-99. "
    "They fill a preregistered lambda grid; they are not a new independent replication. "
    "Formal class follows the regime mean. Intervals describe stochastic-seed variation under this frozen architecture "
    "and do not replace that class, cover a new population, or provide simultaneous 95 percent coverage across cells. "
    "A regime mean below 0.025 does not mean every tested history has attenuated. "
    "The result is not permanent path dependence, a phase transition, a critical point, a proof of metastability, "
    "human realism, or external validity. A grid band is not an interpolated critical lambda."
)
STATISTICAL_RULES = (
    "Mean D uses a seed-level percentile bootstrap with 10000 draws and seed 20260927. "
    "One draw selects the same seeds for every lambda and horizon. "
    "Adjacent H2000 contrasts use that paired-seed bootstrap. "
    "Seed persistent rate reports k/50 and a Wilson 95 percent interval. "
    "An all-zero sample has mean interval [0, 0] and does not identify an unobserved tail. "
    "These rules are fixed before the run."
)


def _lambda_label(value: float) -> str:
    return f"{value:.2f}"


def lambdas_for(regime_id: str) -> tuple[float, ...]:
    return S0_LAMBDAS if regime_id == "S0" else SOCIAL_LAMBDAS


def anchor_keys() -> list[str]:
    keys = []
    for regime_id, lambdas in ANCHOR_LAMBDAS.items():
        for seed in HELDOUT_SEEDS:
            for history in HISTORIES:
                for lambda_value in lambdas:
                    keys.append(f"{regime_id}:{seed}:{history}:{_lambda_label(lambda_value)}")
    return keys


def horizon_onset(levels: dict[float, float]) -> dict[str, Any]:
    if set(levels) != set(SOCIAL_LAMBDAS):
        raise RuntimeError("experiment 015 onset uses only lambda 0.40, 0.60, 0.80, and 1.00")
    persists = [levels[value] >= PERSISTENCE_MIN for value in SOCIAL_LAMBDAS]
    seen = False
    for flag in persists:
        if seen and not flag:
            return {
                "status": "NONMONOTONIC",
                "horizon_onset_lambda": None,
                "horizon_onset_band": None,
                "minimum_tested_already_persistent": False,
            }
        if flag:
            seen = True
    if not any(persists):
        return {
            "status": "NONE",
            "horizon_onset_lambda": None,
            "horizon_onset_band": None,
            "minimum_tested_already_persistent": False,
        }
    index = persists.index(True)
    if index == 0:
        return {
            "status": "MINIMUM_TESTED_PERSISTENT",
            "horizon_onset_lambda": None,
            "horizon_onset_band": None,
            "minimum_tested_already_persistent": True,
            "note": "the smallest tested lambda already meets the regime-mean threshold; the lower bound is unknown and the onset is not 0.40 exactly",
        }
    previous = SOCIAL_LAMBDAS[index - 1]
    onset = SOCIAL_LAMBDAS[index]
    return {
        "status": "ONSET",
        "horizon_onset_lambda": onset,
        "horizon_onset_band": {
            "lower_exclusive": previous,
            "upper_inclusive": onset,
            "label": f"({_lambda_label(previous)}, {_lambda_label(onset)}]",
        },
        "minimum_tested_already_persistent": False,
    }


def wilson_interval(successes: int, n: int = 50) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise RuntimeError("Wilson interval requires a count inside the observed sample")
    proportion = successes / n
    z2 = WILSON_Z * WILSON_Z
    denominator = 1.0 + z2 / n
    center = (proportion + z2 / (2.0 * n)) / denominator
    margin = WILSON_Z * math.sqrt(proportion * (1.0 - proportion) / n + z2 / (4.0 * n * n)) / denominator
    return center - margin, center + margin


def _bootstrap_means(columns: list[list[float]], *, draws: int, seed: int) -> list[list[float]]:
    width = len(columns)
    if width == 0:
        raise RuntimeError("bootstrap requires a column")
    size = len(columns[0])
    if size == 0 or any(len(column) != size for column in columns):
        raise RuntimeError("paired bootstrap columns must share one seed order")
    sampler = Random(seed)
    samples = [[] for _column in range(width)]
    for _draw in range(draws):
        indexes = [sampler.randrange(size) for _index in range(size)]
        for cell, column in enumerate(columns):
            samples[cell].append(statistics.fmean(column[index] for index in indexes))
    return samples


def paired_mean_intervals(
    columns: list[list[float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = ANALYSIS_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    samples = _bootstrap_means(columns, draws=draws, seed=seed)
    intervals = []
    for column, draws_for_column in zip(columns, samples, strict=True):
        point = statistics.fmean(column)
        degenerate = all(value == 0.0 for value in column)
        ordered = sorted(draws_for_column)
        low = ordered[int(0.025 * (draws - 1))]
        high = ordered[int(0.975 * (draws - 1))]
        if degenerate:
            low, high = 0.0, 0.0
        intervals.append(
            {
                "mean": point,
                "ci95_low": low,
                "ci95_high": high,
                "degenerate_zero": degenerate,
                "n_seeds": len(column),
                "bootstrap_draws": draws,
                "bootstrap_seed": seed,
                "note": ZERO_NOTE if degenerate else None,
            }
        )
    return intervals


def set_relation(early: list[bool], late: list[bool]) -> dict[str, Any]:
    counts = transition_counts(early, late)
    early_ids = {index for index, flag in enumerate(early) if flag}
    late_ids = {index for index, flag in enumerate(late) if flag}
    overlap = jaccard(early_ids, late_ids)
    if counts["PP"] == 0 and counts["AP"] == 0 and counts["PA"] == 0:
        relation = "empty"
    elif counts["AP"] and counts["PA"]:
        relation = "replaced"
    elif counts["AP"]:
        relation = "expanded"
    elif counts["PA"]:
        relation = "shrunk"
    else:
        relation = "retained"
    return {"counts": counts, "jaccard": overlap, "relation": relation}


def new_lambdas_allowed(audit: dict[str, Any]) -> bool:
    return audit["matched"] == 600 and audit["mismatched"] == 0 and audit["anchor_count"] == 600


def build_anchor_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatched = [row for row in rows if not row["match"]]
    return {
        "experiment_id": "experiment_015_long_horizon_dose_response",
        "comparison": "full t=0-2000 canonical trajectory hash against experiment 014",
        "anchor_count": len(rows),
        "matched": len(rows) - len(mismatched),
        "mismatched": len(mismatched),
        "new_lambdas_allowed": len(rows) == 600 and not mismatched,
        "rows": rows,
        "llm_calls": 0,
    }


@contextmanager
def dose_grid() -> Iterator[None]:
    original = experiment_013.LAMBDAS
    experiment_013.LAMBDAS = tuple(dict.fromkeys((*original, *SOCIAL_LAMBDAS, *S0_LAMBDAS)))
    try:
        yield
    finally:
        experiment_013.LAMBDAS = original


def _refuse_nonempty_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("experiment 015 refuses to write into a non-empty output directory")


def _refuse_frozen_directories(destination: Path) -> None:
    paths = [getattr(experiment_014, f"WORLD_V{index:02d}_DIR") for index in range(1, 14)]
    paths.extend((experiment_014.OUTPUT_DIR, experiment_014.ATLAS_DIR, experiment_014.CALIBRATION_DIR))
    if destination.resolve() in {path.resolve() for path in paths}:
        raise RuntimeError("experiment 015 must not write into frozen world outputs")


def _pack_donors(donors: dict[str, dict[int, dict[str, dict[str, float]]]], agent_ids: tuple[str, ...]) -> bytes:
    chunks = []
    for history in HISTORIES:
        field = donors[history]
        for timestep in range(ENVIRONMENT_STEP, TIMESTEPS + 1):
            step = field[timestep]
            for agent_id in agent_ids:
                scores = step[agent_id]
                chunks.append(struct.pack("<5d", *(float(scores[place_id]) for place_id in PLACE_IDS)))
    return b"".join(chunks)


def _unpack_donors(raw: bytes, agent_ids: tuple[str, ...]) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    steps = TIMESTEPS - ENVIRONMENT_STEP + 1
    expected = 2 * steps * len(agent_ids) * 5 * 8
    if len(raw) != expected:
        raise RuntimeError("donor cache size does not match the 2000-step field")
    donors: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    offset = 0
    for history in HISTORIES:
        donors[history] = {}
        for timestep in range(ENVIRONMENT_STEP, TIMESTEPS + 1):
            step = {}
            for agent_id in agent_ids:
                values = struct.unpack_from("<5d", raw, offset)
                offset += 40
                step[agent_id] = {place_id: values[index] for index, place_id in enumerate(PLACE_IDS)}
            donors[history][timestep] = step
    return donors


def _window_record(control: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if len(control["series"]) != TIMESTEPS + 1 or len(perturbed["series"]) != TIMESTEPS + 1:
        raise RuntimeError("window metrics require t=0 through t=2000")
    distances = [
        _tv(_share_of(perturbed["series"][timestep]), _share_of(control["series"][timestep]))
        for timestep in range(TIMESTEPS + 1)
    ]
    record = {}
    for name, start, stop in HORIZONS:
        value = statistics.fmean(distances[start : stop + 1])
        record[name] = {
            "D_horizon": value,
            "persistence_class": classify_persistence(value),
            "seed_persistent": value >= PERSISTENCE_MIN,
        }
    return record


def _check_result(result: dict[str, Any], lambda_value: float) -> None:
    if result["llm_calls"] != 0:
        raise RuntimeError("experiment 015 called a model")
    if not math.isfinite(result["max_blend_error"]) or result["max_blend_error"] > BLEND_TOLERANCE:
        raise RuntimeError("blend audit failed")
    if lambda_value == 1.0 and result["used_trace"] != result["endo_trace"]:
        raise RuntimeError("lambda 1 used field is not the endogenous field")


def _release(result: dict[str, Any]) -> None:
    for key in ("series", "moves", "probabilities", "used_trace", "endo_trace", "mid_trace", "endogenous_field"):
        result.pop(key, None)


def _simulate(atlas, initial, spec, seed, history, lambda_value, plan, donors, require_donors, on_step) -> dict[str, Any]:
    result = simulate_history(
        atlas,
        initial,
        spec,
        choice_seed_value=seed,
        history=history,
        lambda_value=lambda_value,
        plan=plan,
        donors=donors,
        timesteps=TIMESTEPS,
        require_donors=require_donors,
        on_step=on_step,
    )
    _check_result(result, lambda_value)
    return result


def _collect_anchor_seed(atlas, initial, spec, seed, plan, agent_ids, references) -> dict[str, Any]:
    numeric = _empty_numeric()

    def watch(logged, current, memory, actions, _audit) -> None:
        _observe_memory(spec, memory, numeric)

    donors: dict[str, Any] = {}
    lambda1 = {}
    for history in HISTORIES:
        result = _simulate(
            atlas, initial, spec, seed, history, 1.0, plan, None, False, watch if spec["regime_id"] == "EXP99" else None
        )
        _observe_scores(result["endogenous_field"], numeric)
        donors[history] = result["endogenous_field"]
        lambda1[history] = result
    if spec["regime_id"] == "EXP99" and (numeric["nonfinite"] or numeric["exploded"] or numeric["max_abs"] > ACCUMULATOR_LIMIT):
        raise RuntimeError(f"{spec['regime_id']} seed {seed} failed the numerical stability audit")
    levels = {}
    anchors = []
    safety = []
    base_rng = list(lambda1["CONTROL"]["rng_draw_counts"])
    prefixes = {history: trajectory_hash(lambda1[history]["series"][:ENVIRONMENT_STEP]) for history in HISTORIES}
    canonicals = {history: lambda1[history]["canonical_dynamics_sha256"] for history in HISTORIES}
    for lambda_value in ANCHOR_LAMBDAS[spec["regime_id"]]:
        pair = {}
        for history in HISTORIES:
            if lambda_value == 1.0:
                result = lambda1[history]
            else:
                result = _simulate(atlas, initial, spec, seed, history, lambda_value, plan, donors, True, None)
            if result["rng_draw_counts"] != base_rng:
                raise RuntimeError("lambda or history changed the RNG schedule")
            if trajectory_hash(result["series"][:ENVIRONMENT_STEP]) != prefixes[history]:
                raise RuntimeError("blending changed the pre-t22 history")
            if history == "PERTURBED" and result["override_count"] != OVERRIDE_COUNT:
                raise RuntimeError("perturbed history does not have exactly two overrides")
            if history == "CONTROL" and result["override_count"] != 0:
                raise RuntimeError("control history has an override")
            pair[history] = result
            safety.append(_branch_safety(spec, seed, history, lambda_value, result))
        if spec["regime_id"] == "S0":
            for history in HISTORIES:
                if pair[history]["canonical_dynamics_sha256"] != canonicals[history]:
                    raise RuntimeError("S0 lambda levels are not dynamically equivalent through t=2000")
        windows = _window_record(pair["CONTROL"], pair["PERTURBED"])
        hashes = {}
        for history, result in pair.items():
            key = f"{spec['regime_id']}:{seed}:{history}:{_lambda_label(lambda_value)}"
            actual = result["trajectory_sha256"]
            expected = references[key]
            anchors.append({"key": key, "expected": expected, "actual": actual, "match": actual == expected})
            hashes[history] = actual
            _release(result)
        levels[_lambda_label(lambda_value)] = {
            "lambda": lambda_value,
            "horizons": windows,
            "control_trajectory_sha256": hashes["CONTROL"],
            "perturbed_trajectory_sha256": hashes["PERTURBED"],
            "max_blend_error": max(pair[history]["max_blend_error"] for history in HISTORIES),
        }
    packed = _pack_donors(donors, agent_ids)
    del donors
    return {
        "regime_id": spec["regime_id"],
        "choice_seed": seed,
        "levels": levels,
        "numeric": numeric if spec["regime_id"] == "EXP99" else None,
        "anchors": anchors,
        "safety": safety,
        "packed_donors": packed,
        "rng_draw_counts": base_rng,
        "prefix_sha256": prefixes,
        "llm_calls": 0,
    }


def _add_new_lambdas(atlas, initial, spec, record, plan, agent_ids) -> list[dict[str, Any]]:
    donors = _unpack_donors(record.pop("packed_donors"), agent_ids)
    safety = []
    for lambda_value in NEW_LAMBDAS:
        pair = {}
        for history in HISTORIES:
            result = _simulate(atlas, initial, spec, record["choice_seed"], history, lambda_value, plan, donors, True, None)
            if result["rng_draw_counts"] != record["rng_draw_counts"]:
                raise RuntimeError("new lambda changed the RNG schedule")
            if trajectory_hash(result["series"][:ENVIRONMENT_STEP]) != record["prefix_sha256"][history]:
                raise RuntimeError("new lambda changed the pre-t22 history")
            pair[history] = result
            safety.append(_branch_safety(spec, record["choice_seed"], history, lambda_value, result))
        windows = _window_record(pair["CONTROL"], pair["PERTURBED"])
        record["levels"][_lambda_label(lambda_value)] = {
            "lambda": lambda_value,
            "horizons": windows,
            "control_trajectory_sha256": pair["CONTROL"]["trajectory_sha256"],
            "perturbed_trajectory_sha256": pair["PERTURBED"]["trajectory_sha256"],
            "max_blend_error": max(pair[history]["max_blend_error"] for history in HISTORIES),
        }
        for result in pair.values():
            _release(result)
    del donors
    return safety


def _level_rows(rows: list[dict[str, Any]], lambda_value: float) -> list[dict[str, Any]]:
    label = _lambda_label(lambda_value)
    return [row["levels"][label] for row in sorted(rows, key=lambda item: item["choice_seed"])]


def _seed_ids(rows: list[dict[str, Any]], lambda_value: float, horizon: str) -> list[int]:
    label = _lambda_label(lambda_value)
    return sorted(
        row["choice_seed"] for row in rows if row["levels"][label]["horizons"][horizon]["seed_persistent"]
    )


def _flags(rows: list[dict[str, Any]], lambda_value: float, horizon: str) -> list[bool]:
    ordered = sorted(rows, key=lambda item: item["choice_seed"])
    label = _lambda_label(lambda_value)
    return [row["levels"][label]["horizons"][horizon]["seed_persistent"] for row in ordered]


def _uncertainty(rows: list[dict[str, Any]], lambdas: tuple[float, ...]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: item["choice_seed"])
    columns = []
    labels = []
    for name, _start, _stop in HORIZONS:
        for lambda_value in lambdas:
            label = _lambda_label(lambda_value)
            columns.append([row["levels"][label]["horizons"][name]["D_horizon"] for row in ordered])
            labels.append(f"{name}:{label}")
    contrasts = []
    if lambdas == SOCIAL_LAMBDAS:
        for later, earlier in ((0.6, 0.4), (0.8, 0.6), (1.0, 0.8)):
            later_label = _lambda_label(later)
            earlier_label = _lambda_label(earlier)
            columns.append(
                [
                    row["levels"][later_label]["horizons"]["H2000"]["D_horizon"]
                    - row["levels"][earlier_label]["horizons"]["H2000"]["D_horizon"]
                    for row in ordered
                ]
            )
            contrast_label = f"H2000:{later_label}-{earlier_label}"
            labels.append(contrast_label)
            contrasts.append(contrast_label)
    intervals = paired_mean_intervals(columns)
    cells = {label: interval for label, interval in zip(labels, intervals, strict=True)}
    return {"cells": cells, "h2000_contrasts": contrasts}


def _horizon_block(rows: list[dict[str, Any]], lambda_value: float, horizon: str, interval: dict[str, Any]) -> dict[str, Any]:
    values = [level["horizons"][horizon]["D_horizon"] for level in _level_rows(rows, lambda_value)]
    ids = _seed_ids(rows, lambda_value, horizon)
    low, high = wilson_interval(len(ids), len(rows))
    mean = statistics.fmean(values)
    return {
        "D_horizon": mean,
        "persistence_class": classify_persistence(mean),
        "mean_ci95": [interval["ci95_low"], interval["ci95_high"]],
        "mean_ci_degenerate_zero": interval["degenerate_zero"],
        "mean_ci_note": interval["note"],
        "seed_persistent_count": len(ids),
        "seed_persistent_rate": len(ids) / len(rows),
        "seed_persistent_ids": ids,
        "wilson95": [low, high],
        "distribution": _dist(values),
    }


def _regime_summary(spec, rows, worlds, uncertainty) -> dict[str, Any]:
    lambdas = lambdas_for(spec["regime_id"])
    levels = {}
    for lambda_value in lambdas:
        label = _lambda_label(lambda_value)
        horizons = {
            name: _horizon_block(rows, lambda_value, name, uncertainty["cells"][f"{name}:{label}"])
            for name, _start, _stop in HORIZONS
        }
        ordered = sorted(rows, key=lambda item: item["choice_seed"])
        early = [row["levels"][label]["horizons"]["H500"]["seed_persistent"] for row in ordered]
        late = [row["levels"][label]["horizons"]["H2000"]["seed_persistent"] for row in ordered]
        levels[label] = {
            "lambda": lambda_value,
            "horizons": horizons,
            "long_horizon_class": long_horizon_class([horizons[name]["D_horizon"] for name, _start, _stop in HORIZONS]),
            "h500_to_h2000": set_relation(early, late),
            "max_blend_error": max(row["levels"][label]["max_blend_error"] for row in rows),
        }
    adjacent = {}
    if lambdas == SOCIAL_LAMBDAS:
        for name, _start, _stop in HORIZONS:
            adjacent[name] = {}
            for later, earlier in ((0.6, 0.4), (0.8, 0.6), (1.0, 0.8)):
                key = f"{_lambda_label(earlier)}->{_lambda_label(later)}"
                adjacent[name][key] = set_relation(_flags(rows, earlier, name), _flags(rows, later, name))
    onsets = None
    if lambdas == SOCIAL_LAMBDAS:
        onsets = {
            name: horizon_onset({value: levels[_lambda_label(value)]["horizons"][name]["D_horizon"] for value in SOCIAL_LAMBDAS})
            for name, _start, _stop in HORIZONS
        }
    contrasts = {
        label: uncertainty["cells"][label]
        for label in uncertainty["h2000_contrasts"]
    }
    return {
        "n_seeds": len(rows),
        "levels": levels,
        "adjacent_lambda": adjacent,
        "horizon_onsets": onsets,
        "h2000_paired_contrasts": contrasts,
        "safety": {
            history: {
                _lambda_label(lambda_value): _safety_pair(worlds, spec["regime_id"], history, lambda_value)
                for lambda_value in lambdas
            }
            for history in HISTORIES
        },
        "llm_calls": 0,
    }


def overall_conclusion(regimes: dict[str, Any]) -> str:
    parts = [
        "Experiment 015 fills the preregistered long-horizon grid at lambda 0.40, 0.60, 0.80, and 1.00 on seeds 50-99.",
        "This is a dose fill-in, not a new independent replication.",
        "Formal persistence class follows the regime mean. Bootstrap intervals, Wilson intervals, medians, and seed rates do not replace that class.",
        "A regime mean below 0.025 does not mean every tested history has attenuated.",
        "Grid onset intervals describe adjacent tested points only and are not an interpolated critical lambda.",
    ]
    for regime_id in ("RW50", "EXP99"):
        cell = regimes[regime_id]["levels"]["1.00"]["horizons"]["H2000"]
        count = cell["seed_persistent_count"]
        if count >= 45:
            parts.append(
                f"{regime_id} lambda 1.00 at H2000 is {cell['persistence_class']} "
                f"with {count}/50 tested seeds persistent, so the mean is carried by nearly the whole tested cohort."
            )
        else:
            parts.append(
                f"{regime_id} lambda 1.00 at H2000 is {cell['persistence_class']} with {count}/50 tested seeds persistent."
            )
        leftovers = []
        for label, level in regimes[regime_id]["levels"].items():
            horizon = level["horizons"]["H2000"]
            if horizon["persistence_class"] == "attenuated" and horizon["seed_persistent_count"] > 0:
                leftovers.append(f"{regime_id} lambda {label} retains {horizon['seed_persistent_count']}/50 persistent seeds")
        if leftovers:
            parts.append("Mean attenuation with remaining seed persistence at H2000: " + "; ".join(leftovers) + ".")
    parts.append(SCOPE_SENTENCE)
    return " ".join(parts)


def _replay(atlas, initial, plans, metrics) -> list[dict[str, Any]]:
    stored = {}
    for row in metrics:
        for level in row["levels"].values():
            for history, field in (("CONTROL", "control_trajectory_sha256"), ("PERTURBED", "perturbed_trajectory_sha256")):
                stored[(row["regime_id"], history, level["lambda"], row["choice_seed"])] = level[field]
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    report = []
    for regime_id, history, lambda_value, seed in REPLAYS:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        if (regime_id, seed) not in cache:
            donors: dict[str, Any] = {}
            trajectories = {}
            for donor_history in HISTORIES:
                result = _simulate(
                    atlas, initial, spec, seed, donor_history, 1.0, plans[(regime_id, seed)], None, False, None
                )
                donors[donor_history] = result["endogenous_field"]
                trajectories[donor_history] = result["trajectory_sha256"]
                _release(result)
            cache[(regime_id, seed)] = {"donors": donors, "trajectories": trajectories}
        cached = cache[(regime_id, seed)]
        if lambda_value == 1.0:
            actual = cached["trajectories"][history]
        else:
            result = _simulate(
                atlas, initial, spec, seed, history, lambda_value, plans[(regime_id, seed)], cached["donors"], True, None
            )
            actual = result["trajectory_sha256"]
            _release(result)
        expected = stored[(regime_id, history, lambda_value, seed)]
        report.append(
            {
                "key": f"{regime_id}:{seed}:{history}:{_lambda_label(lambda_value)}",
                "match": actual == expected,
                "llm_calls": 0,
            }
        )
    return report


def _config(seeds, free_bytes, references_hash) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_15",
        "experiment_id": "experiment_015_long_horizon_dose_response",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "normalization": CENTERED_NORMALIZATION,
        "choice_seeds": list(seeds),
        "timesteps": TIMESTEPS,
        "social_lambdas": list(SOCIAL_LAMBDAS),
        "s0_lambdas": list(S0_LAMBDAS),
        "new_lambdas": list(NEW_LAMBDAS),
        "horizons": [{"name": name, "start": start, "stop": stop} for name, start, stop in HORIZONS],
        "primary_endpoint": "H2000",
        "intervention_source": "data/worlds/world_v0_13/heldout_interventions.jsonl",
        "anchor_source": "data/worlds/world_v0_14/run_manifest.json",
        "anchor_manifest_sha256": references_hash,
        "statistical_rules": STATISTICAL_RULES,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": ANALYSIS_BOOTSTRAP_SEED,
        "persistence_min": PERSISTENCE_MIN,
        "classification": "regime mean only; seed rate and intervals are reported separately",
        "cohort_note": "Seeds 50-99 are the Experiment 012 held-out cohort and are not a new validation sample for Experiment 015.",
        "disk_free_bytes_before_run": free_bytes,
        "estimated_output_bytes": ESTIMATED_OUTPUT_BYTES,
        "scope": SCOPE_SENTENCE,
        "llm_calls": 0,
    }
    if payload["utility_atlas_sha256"] != EXPECTED_ATLAS_SHA256:
        raise RuntimeError("utility atlas hash is not the frozen experiment atlas")
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _manifest(destination: Path, config, summary) -> dict[str, Any]:
    names = (
        "world_v0_15_config.json",
        "anchor_lineage_audit.json",
        "seed_window_metrics.jsonl",
        "uncertainty_analysis.json",
        "adjacent_contrasts.json",
        "seed_identity_analysis.json",
        "horizon_onset_analysis.json",
        "experiment_015_summary.json",
    )
    return {
        "experiment_id": summary["experiment_id"],
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "seed_count": summary["seed_count"],
        "timesteps": TIMESTEPS,
        "complete": summary["gates"]["complete"],
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "gates": summary["gates"],
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": summary["trajectory_sha256"],
        "files": {name: _file_sha256(destination / name) for name in names},
    }


def _write_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in metrics:
            for label, level in row["levels"].items():
                record = {
                    "regime": row["regime_id"],
                    "choice_seed": row["choice_seed"],
                    "lambda": level["lambda"],
                    "lambda_label": label,
                    "horizons": level["horizons"],
                    "control_trajectory_sha256": level["control_trajectory_sha256"],
                    "perturbed_trajectory_sha256": level["perturbed_trajectory_sha256"],
                    "max_blend_error": level["max_blend_error"],
                    "llm_calls": 0,
                }
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _trajectory_map(metrics: list[dict[str, Any]]) -> dict[str, str]:
    trajectories = {}
    for row in metrics:
        for level in row["levels"].values():
            label = _lambda_label(level["lambda"])
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:CONTROL:{label}"] = level["control_trajectory_sha256"]
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:PERTURBED:{label}"] = level["perturbed_trajectory_sha256"]
    return trajectories


def _safety_flags(regimes: dict[str, Any]) -> list[bool]:
    flags = []
    for block in regimes.values():
        for histories in block["safety"].values():
            for pair in histories.values():
                flags.append(bool(pair["primary_1901_2000"]["unstable"]))
                flags.append(bool(pair["lineage_401_500"]["unstable"]))
    return flags


def run_experiment(
    atlas: FrozenUtilityAtlas | None = None,
    *,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = HELDOUT_SEEDS,
    timesteps: int = TIMESTEPS,
) -> Path:
    destination = output_dir or OUTPUT_DIR
    _refuse_nonempty_output(destination)
    _refuse_frozen_directories(destination)
    if timesteps != TIMESTEPS or tuple(seeds) != HELDOUT_SEEDS:
        raise RuntimeError("the preregistered run is seeds 50-99 and 2000 timesteps")
    parent = destination if destination.exists() else destination.parent
    free = shutil.disk_usage(parent).free
    if free < MINIMUM_FREE_BYTES or free < ESTIMATED_OUTPUT_BYTES:
        raise RuntimeError("not enough free disk for experiment 015")
    print(f"disk free {free} bytes; estimated output under {ESTIMATED_OUTPUT_BYTES} bytes", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    source = load_frozen_atlas() if atlas is None else atlas
    references_hash = _file_sha256(WORLD_V14_DIR / "run_manifest.json")
    references = json.loads((WORLD_V14_DIR / "run_manifest.json").read_text(encoding="utf-8"))["trajectory_sha256"]
    missing = [key for key in anchor_keys() if key not in references]
    if missing:
        raise RuntimeError("experiment 014 manifest does not contain every required anchor")
    plans = load_intervention_registry()
    initial = load_initial_locations()
    agent_ids = tuple(sorted(initial))
    config = _config(seeds, free, references_hash)
    write_json(destination / "world_v0_15_config.json", config)
    started = time.perf_counter()
    metrics = []
    anchor_rows = []
    safety = []
    donor_cache = {}
    with dose_grid():
        for spec in REGIMES:
            for seed in seeds:
                print(f"anchor {spec['regime_id']} seed {seed}", flush=True)
                record = _collect_anchor_seed(source, initial, spec, seed, plans[(spec["regime_id"], seed)], agent_ids, references)
                anchor_rows.extend(record["anchors"])
                safety.extend(record["safety"])
                if spec["regime_id"] == "S0":
                    record.pop("packed_donors", None)
                else:
                    donor_cache[(spec["regime_id"], seed)] = record.pop("packed_donors")
                record.pop("anchors", None)
                record.pop("safety", None)
                metrics.append(record)
        audit = build_anchor_audit(anchor_rows)
        write_json(destination / "anchor_lineage_audit.json", audit)
        if not new_lambdas_allowed(audit):
            raise RuntimeError("anchor lineage failed; new lambdas were not simulated")
        for spec in REGIMES:
            if spec["regime_id"] == "S0":
                continue
            for record in metrics:
                if record["regime_id"] != spec["regime_id"]:
                    continue
                print(f"new lambda {spec['regime_id']} seed {record['choice_seed']}", flush=True)
                record["packed_donors"] = donor_cache.pop((spec["regime_id"], record["choice_seed"]))
                safety.extend(
                    _add_new_lambdas(source, initial, spec, record, plans[(spec["regime_id"], record["choice_seed"])], agent_ids)
                )
    if donor_cache:
        raise RuntimeError("donor cache was not consumed")
    for record in metrics:
        record.pop("packed_donors", None)
        record.pop("rng_draw_counts", None)
        record.pop("prefix_sha256", None)
        if set(record["levels"]) != {_lambda_label(value) for value in lambdas_for(record["regime_id"])}:
            raise RuntimeError("a preregistered lambda is missing")
    print("replay", flush=True)
    with dose_grid():
        replay = _replay(source, initial, plans, metrics)
    regimes = {}
    uncertainties = {}
    for spec in REGIMES:
        rows = [row for row in metrics if row["regime_id"] == spec["regime_id"]]
        uncertainties[spec["regime_id"]] = _uncertainty(rows, lambdas_for(spec["regime_id"]))
        regimes[spec["regime_id"]] = _regime_summary(spec, rows, safety, uncertainties[spec["regime_id"]])
    numeric_rows = [row["numeric"] for row in metrics if row["regime_id"] == "EXP99"]
    numeric = _numeric_summary(numeric_rows)
    safety_flags = _safety_flags(regimes)
    replay_ok = all(item["match"] for item in replay)
    world_count = sum(2 * len(lambdas_for(spec["regime_id"])) * len(seeds) for spec in REGIMES)
    gates = {
        "anchor_lineage_600": True,
        "anchor_count": 600,
        "s0_lambda_equivalence": True,
        "blend_identity": True,
        "rng_schedule": True,
        "numerical_stability": numeric["passed"],
        "safety_primary_all_stable": not any(safety_flags),
        "exact_replay": replay_ok,
        "complete": world_count == 1000 and numeric["passed"] and replay_ok,
        "llm_calls": 0,
    }
    summary = {
        "experiment_id": "experiment_015_long_horizon_dose_response",
        "world_count": world_count,
        "seed_count": len(seeds),
        "timesteps": TIMESTEPS,
        "llm_calls": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "gates": gates,
        "numerical_stability": numeric,
        "exact_replay": replay,
        "regimes": regimes,
        "conclusion": overall_conclusion(regimes),
        "trajectory_sha256": _trajectory_map(metrics),
        "cohort_note": config["cohort_note"],
        "statistical_rules": STATISTICAL_RULES,
    }
    write_json(destination / "experiment_015_summary.json", summary)
    write_json(destination / "uncertainty_analysis.json", {"experiment_id": summary["experiment_id"], "regimes": uncertainties, "llm_calls": 0})
    write_json(
        destination / "adjacent_contrasts.json",
        {
            "experiment_id": summary["experiment_id"],
            "regimes": {name: block["h2000_paired_contrasts"] for name, block in regimes.items()},
            "llm_calls": 0,
        },
    )
    write_json(
        destination / "seed_identity_analysis.json",
        {
            "experiment_id": summary["experiment_id"],
            "note": "Set relations describe retention, expansion, shrinkage, or replacement. Seed ids are not a causal type.",
            "regimes": {
                name: {
                    "levels": {
                        label: {
                            "horizons": {
                                horizon: {
                                    "persistent_seed_ids": level["horizons"][horizon]["seed_persistent_ids"],
                                    "persistent_count": level["horizons"][horizon]["seed_persistent_count"],
                                    "persistent_rate": level["horizons"][horizon]["seed_persistent_rate"],
                                    "wilson95": level["horizons"][horizon]["wilson95"],
                                    "distribution": level["horizons"][horizon]["distribution"],
                                }
                                for horizon in level["horizons"]
                            },
                            "h500_to_h2000": level["h500_to_h2000"],
                        }
                        for label, level in block["levels"].items()
                    },
                    "adjacent_lambda": block["adjacent_lambda"],
                }
                for name, block in regimes.items()
            },
            "llm_calls": 0,
        },
    )
    write_json(
        destination / "horizon_onset_analysis.json",
        {
            "experiment_id": summary["experiment_id"],
            "grid": list(SOCIAL_LAMBDAS),
            "note": "Bands are adjacent tested points. The interior of a band was not tested, and a band is not a critical lambda.",
            "regimes": {
                name: {
                    "horizon_onsets": block["horizon_onsets"],
                    "long_horizon_class": {label: level["long_horizon_class"] for label, level in block["levels"].items()},
                }
                for name, block in regimes.items()
                if name != "S0"
            },
            "llm_calls": 0,
        },
    )
    _write_metrics(destination / "seed_window_metrics.jsonl", metrics)
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary))
    if not gates["complete"]:
        raise RuntimeError("experiment 015 finished with a failed completion gate")
    print(f"complete worlds={world_count} elapsed={summary['elapsed_seconds']:.1f}s", flush=True)
    return destination


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
