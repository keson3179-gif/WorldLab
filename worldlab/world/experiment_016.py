"""Experiment 016: independent-seed replication of the Experiment 015 long-horizon pattern.

Seeds 100-199 are the formal cohort. Seeds 50, 75, and 99 only verify that this
module reproduces Experiment 015 trajectories. Replication labels are fixed
before those new seeds are simulated. This module does not call a model.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import statistics
import time
from pathlib import Path
from random import Random
from typing import Any

import world.experiment_005 as memory_experiment
import world.experiment_013 as experiment_013
import world.experiment_014 as experiment_014
import world.experiment_015 as experiment_015
from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import trajectory_hash
from world.experiment_005 import CENTERED_NORMALIZATION, FAMILIARITY_WEIGHT
from world.experiment_009 import classify_persistence
from world.experiment_012 import HISTORIES, REGIMES
from world.experiment_013 import (
    ENVIRONMENT_STEP,
    OVERRIDE_COUNT,
    build_intervention_record,
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
    load_intervention_registry,
    long_horizon_class,
)
from world.experiment_015 import (
    ANALYSIS_BOOTSTRAP_SEED,
    BOOTSTRAP_DRAWS,
    PERSISTENCE_MIN,
    S0_LAMBDAS,
    SOCIAL_LAMBDAS,
    ZERO_NOTE,
    _release,
    _simulate,
    _window_record,
    dose_grid,
    horizon_onset,
    lambdas_for,
    paired_mean_intervals,
    set_relation,
    wilson_interval,
)
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_16"
WORLD_V15_DIR = experiment_015.OUTPUT_DIR
FORMAL_SEEDS = tuple(range(100, 200))
LINEAGE_SEEDS = (50, 75, 99)
TIMESTEPS = experiment_015.TIMESTEPS
ESTIMATED_OUTPUT_BYTES = 2_000_000_000
MINIMUM_FREE_BYTES = 40_000_000_000
TRACE_SCAN_LIMIT = 2_000_000
REPLAYS = (
    ("RW50", "PERTURBED", 0.4, 100),
    ("RW50", "CONTROL", 0.6, 125),
    ("EXP99", "PERTURBED", 0.8, 199),
    ("EXP99", "CONTROL", 1.0, 100),
    ("S0", "CONTROL", 0.0, 100),
    ("S0", "CONTROL", 1.0, 100),
)
HYPER_NOTE = (
    "The hypergeometric interval is a central 95 percent reference for the overlap of two seed sets "
    "when labels are exchangeable and the set sizes are fixed. It is not a confidence interval for an effect, "
    "it does not identify a persistence mechanism, and a different Jaccard alone does not prove a difference in identity stability."
)
REPLICATION_RULES = (
    "H2000 onset replicates only when D at lambda 0.40 is below 0.025 and D at 0.60, 0.80, and 1.00 are all at least 0.025. "
    "Both regimes: FULL_ONSET_REPLICATION. One regime: PARTIAL_ONSET_REPLICATION. Neither: ONSET_NOT_REPLICATED. "
    "CLASS_PATTERN_REPLICATED additionally requires attenuated, persistent residual, persistent residual, and persistent residual. "
    "Adjacent direction uses the signs of the three H2000 mean differences. "
    "A paired interval lying above 0 does not change the onset or class labels. "
    "An independent-cohort interval that contains 0 is not evidence that replication succeeded. "
    "The primary identity diagnostic is EXP99 at H2000 from lambda 0.40 to 0.60. "
    "AP>0 and PA>0 describes replacement and does not decide onset replication. "
    "Formal classification uses seeds 100-199 only."
)
SCOPE_SENTENCE = (
    "This is a 2000-step stochastic replication under one frozen architecture and the preregistered lambda grid. "
    "It is not a new population, an independent model, permanent path dependence, a critical point, a phase transition, "
    "a proof of metastability, human behavioral validity, or external validity. "
    "Intervals are pointwise descriptions of seed variation and are not simultaneous 95 percent coverage."
)


def _lambda_label(value: float) -> str:
    return f"{value:.2f}"


def formal_world_count(seeds: tuple[int, ...] = FORMAL_SEEDS) -> int:
    return sum(len(HISTORIES) * len(lambdas_for(spec["regime_id"])) * len(seeds) for spec in REGIMES)


def lineage_keys() -> list[str]:
    keys = []
    for seed in LINEAGE_SEEDS:
        for regime_id in ("RW50", "EXP99", "S0"):
            for lambda_value in lambdas_for(regime_id):
                for history in HISTORIES:
                    keys.append(f"{regime_id}:{seed}:{history}:{_lambda_label(lambda_value)}")
    return keys


def onset_replication(levels: dict[float, float]) -> str:
    if set(levels) != set(SOCIAL_LAMBDAS):
        raise RuntimeError("onset replication uses the fixed social grid")
    replicated = (
        levels[0.4] < PERSISTENCE_MIN
        and levels[0.6] >= PERSISTENCE_MIN
        and levels[0.8] >= PERSISTENCE_MIN
        and levels[1.0] >= PERSISTENCE_MIN
    )
    return "ONSET_REPLICATED" if replicated else "ONSET_NOT_REPLICATED"


def overall_onset_replication(labels: dict[str, str]) -> str:
    flags = [labels["RW50"] == "ONSET_REPLICATED", labels["EXP99"] == "ONSET_REPLICATED"]
    if all(flags):
        return "FULL_ONSET_REPLICATION"
    if any(flags):
        return "PARTIAL_ONSET_REPLICATION"
    return "ONSET_NOT_REPLICATED"


def class_pattern_replicated(classes: dict[float, str]) -> bool:
    return (
        classes[0.4] == "attenuated"
        and classes[0.6] == "persistent residual"
        and classes[0.8] == "persistent residual"
        and classes[1.0] == "persistent residual"
    )


def adjacent_direction(differences: list[float]) -> str:
    if len(differences) != 3:
        raise RuntimeError("adjacent direction uses the three fixed contrasts")
    positive = [value > 0.0 for value in differences]
    if all(positive):
        return "ALL_ADJACENT_DIRECTIONS_REPLICATED"
    if any(positive):
        return "PARTIAL_ADJACENT_DIRECTIONS"
    return "NO_POSITIVE_ADJACENT_DIRECTION"


def replacement_reproduced(counts: dict[str, int]) -> bool:
    return counts["AP"] > 0 and counts["PA"] > 0


def hypergeometric_central_95(n: int, k_a: int, k_b: int) -> tuple[int, int]:
    if n <= 0 or not 0 <= k_a <= n or not 0 <= k_b <= n:
        raise RuntimeError("hypergeometric reference requires set sizes inside the cohort")
    lower_support = max(0, k_a + k_b - n)
    upper_support = min(k_a, k_b)
    logs = [
        math.lgamma(k_a + 1)
        - math.lgamma(overlap + 1)
        - math.lgamma(k_a - overlap + 1)
        + math.lgamma(n - k_a + 1)
        - math.lgamma(k_b - overlap + 1)
        - math.lgamma(n - k_a - (k_b - overlap) + 1)
        - (math.lgamma(n + 1) - math.lgamma(k_b + 1) - math.lgamma(n - k_b + 1))
        for overlap in range(lower_support, upper_support + 1)
    ]
    peak = max(logs)
    weights = [math.exp(log - peak) for log in logs]
    total = math.fsum(weights)
    cumulative = 0.0
    lower = upper_support
    upper = upper_support
    lower_found = False
    for overlap, weight in zip(range(lower_support, upper_support + 1), weights, strict=True):
        cumulative += weight / total
        if not lower_found and cumulative >= 0.025:
            lower = overlap
            lower_found = True
        if cumulative >= 0.975:
            upper = overlap
            break
    return lower, upper


def identity_reference(early: list[bool], late: list[bool]) -> dict[str, Any]:
    relation = set_relation(early, late)
    n = len(early)
    k_a = sum(early)
    k_b = sum(late)
    observed = relation["counts"]["PP"]
    low, high = hypergeometric_central_95(n, k_a, k_b)
    relation.update(
        {
            "n": n,
            "k_a": k_a,
            "k_b": k_b,
            "observed_overlap": observed,
            "expected_overlap": (k_a * k_b / n) if n else None,
            "hypergeometric_central_95": [low, high],
            "replacement_reproduced": replacement_reproduced(relation["counts"]),
            "reference_note": HYPER_NOTE,
        }
    )
    return relation


def _independent_indexes(old_n: int, new_n: int, *, draws: int, seed: int) -> list[tuple[list[int], list[int]]]:
    sampler = Random(seed)
    draws_out = []
    for _draw in range(draws):
        old_index = [sampler.randrange(old_n) for _index in range(old_n)]
        new_index = [sampler.randrange(new_n) for _index in range(new_n)]
        draws_out.append((old_index, new_index))
    return draws_out


def independent_mean_differences(
    old_columns: list[list[float]],
    new_columns: list[list[float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = ANALYSIS_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    if len(old_columns) != len(new_columns) or not old_columns:
        raise RuntimeError("independent comparison requires paired columns across cohorts")
    old_n = len(old_columns[0])
    new_n = len(new_columns[0])
    if any(len(column) != old_n for column in old_columns) or any(len(column) != new_n for column in new_columns):
        raise RuntimeError("each cohort must keep one seed order across lambda and horizon")
    if old_n == new_n:
        raise RuntimeError("the historical and replication cohorts are different sizes and are not paired by seed id")
    samples = [[] for _column in old_columns]
    for old_index, new_index in _independent_indexes(old_n, new_n, draws=draws, seed=seed):
        for cell, (old_column, new_column) in enumerate(zip(old_columns, new_columns, strict=True)):
            old_mean = statistics.fmean(old_column[index] for index in old_index)
            new_mean = statistics.fmean(new_column[index] for index in new_index)
            samples[cell].append(new_mean - old_mean)
    intervals = []
    for old_column, new_column, draws_for_column in zip(old_columns, new_columns, samples, strict=True):
        point = statistics.fmean(new_column) - statistics.fmean(old_column)
        degenerate = all(value == 0.0 for value in draws_for_column)
        ordered = sorted(draws_for_column)
        low = ordered[int(0.025 * (draws - 1))]
        high = ordered[int(0.975 * (draws - 1))]
        if degenerate:
            low, high = 0.0, 0.0
        note = "A confidence interval containing 0 does not establish replication."
        if degenerate:
            note = f"{ZERO_NOTE} {note}"
        intervals.append(
            {
                "mean_difference": point,
                "ci95_low": low,
                "ci95_high": high,
                "degenerate_zero": degenerate,
                "old_n": old_n,
                "new_n": new_n,
                "note": note,
            }
        )
    return intervals


def _audit_display_path(path: Path, scan_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return resolved.relative_to(scan_root.resolve()).as_posix()


def _trajectory_seed(directory_name: str, key: str) -> int | None:
    parts = key.split(":")
    if directory_name == "world_v0_6":
        if len(parts) == 3 and parts[2].isdigit():
            return int(parts[2])
        return None
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _collect_choice_seeds(node: Any, found: set[int]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "choice_seeds" and isinstance(value, list):
                found.update(int(item) for item in value)
            elif key == "choice_seed" and isinstance(value, int):
                found.add(value)
            else:
                _collect_choice_seeds(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_choice_seeds(item, found)


def audit_seeds(data_root: Path | None = None) -> dict[str, Any]:
    root = data_root or (PROJECT_ROOT / "data")
    proposed = set(FORMAL_SEEDS)
    sources = []
    overlaps = []
    unchecked = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".jsonl"}:
            continue
        size = path.stat().st_size
        if path.suffix == ".jsonl" and size > TRACE_SCAN_LIMIT:
            unchecked.append({"path": _audit_display_path(path, root), "bytes": size, "reason": "large trace; seed index taken from its config and manifest"})
            continue
        if path.suffix == ".json" and size > 20_000_000:
            unchecked.append({"path": _audit_display_path(path, root), "bytes": size, "reason": "json above the index scan limit"})
            overlaps.append(_audit_display_path(path, root))
            continue
        choice_seeds: set[int] = set()
        trajectory_seeds: set[int] = set()
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            _collect_choice_seeds(payload, choice_seeds)
            trajectories = payload.get("trajectory_sha256") if isinstance(payload, dict) else None
            if isinstance(trajectories, dict):
                for key in trajectories:
                    seed = _trajectory_seed(path.parent.name, key)
                    if seed is not None:
                        trajectory_seeds.add(seed)
        else:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict) and "choice_seed" in row:
                        choice_seeds.add(int(row["choice_seed"]))
        relevant = choice_seeds or trajectory_seeds
        if not relevant and path.suffix == ".jsonl":
            continue
        record = {
            "path": _audit_display_path(path, root),
            "choice_seed_count": len(choice_seeds),
            "choice_seed_min": min(choice_seeds) if choice_seeds else None,
            "choice_seed_max": max(choice_seeds) if choice_seeds else None,
            "trajectory_seed_count": len(trajectory_seeds),
            "trajectory_seed_min": min(trajectory_seeds) if trajectory_seeds else None,
            "trajectory_seed_max": max(trajectory_seeds) if trajectory_seeds else None,
        }
        if path.parent.name == "world_v0_6":
            record["trajectory_note"] = "world_v0_6 keys are architecture:social_weight:choice_seed; 100 and 150 are weights, and the choice seed is the third field"
        hit = sorted((choice_seeds | trajectory_seeds) & proposed)
        if hit:
            record["overlap"] = hit
            overlaps.append(record["path"])
        if relevant:
            sources.append(record)
    passed = not overlaps
    return {
        "experiment_id": "experiment_016_independent_seed_replication",
        "proposed_choice_seeds": list(FORMAL_SEEDS),
        "passed": passed,
        "overlap_sources": overlaps,
        "sources": sources,
        "unchecked_large_traces": unchecked,
        "judgment": (
            "Seeds 100-199 do not appear as choice seeds in the checked configs, manifests, or small seed indexes. "
            "No substantive persistence result for these seeds was available, and the grid, thresholds, horizon, and seed range were fixed before this run."
            if passed
            else "The proposed seeds collide with an existing choice-seed index, or a required index could not be read."
        ),
        "design_fixed_before_new_seed_results": True,
        "llm_calls": 0,
    }


def _refuse_nonempty_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("experiment 016 refuses to write into a non-empty output directory")


def _refuse_frozen_directories(destination: Path) -> None:
    paths = [getattr(experiment_014, f"WORLD_V{index:02d}_DIR") for index in range(1, 14)]
    paths.extend(
        (
            experiment_014.OUTPUT_DIR,
            experiment_015.OUTPUT_DIR,
            experiment_014.ATLAS_DIR,
            experiment_014.CALIBRATION_DIR,
        )
    )
    if destination.resolve() in {path.resolve() for path in paths}:
        raise RuntimeError("experiment 016 must not write into frozen world outputs")


def _run_seed(atlas, initial, spec, seed, plan, references: dict[str, str] | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    numeric = _empty_numeric()

    def watch(logged, current, memory, actions, _audit) -> None:
        _observe_memory(spec, memory, numeric)

    donors: dict[str, Any] = {}
    stored = {}
    for history in HISTORIES:
        result = _simulate(
            atlas,
            initial,
            spec,
            seed,
            history,
            1.0,
            plan,
            None,
            False,
            watch if spec["regime_id"] == "EXP99" else None,
        )
        _observe_scores(result["endogenous_field"], numeric)
        donors[history] = result["endogenous_field"]
        stored[history] = result
    if spec["regime_id"] == "EXP99" and (numeric["nonfinite"] or numeric["exploded"] or numeric["max_abs"] > ACCUMULATOR_LIMIT):
        raise RuntimeError(f"{spec['regime_id']} seed {seed} failed the numerical stability audit")
    base_rng = list(stored["CONTROL"]["rng_draw_counts"])
    prefixes = {history: trajectory_hash(stored[history]["series"][:ENVIRONMENT_STEP]) for history in HISTORIES}
    canonicals = {history: stored[history]["canonical_dynamics_sha256"] for history in HISTORIES}
    levels = {}
    safety = []
    audits = []
    for lambda_value in lambdas_for(spec["regime_id"]):
        pair = {}
        for history in HISTORIES:
            if lambda_value == 1.0:
                result = stored[history]
            else:
                result = _simulate(atlas, initial, spec, seed, history, lambda_value, plan, donors, True, None)
            if result["llm_calls"] != 0:
                raise RuntimeError("experiment 016 called a model")
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
                    raise RuntimeError("S0 lambda levels are not dynamically equivalent")
        windows = _window_record(pair["CONTROL"], pair["PERTURBED"])
        hashes = {}
        for history, result in pair.items():
            key = f"{spec['regime_id']}:{seed}:{history}:{_lambda_label(lambda_value)}"
            actual = result["trajectory_sha256"]
            if references is not None:
                audits.append({"key": key, "expected": references[key], "actual": actual, "match": actual == references[key]})
            hashes[history] = actual
            _release(result)
        levels[_lambda_label(lambda_value)] = {
            "lambda": lambda_value,
            "horizons": windows,
            "control_trajectory_sha256": hashes["CONTROL"],
            "perturbed_trajectory_sha256": hashes["PERTURBED"],
            "max_blend_error": max(pair[history]["max_blend_error"] for history in HISTORIES),
        }
    del donors
    record = {
        "regime_id": spec["regime_id"],
        "choice_seed": seed,
        "levels": levels,
        "numeric": numeric if spec["regime_id"] == "EXP99" else None,
        "llm_calls": 0,
    }
    return record, safety, audits


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: item["choice_seed"])


def _columns(rows: list[dict[str, Any]], lambdas: tuple[float, ...]) -> tuple[list[list[float]], list[str], list[str]]:
    ordered = _ordered(rows)
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
                    row["levels"][later_label]["horizons"]["H2000"]["D_horizon"] - row["levels"][earlier_label]["horizons"]["H2000"]["D_horizon"]
                    for row in ordered
                ]
            )
            contrast = f"H2000:{later_label}-{earlier_label}"
            labels.append(contrast)
            contrasts.append(contrast)
    return columns, labels, contrasts


def _mean_columns(rows: list[dict[str, Any]], lambdas: tuple[float, ...]) -> tuple[list[list[float]], list[str]]:
    columns, labels, contrasts = _columns(rows, lambdas)
    kept = [index for index, label in enumerate(labels) if label not in contrasts]
    return [columns[index] for index in kept], [labels[index] for index in kept]


def _uncertainty(rows: list[dict[str, Any]], lambdas: tuple[float, ...]) -> dict[str, Any]:
    columns, labels, contrasts = _columns(rows, lambdas)
    intervals = paired_mean_intervals(columns)
    return {"cells": {label: interval for label, interval in zip(labels, intervals, strict=True)}, "h2000_contrasts": contrasts}


def _horizon_block(rows: list[dict[str, Any]], lambda_value: float, horizon: str, interval: dict[str, Any]) -> dict[str, Any]:
    ordered = _ordered(rows)
    label = _lambda_label(lambda_value)
    values = [row["levels"][label]["horizons"][horizon]["D_horizon"] for row in ordered]
    ids = [row["choice_seed"] for row in ordered if row["levels"][label]["horizons"][horizon]["seed_persistent"]]
    low, high = wilson_interval(len(ids), len(rows))
    mean = statistics.fmean(values)
    return {
        "D_horizon": mean,
        "persistence_class": classify_persistence(mean),
        "mean_ci95": [interval["ci95_low"], interval["ci95_high"]],
        "mean_ci_degenerate_zero": interval["degenerate_zero"],
        "mean_ci_note": interval["note"],
        "distribution": _dist(values),
        "seed_persistent_count": len(ids),
        "seed_persistent_rate": len(ids) / len(rows),
        "seed_persistent_ids": ids,
        "wilson95": [low, high],
        "n": len(rows),
    }


def _flags(rows: list[dict[str, Any]], lambda_value: float, horizon: str) -> list[bool]:
    label = _lambda_label(lambda_value)
    return [row["levels"][label]["horizons"][horizon]["seed_persistent"] for row in _ordered(rows)]


def _regime_summary(spec, rows, worlds, uncertainty) -> dict[str, Any]:
    lambdas = lambdas_for(spec["regime_id"])
    levels = {}
    for lambda_value in lambdas:
        label = _lambda_label(lambda_value)
        horizons = {
            name: _horizon_block(rows, lambda_value, name, uncertainty["cells"][f"{name}:{label}"])
            for name, _start, _stop in HORIZONS
        }
        levels[label] = {
            "lambda": lambda_value,
            "horizons": horizons,
            "long_horizon_class": long_horizon_class([horizons[name]["D_horizon"] for name, _start, _stop in HORIZONS]),
            "long_horizon_note": "LH-B means all four named windows meet the regime-mean threshold. It does not mean every timestep is persistent.",
            "h500_to_h2000": identity_reference(_flags(rows, lambda_value, "H500"), _flags(rows, lambda_value, "H2000")),
            "max_blend_error": max(row["levels"][label]["max_blend_error"] for row in rows),
        }
    adjacent = {}
    onsets = None
    if lambdas == SOCIAL_LAMBDAS:
        for name, _start, _stop in HORIZONS:
            adjacent[name] = {}
            for later, earlier in ((0.6, 0.4), (0.8, 0.6), (1.0, 0.8)):
                key = f"{_lambda_label(earlier)}->{_lambda_label(later)}"
                adjacent[name][key] = identity_reference(_flags(rows, earlier, name), _flags(rows, later, name))
        onsets = {
            name: horizon_onset({value: levels[_lambda_label(value)]["horizons"][name]["D_horizon"] for value in SOCIAL_LAMBDAS})
            for name, _start, _stop in HORIZONS
        }
    return {
        "n_seeds": len(rows),
        "levels": levels,
        "adjacent_lambda": adjacent,
        "horizon_onsets": onsets,
        "h2000_paired_contrasts": {label: uncertainty["cells"][label] for label in uncertainty["h2000_contrasts"]},
        "safety": {
            history: {
                _lambda_label(lambda_value): _safety_pair(worlds, spec["regime_id"], history, lambda_value)
                for lambda_value in lambdas
            }
            for history in HISTORIES
        },
        "llm_calls": 0,
    }


def _replication(regimes: dict[str, Any]) -> dict[str, Any]:
    onset_labels = {}
    class_labels = {}
    directions = {}
    for regime_id in ("RW50", "EXP99"):
        block = regimes[regime_id]
        means = {level["lambda"]: level["horizons"]["H2000"]["D_horizon"] for level in block["levels"].values()}
        classes = {level["lambda"]: level["horizons"]["H2000"]["persistence_class"] for level in block["levels"].values()}
        contrasts = [
            block["h2000_paired_contrasts"][f"H2000:{_lambda_label(later)}-{_lambda_label(earlier)}"]
            for later, earlier in ((0.6, 0.4), (0.8, 0.6), (1.0, 0.8))
        ]
        onset_labels[regime_id] = onset_replication(means)
        class_labels[regime_id] = class_pattern_replicated(classes)
        directions[regime_id] = {
            "label": adjacent_direction([item["mean"] for item in contrasts]),
            "ci_entirely_above_zero": [item["ci95_low"] > 0.0 for item in contrasts],
            "differences": contrasts,
        }
    primary = regimes["EXP99"]["adjacent_lambda"]["H2000"]["0.40->0.60"]
    return {
        "onset": {
            "RW50": {"label": onset_labels["RW50"], "grid_onset": regimes["RW50"]["horizon_onsets"]["H2000"]},
            "EXP99": {"label": onset_labels["EXP99"], "grid_onset": regimes["EXP99"]["horizon_onsets"]["H2000"]},
            "overall": overall_onset_replication(onset_labels),
        },
        "class_pattern": class_labels,
        "adjacent_direction": directions,
        "primary_identity": {
            "cell": "EXP99 H2000 lambda 0.40 to 0.60",
            "relation": primary,
            "replacement_reproduced": primary["replacement_reproduced"],
            "decides_onset_replication": False,
        },
        "llm_calls": 0,
    }


def _cohort_comparison(old_rows: dict[str, list[dict[str, Any]]], new_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    report = {}
    for regime_id, lambdas in (("RW50", SOCIAL_LAMBDAS), ("EXP99", SOCIAL_LAMBDAS), ("S0", S0_LAMBDAS)):
        old_columns, labels = _mean_columns(old_rows[regime_id], lambdas)
        new_columns, new_labels = _mean_columns(new_rows[regime_id], lambdas)
        if labels != new_labels:
            raise RuntimeError("cohort comparison columns are not aligned")
        intervals = independent_mean_differences(old_columns, new_columns)
        report[regime_id] = {label: interval for label, interval in zip(labels, intervals, strict=True)}
    return {
        "experiment_id": "experiment_016_independent_seed_replication",
        "note": "Seeds 50-99 and 100-199 are independent samples. Indexes are drawn separately. A difference interval containing 0 is not a replication success.",
        "regimes": report,
        "llm_calls": 0,
    }


def _load_previous(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item["regime"], int(item["choice_seed"]))
            row = grouped.setdefault(key, {"regime_id": item["regime"], "choice_seed": int(item["choice_seed"]), "levels": {}})
            row["levels"][item["lambda_label"]] = item
    by_regime: dict[str, list[dict[str, Any]]] = {}
    for (regime_id, _seed), row in grouped.items():
        by_regime.setdefault(regime_id, []).append(row)
    for regime_id, rows in by_regime.items():
        rows.sort(key=lambda item: item["choice_seed"])
        if [row["choice_seed"] for row in rows] != list(range(50, 100)):
            raise RuntimeError("experiment 015 metrics are not the seeds 50-99 comparison cohort")
    return by_regime


def _replay(atlas, initial, plans, metrics) -> list[dict[str, Any]]:
    stored = {}
    for row in metrics:
        for level in row["levels"].values():
            for history, field in (("CONTROL", "control_trajectory_sha256"), ("PERTURBED", "perturbed_trajectory_sha256")):
                stored[(row["regime_id"], history, level["lambda"], row["choice_seed"])] = level[field]
    report = []
    for regime_id, history, lambda_value, seed in REPLAYS:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        donors = {}
        donor_hash = {}
        for donor_history in HISTORIES:
            result = _simulate(atlas, initial, spec, seed, donor_history, 1.0, plans[(regime_id, seed)], None, False, None)
            donors[donor_history] = result["endogenous_field"]
            donor_hash[donor_history] = result["trajectory_sha256"]
            _release(result)
        if lambda_value == 1.0:
            actual = donor_hash[history]
        else:
            result = _simulate(atlas, initial, spec, seed, history, lambda_value, plans[(regime_id, seed)], donors, True, None)
            actual = result["trajectory_sha256"]
            _release(result)
        expected = stored[(regime_id, history, lambda_value, seed)]
        report.append({"key": f"{regime_id}:{seed}:{history}:{_lambda_label(lambda_value)}", "match": actual == expected, "llm_calls": 0})
        del donors
    return report


def _build_registry(atlas, initial, seeds: tuple[int, ...]) -> dict[tuple[str, int], dict[str, Any]]:
    registry = {}
    for spec in REGIMES:
        for seed in seeds:
            record = build_intervention_record(atlas, initial, spec, seed)
            if int(record["override_count"]) != OVERRIDE_COUNT or int(record["override_consumes_rng"]) != 0:
                raise RuntimeError("new intervention registry does not keep the frozen override contract")
            if int(record["choice_seed"]) != seed:
                raise RuntimeError("intervention record was assigned to the wrong seed")
            registry[(spec["regime_id"], seed)] = record
    return registry


def _registry_token(registry: dict[tuple[str, int], dict[str, Any]]) -> str:
    rows = [registry[(spec["regime_id"], seed)]["registry_sha256"] for spec in REGIMES for seed in FORMAL_SEEDS]
    return canonical_sha256(rows)


def _config(free_bytes: int, audit: dict[str, Any]) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_16",
        "experiment_id": "experiment_016_independent_seed_replication",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "normalization": CENTERED_NORMALIZATION,
        "formal_choice_seeds": list(FORMAL_SEEDS),
        "lineage_choice_seeds": list(LINEAGE_SEEDS),
        "timesteps": TIMESTEPS,
        "social_lambdas": list(SOCIAL_LAMBDAS),
        "s0_lambdas": list(S0_LAMBDAS),
        "horizons": [{"name": name, "start": start, "stop": stop} for name, start, stop in HORIZONS],
        "primary_endpoint": "H2000",
        "replication_rules": REPLICATION_RULES,
        "statistical_rules": experiment_015.STATISTICAL_RULES,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": ANALYSIS_BOOTSTRAP_SEED,
        "persistence_min": PERSISTENCE_MIN,
        "classification": "regime mean only; seed rate and intervals do not replace the mean",
        "cohort_note": "Seeds 50-99 remain the historical comparison cohort. Seeds 100-199 are the only formal replication sample.",
        "anchor_source": "data/worlds/world_v0_15/run_manifest.json",
        "anchor_manifest_sha256": _file_sha256(WORLD_V15_DIR / "run_manifest.json"),
        "previous_metrics_sha256": _file_sha256(WORLD_V15_DIR / "seed_window_metrics.jsonl"),
        "seed_audit_passed": audit["passed"],
        "python": platform.python_version(),
        "implementation_sha256": _file_sha256(Path(__file__)),
        "requirements_lock_sha256": _file_sha256(PROJECT_ROOT / "requirements-lock.txt"),
        "disk_free_bytes_before_run": free_bytes,
        "estimated_output_bytes": ESTIMATED_OUTPUT_BYTES,
        "scope": SCOPE_SENTENCE,
        "llm_calls": 0,
    }
    if payload["utility_atlas_sha256"] != experiment_014.EXPECTED_ATLAS_SHA256:
        raise RuntimeError("utility atlas hash is not the frozen experiment atlas")
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


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


def _write_registry(path: Path, registry: dict[tuple[str, int], dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for spec in REGIMES:
            for seed in FORMAL_SEEDS:
                record = registry[(spec["regime_id"], seed)]
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _trajectory_map(metrics: list[dict[str, Any]]) -> dict[str, str]:
    trajectories = {}
    for row in metrics:
        for level in row["levels"].values():
            label = _lambda_label(level["lambda"])
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:CONTROL:{label}"] = level["control_trajectory_sha256"]
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:PERTURBED:{label}"] = level["perturbed_trajectory_sha256"]
    return trajectories


def _safety_stable(regimes: dict[str, Any], window: str) -> bool:
    for block in regimes.values():
        for histories in block["safety"].values():
            for pair in histories.values():
                if pair[window]["unstable"]:
                    return False
    return True


def _conclusion(replication: dict[str, Any]) -> str:
    onset = replication["onset"]["overall"]
    classes = replication["class_pattern"]
    directions = {name: item["label"] for name, item in replication["adjacent_direction"].items()}
    identity = replication["primary_identity"]["replacement_reproduced"]
    return " ".join(
        [
            f"The preregistered H2000 onset pattern is {onset}.",
            "Onset labels do not mean that every secondary pattern replicated.",
            f"H2000 three-class pattern replicated for RW50: {classes['RW50']}; EXP99: {classes['EXP99']}.",
            f"Adjacent mean directions are RW50 {directions['RW50']} and EXP99 {directions['EXP99']}.",
            (
                "The primary EXP99 H2000 lambda 0.40 to 0.60 persistent sets show simultaneous entry and exit."
                if identity
                else "The primary EXP99 H2000 lambda 0.40 to 0.60 comparison does not show simultaneous entry and exit."
            ),
            "Formal classification uses the 100 new seeds only.",
            SCOPE_SENTENCE,
        ]
    )


def run_experiment(
    atlas: FrozenUtilityAtlas | None = None,
    *,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = FORMAL_SEEDS,
    timesteps: int = TIMESTEPS,
) -> Path:
    destination = output_dir or OUTPUT_DIR
    _refuse_nonempty_output(destination)
    _refuse_frozen_directories(destination)
    if timesteps != TIMESTEPS or tuple(seeds) != FORMAL_SEEDS:
        raise RuntimeError("the preregistered replication uses seeds 100-199 and 2000 timesteps")
    if set(FORMAL_SEEDS) & set(range(100)):
        raise RuntimeError("formal seeds overlap the historical cohorts")
    parent = destination if destination.exists() else destination.parent
    free = shutil.disk_usage(parent).free
    if free < MINIMUM_FREE_BYTES or free < ESTIMATED_OUTPUT_BYTES:
        raise RuntimeError("not enough free disk for experiment 016")
    print(f"disk free {free} bytes; estimated compact output under {ESTIMATED_OUTPUT_BYTES} bytes", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    audit = audit_seeds()
    write_json(destination / "seed_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("seed audit failed; the formal cohort was not simulated")
    source = load_frozen_atlas() if atlas is None else atlas
    config = _config(free, audit)
    write_json(destination / "world_v0_16_config.json", config)
    started = time.perf_counter()
    references = json.loads((WORLD_V15_DIR / "run_manifest.json").read_text(encoding="utf-8"))["trajectory_sha256"]
    missing = [key for key in lineage_keys() if key not in references]
    if missing:
        raise RuntimeError("experiment 015 manifest does not contain every lineage key")
    old_plans = load_intervention_registry()
    initial = load_initial_locations()
    lineage_rows = []
    with dose_grid():
        for seed in LINEAGE_SEEDS:
            for spec in REGIMES:
                print(f"lineage {spec['regime_id']} seed {seed}", flush=True)
                _record, _safety, audits = _run_seed(source, initial, spec, seed, old_plans[(spec["regime_id"], seed)], references)
                lineage_rows.extend(audits)
    lineage = {
        "experiment_id": config["experiment_id"],
        "comparison": "full t=0-2000 trajectory hash against experiment 015 for seeds 50, 75, and 99",
        "anchor_count": len(lineage_rows),
        "matched": sum(row["match"] for row in lineage_rows),
        "mismatched": sum(not row["match"] for row in lineage_rows),
        "rows": lineage_rows,
        "llm_calls": 0,
    }
    write_json(destination / "implementation_lineage_audit.json", lineage)
    if lineage["anchor_count"] != 60 or lineage["mismatched"]:
        raise RuntimeError("lineage verification failed; the new cohort was not simulated")
    print("build intervention registry", flush=True)
    with dose_grid():
        registry = _build_registry(source, initial, FORMAL_SEEDS)
        rebuilt = _build_registry(source, initial, FORMAL_SEEDS)
    if _registry_token(registry) != _registry_token(rebuilt):
        raise RuntimeError("rebuilt intervention registry hash does not match")
    _write_registry(destination / "replication_interventions.jsonl", registry)
    write_json(
        destination / "replication_intervention_registry.json",
        {"registry_sha256": _registry_token(registry), "record_count": len(registry), "llm_calls": 0},
    )
    metrics = []
    safety = []
    with dose_grid():
        for spec in REGIMES:
            for seed in FORMAL_SEEDS:
                print(f"formal {spec['regime_id']} seed {seed}", flush=True)
                record, seed_safety, _audits = _run_seed(source, initial, spec, seed, registry[(spec["regime_id"], seed)], None)
                metrics.append(record)
                safety.extend(seed_safety)
    if {row["choice_seed"] for row in metrics} != set(FORMAL_SEEDS):
        raise RuntimeError("formal metrics do not contain exactly seeds 100-199")
    if formal_world_count() != 2000:
        raise RuntimeError("formal world count is not 2000")
    print("replay", flush=True)
    with dose_grid():
        replay = _replay(source, initial, registry, metrics)
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("shared social weight was not restored")
    old_metrics = _load_previous(WORLD_V15_DIR / "seed_window_metrics.jsonl")
    grouped = {spec["regime_id"]: [row for row in metrics if row["regime_id"] == spec["regime_id"]] for spec in REGIMES}
    uncertainties = {spec["regime_id"]: _uncertainty(grouped[spec["regime_id"]], lambdas_for(spec["regime_id"])) for spec in REGIMES}
    regimes = {
        spec["regime_id"]: _regime_summary(spec, grouped[spec["regime_id"]], safety, uncertainties[spec["regime_id"]])
        for spec in REGIMES
    }
    replication = _replication(regimes)
    comparison = _cohort_comparison(old_metrics, grouped)
    numeric = _numeric_summary([row["numeric"] for row in metrics if row["regime_id"] == "EXP99"])
    replay_ok = all(item["match"] for item in replay)
    gates = {
        "seed_audit": True,
        "lineage_60": True,
        "registry_rebuild": True,
        "formal_seed_count": 100,
        "formal_world_count": 2000,
        "s0_lambda_equivalence": True,
        "blend_identity": True,
        "rng_schedule": True,
        "numerical_stability": numeric["passed"],
        "safety_primary_all_stable": _safety_stable(regimes, "primary_1901_2000"),
        "safety_lineage_all_stable": _safety_stable(regimes, "lineage_401_500"),
        "exact_replay": replay_ok,
        "complete": numeric["passed"] and replay_ok,
        "llm_calls": 0,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "world_count": 2000,
        "seed_count": 100,
        "lineage_world_count": 60,
        "timesteps": TIMESTEPS,
        "llm_calls": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "gates": gates,
        "numerical_stability": numeric,
        "exact_replay": replay,
        "replication": replication,
        "regimes": regimes,
        "conclusion": _conclusion(replication),
        "trajectory_sha256": _trajectory_map(metrics),
        "replication_rules": REPLICATION_RULES,
        "scope": SCOPE_SENTENCE,
    }
    write_json(destination / "experiment_016_summary.json", summary)
    write_json(destination / "uncertainty_analysis.json", {"experiment_id": summary["experiment_id"], "regimes": uncertainties, "llm_calls": 0})
    write_json(
        destination / "adjacent_contrasts.json",
        {
            "experiment_id": summary["experiment_id"],
            "regimes": {name: block["h2000_paired_contrasts"] for name, block in regimes.items()},
            "llm_calls": 0,
        },
    )
    write_json(destination / "cohort_comparison.json", comparison)
    write_json(
        destination / "seed_identity_analysis.json",
        {
            "experiment_id": summary["experiment_id"],
            "note": "Seed ids are not compared across cohorts. " + HYPER_NOTE,
            "primary_identity": replication["primary_identity"],
            "regimes": {
                name: {
                    "levels": {
                        label: {"h500_to_h2000": level["h500_to_h2000"], "horizons": {
                            horizon: {
                                "persistent_seed_ids": cell["seed_persistent_ids"],
                                "persistent_count": cell["seed_persistent_count"],
                                "persistent_rate": cell["seed_persistent_rate"],
                                "wilson95": cell["wilson95"],
                                "distribution": cell["distribution"],
                            }
                            for horizon, cell in level["horizons"].items()
                        }}
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
    write_json(destination / "replication_analysis.json", {"experiment_id": summary["experiment_id"], **replication})
    _write_metrics(destination / "seed_window_metrics.jsonl", metrics)
    manifest = {
        "experiment_id": summary["experiment_id"],
        "llm_calls": 0,
        "world_count": 2000,
        "seed_count": 100,
        "lineage_world_count": 60,
        "timesteps": TIMESTEPS,
        "complete": gates["complete"],
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "implementation_sha256": config["implementation_sha256"],
        "python": config["python"],
        "gates": gates,
        "conclusion": summary["conclusion"],
        "exact_replay": replay,
        "trajectory_sha256": summary["trajectory_sha256"],
        "files": {
            name: _file_sha256(destination / name)
            for name in (
                "world_v0_16_config.json",
                "seed_audit.json",
                "replication_interventions.jsonl",
                "replication_intervention_registry.json",
                "implementation_lineage_audit.json",
                "seed_window_metrics.jsonl",
                "uncertainty_analysis.json",
                "adjacent_contrasts.json",
                "cohort_comparison.json",
                "seed_identity_analysis.json",
                "horizon_onset_analysis.json",
                "replication_analysis.json",
                "experiment_016_summary.json",
            )
        },
    }
    write_json(destination / "run_manifest.json", manifest)
    if not gates["complete"]:
        raise RuntimeError("experiment 016 finished with a failed completion gate")
    print(f"complete worlds=2000 elapsed={summary['elapsed_seconds']:.1f}s", flush=True)
    return destination


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
