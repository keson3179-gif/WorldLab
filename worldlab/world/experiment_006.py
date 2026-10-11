"""Experiment 006: social-weight by memory-timescale calibration.

Four centered architectures from Experiment 005, at weights 25, 50, 100, and 150.
CUM stays outside the factorial. Familiarity weight stays 0. This module does not call a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import world.experiment_005 as memory_experiment
from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.dynamics_calibration import OUTPUT_DIR as CALIBRATION_DIR
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_003 import OUTPUT_DIR as WORLD_V03_DIR
from world.experiment_004 import (
    OUTPUT_DIR as WORLD_V04_DIR,
    _entropy,
    _hhi,
    _js_distance,
    _share,
    _tv,
    _unique_leader,
    trajectory_hash,
)
from world.experiment_005 import (
    CENTERED_NORMALIZATION,
    FAMILIARITY_WEIGHT,
    OUTPUT_DIR as WORLD_V05_DIR,
    _between,
    _category_entropy,
    _dominant_counts,
    _make_memory,
    _mean_sd,
    _presence,
    is_unstable,
    simulate_memory,
)
from world.geography import PLACE_IDS, occupancy_of
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_6"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
WEIGHTS = (25.0, 50.0, 100.0, 150.0)
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
ALTERNATION_UPPER_QUARTILE = 0.694375
CONSECUTIVE_TV_UPPER_QUARTILE = 0.36148125
EXTREME_LATE_HHI = 0.40
EXTREME_DOMINANT_SHARE = 0.60
EXTREME_SATURATION = 0.10
ARCHITECTURES = (
    {"condition_id": "RW20", "memory": "rolling", "window": 20, "decay": None},
    {"condition_id": "RW50", "memory": "rolling", "window": 50, "decay": None},
    {"condition_id": "EXP95", "memory": "exponential", "window": None, "decay": 0.95},
    {"condition_id": "EXP99", "memory": "exponential", "window": None, "decay": 0.99},
)
REPLAYS = (
    {"condition_id": "RW20", "social_weight": 50.0, "choice_seed": 0},
    {"condition_id": "RW50", "social_weight": 100.0, "choice_seed": 0},
    {"condition_id": "EXP95", "social_weight": 100.0, "choice_seed": 0},
    {"condition_id": "EXP99", "social_weight": 150.0, "choice_seed": 0},
)
DOSE_OUTCOMES = (
    "ranking_change_rate",
    "destination_change_rate",
    "late_hhi",
    "late_entropy",
    "late_dominant_share",
    "early_leader_retention",
    "between_seed_late_tv",
    "between_seed_late_js_distance",
)
DOSE_FLAT_TOLERANCE = {
    "ranking_change_rate": 0.01,
    "destination_change_rate": 0.01,
    "late_hhi": 0.005,
    "late_entropy": 0.01,
    "late_dominant_share": 0.01,
    "early_leader_retention": 0.02,
    "between_seed_late_tv": 0.005,
    "between_seed_late_js_distance": 0.005,
}
REGRESSION_OUTCOMES = (
    "ranking_change_rate",
    "destination_change_rate",
    "late_hhi",
    "late_entropy",
    "late_dominant_share",
    "early_leader_retention",
    "early_late_tv",
    "early_late_js_distance",
)
SENSITIVITY_OUTCOMES = (
    "ranking_change_rate",
    "destination_change_rate",
    "late_hhi",
    "between_seed_late_tv",
)
ANALYSIS_NOTE = (
    "Factorial calibration of four centered social-memory architectures at social weights 25, 50, 100, and 150. "
    "CUM is excluded because its log1p/max scores are not on this scale. "
    "These labels are simulation operating regimes. This summary does not claim collective path dependence, "
    "human realism, destination emergence, lock-in, or self-organization."
)
REGIME_RULE = (
    "A. weak influence: destination_change_rate < 0.10. "
    "B. moderate influence: destination_change_rate >= 0.10 and < 0.30, and unstable is false. "
    "C. strong-stable influence: destination_change_rate >= 0.30 and unstable is false. "
    "D. unstable influence: unstable is true. "
    "Unstable uses the Experiment 005 thresholds exactly: both site_01 and site_02 lag-1 negative and "
    "alternation >= 0.694375, or late HHI >= 0.40, or late dominant share >= 0.60, or saturation >= 0.10, "
    "or consecutive occupancy TV >= 0.36148125."
)
SELECTION_RULE = (
    "A cell is eligible only when unstable is false, destination_change_rate is inside [0.15, 0.35], "
    "ranking_change_rate > 0.25, clipping rate < 0.10, at least three distinct places are final dominant, "
    "and between-seed late TV is greater than the Experiment 004 S0 baseline. "
    "Ties break by destination_change_rate closest to 0.25, then lower clipping rate, then higher between-seed late TV, "
    "then lower social_weight, then architecture id. "
    "If none qualify, selection is none. This is a technical reference rule, not a realism ranking."
)
DOSE_RULE = (
    "Along weights 25, 50, 100, 150, a trajectory is approximately flat when max(mean)-min(mean) is below "
    "the pre-specified tolerance for that outcome. Otherwise it is monotonic increasing, monotonic decreasing, "
    "or non-monotonic, using the raw cell means and allowing equal adjacent steps inside a monotonic run."
)


@contextmanager
def _weight(value: float) -> Iterator[None]:
    original = memory_experiment.SOCIAL_WEIGHT
    memory_experiment.SOCIAL_WEIGHT = float(value)
    try:
        yield
    finally:
        memory_experiment.SOCIAL_WEIGHT = original


def classify_regime(destination_change_rate: float, unstable: bool) -> str:
    if unstable:
        return "unstable influence"
    if destination_change_rate < 0.10:
        return "weak influence"
    if destination_change_rate < 0.30:
        return "moderate influence"
    return "strong-stable influence"


def dose_shape(values: list[float], tolerance: float) -> str:
    if max(values) - min(values) < tolerance:
        return "approximately flat"
    increasing = all(right >= left for left, right in zip(values, values[1:]))
    decreasing = all(right <= left for left, right in zip(values, values[1:]))
    if increasing:
        return "monotonic increasing"
    if decreasing:
        return "monotonic decreasing"
    return "non-monotonic"


def select_reference(cells: list[dict[str, Any]], baseline_tv: float) -> dict[str, Any]:
    eligible = [cell for cell in cells if _eligible(cell, baseline_tv)]
    ordered = sorted(eligible, key=_selection_key)
    chosen = None if not ordered else {
        "architecture": ordered[0]["architecture"],
        "social_weight": ordered[0]["social_weight"],
        "cell_id": ordered[0]["cell_id"],
        "destination_change_rate": ordered[0]["destination_change_rate"],
        "ranking_change_rate": ordered[0]["ranking_change_rate"],
        "clipping_rate": ordered[0]["clipping_rate"],
        "between_seed_late_tv": ordered[0]["between_seed_late_tv"],
        "final_dominant_places": ordered[0]["final_dominant_places"],
    }
    return {
        "rule": SELECTION_RULE,
        "baseline_between_seed_late_tv": baseline_tv,
        "eligible_count": len(ordered),
        "eligible_cells": [
            {
                "cell_id": cell["cell_id"],
                "destination_change_rate": cell["destination_change_rate"],
                "clipping_rate": cell["clipping_rate"],
                "between_seed_late_tv": cell["between_seed_late_tv"],
                "social_weight": cell["social_weight"],
            }
            for cell in ordered
        ],
        "selection": "none" if chosen is None else chosen,
        "note": "Technical reference selection. It is not a behavioral realism ranking.",
    }


def run_experiment(
    atlas: FrozenUtilityAtlas | None = None,
    *,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = CHOICE_SEEDS,
    timesteps: int = TIMESTEPS,
) -> Path:
    destination = output_dir or OUTPUT_DIR
    _refuse_frozen_directories(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = load_frozen_atlas() if atlas is None else atlas
    _require_atlas_hash()
    baseline = _zero_social_baseline()
    locations = load_initial_locations()
    reference = _load_weight_150_locations()
    config = _config(seeds, timesteps, baseline)
    write_json(destination / "world_v0_6_config.json", config)
    state_path = destination / "world_states.jsonl"
    memory_path = destination / "social_memory_states.jsonl"
    action_path = destination / "agent_actions.jsonl"
    for path in (state_path, memory_path, action_path):
        path.write_text("", encoding="utf-8", newline="\n")
    worlds: list[dict[str, Any]] = []
    ratio_values: dict[tuple[str, float], list[float]] = {}
    undefined_ratios: dict[tuple[str, float], int] = {}
    with state_path.open("a", encoding="utf-8", newline="\n") as state_file, memory_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as memory_file, action_path.open("a", encoding="utf-8", newline="\n") as action_file:
        for spec in ARCHITECTURES:
            for weight in WEIGHTS:
                key = (spec["condition_id"], weight)
                ratio_values[key] = []
                undefined_ratios[key] = 0
                for seed in seeds:
                    _write_initial(state_file, memory_file, spec, weight, seed, locations)
                    signal = {"abs_sum": 0.0, "decisions": 0, "ratios": [], "undefined": 0}
                    with _weight(weight):
                        result = simulate_memory(
                            source,
                            locations,
                            _make_memory(spec, list(locations)),
                            choice_seed_value=seed,
                            timesteps=timesteps,
                            on_step=_log(state_file, memory_file, action_file, spec, weight, seed, signal),
                        )
                    if result["llm_calls"] != 0:
                        raise RuntimeError("experiment 006 called a model")
                    if weight == 150.0 and result["series"] != reference[spec["condition_id"]][seed]:
                        raise RuntimeError(
                            f"{spec['condition_id']} weight 150 diverges from Experiment 005 at choice seed {seed}"
                        )
                    worlds.append(_world_metrics(spec, weight, seed, result, signal, timesteps))
                    ratio_values[key].extend(signal["ratios"])
                    undefined_ratios[key] += signal["undefined"]
                    print(
                        f"{spec['condition_id']} w={weight:g} seed={seed} "
                        f"dest={worlds[-1]['destination_change_rate']:.3f} late_hhi={worlds[-1]['late_hhi']:.3f}",
                        flush=True,
                    )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    replays = _replay_all(source, locations, worlds, seeds, timesteps)
    summary = _summarize(worlds, seeds, timesteps, baseline, replays, ratio_values, undefined_ratios)
    write_json(destination / "experiment_006_summary.json", summary)
    factorial = fit_factorial(worlds)
    write_json(destination / "factorial_analysis.json", factorial)
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, worlds))
    return destination


def fit_factorial(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(worlds)
    outcomes = {}
    for outcome in REGRESSION_OUTCOMES:
        outcomes[outcome] = {
            "numeric_weight": _fit_one(frame, outcome, "social_weight"),
            "categorical_weight": _fit_one(frame, outcome, "C(social_weight)"),
            "cell_mean_interaction": _mean_interaction(frame, outcome),
        }
    return {
        "experiment_id": "experiment_006_social_weight_memory",
        "note": (
            "World-level estimated means, contrasts, and effect sizes. "
            "p-values are secondary and do not decide architecture validity. "
            "Between-seed distances are cell-level summaries and stay in the experiment summary."
        ),
        "n_worlds": len(worlds),
        "reference_architecture": "RW20",
        "outcomes": outcomes,
    }


def _fit_one(frame: pd.DataFrame, outcome: str, weight_term: str) -> dict[str, Any]:
    used = frame.loc[frame[outcome].notna(), ["architecture", "social_weight", outcome]].copy()
    used[outcome] = used[outcome].astype(float)
    formula = f"{outcome} ~ C(architecture, Treatment(reference='RW20')) * {weight_term}"
    fitted = ols(formula, data=used).fit()
    table = anova_lm(fitted, typ=2)
    residual = float(table.loc["Residual", "sum_sq"])
    terms = {}
    for name in table.index:
        if name == "Residual":
            continue
        sum_sq = float(table.loc[name, "sum_sq"])
        terms[_term_label(str(name))] = {
            "sum_sq": sum_sq,
            "df": float(table.loc[name, "df"]),
            "partial_eta_squared": None if sum_sq + residual == 0.0 else sum_sq / (sum_sq + residual),
            "secondary_p_value": _finite(table.loc[name, "PR(>F)"]),
        }
    return {
        "n": int(len(used)),
        "r_squared": _finite(fitted.rsquared),
        "terms": terms,
        "slopes_per_weight_unit": None if weight_term.startswith("C(") else _slopes(fitted),
    }


def _slopes(fitted: Any) -> dict[str, float]:
    params = fitted.params
    base = float(params["social_weight"])
    slopes = {"RW20": base}
    for architecture in ("RW50", "EXP95", "EXP99"):
        interaction = next(
            float(params[name])
            for name in params.index
            if architecture in str(name) and str(name).endswith(":social_weight")
        )
        slopes[architecture] = base + interaction
    return {
        architecture: value
        for architecture, value in slopes.items()
    } | {"per_25_social_weight": {architecture: value * 25.0 for architecture, value in slopes.items()}}


def _mean_interaction(frame: pd.DataFrame, outcome: str) -> dict[str, Any]:
    used = frame.loc[frame[outcome].notna()]
    cells = used.groupby(["architecture", "social_weight"], observed=True)[outcome].mean()
    architecture_mean = used.groupby("architecture", observed=True)[outcome].mean()
    weight_mean = used.groupby("social_weight", observed=True)[outcome].mean()
    grand = float(used[outcome].mean())
    residual = {}
    contrasts_vs_rw20 = {}
    contrasts_vs_weight_25 = {}
    for architecture in [item["condition_id"] for item in ARCHITECTURES]:
        residual[architecture] = {}
        contrasts_vs_rw20[architecture] = {}
        contrasts_vs_weight_25[architecture] = {}
        for weight in WEIGHTS:
            cell = float(cells.loc[(architecture, weight)])
            residual[architecture][str(weight)] = cell - float(architecture_mean.loc[architecture]) - float(weight_mean.loc[weight]) + grand
            contrasts_vs_rw20[architecture][str(weight)] = cell - float(cells.loc[("RW20", weight)])
            contrasts_vs_weight_25[architecture][str(weight)] = cell - float(cells.loc[(architecture, 25.0)])
    return {
        "estimated_cell_means": {
            architecture: {str(weight): float(cells.loc[(architecture, weight)]) for weight in WEIGHTS}
            for architecture in [item["condition_id"] for item in ARCHITECTURES]
        },
        "additive_interaction_residual": residual,
        "contrast_versus_rw20": contrasts_vs_rw20,
        "contrast_versus_weight_25": contrasts_vs_weight_25,
    }


def _world_metrics(
    spec: dict[str, Any],
    weight: float,
    seed: int,
    result: dict[str, Any],
    signal: dict[str, Any],
    timesteps: int,
) -> dict[str, Any]:
    series = result["series"]
    early_counts = _presence(series, 1, 20)
    late_counts = _presence(series, 101, timesteps)
    full_counts = _presence(series, 1, timesteps)
    early_share = _share(early_counts)
    late_share = _share(late_counts)
    early_leader = _unique_leader(early_counts)
    late_leader = _unique_leader(late_counts)
    diagnostics = measure_world(series, moves=result["moves"], timesteps=timesteps)
    bucket = result["bucket"]
    decisions = bucket["decisions"]
    return {
        "cell_id": f"{spec['condition_id']}_w{weight:g}",
        "architecture": spec["condition_id"],
        "memory": spec["memory"],
        "social_weight": weight,
        "choice_seed": seed,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "trajectory_sha256": trajectory_hash(series),
        "social_spread": bucket["spread"] / decisions,
        "ranking_change_rate": bucket["rank"] / decisions,
        "destination_change_rate": None if bucket["reconsidered"] == 0 else bucket["dest"] / bucket["reconsidered"],
        "clipping_rate": bucket["saturated"] / bucket["score_n"],
        "mean_abs_social": signal["abs_sum"] / signal["decisions"],
        "late_hhi": _hhi(late_share),
        "late_entropy": _entropy(late_share),
        "late_dominant_share": max(late_share.values()),
        "late_dominant": late_leader,
        "late_share": late_share,
        "full_dominant": _unique_leader(full_counts),
        "early_leader": early_leader,
        "early_leader_retention": None if early_leader is None else float(late_leader == early_leader),
        "late_share_of_early_leader": None if early_leader is None else late_share[early_leader],
        "early_late_tv": _tv(early_share, late_share),
        "early_late_js_distance": _js_distance(early_share, late_share),
        "movement_rate": diagnostics["movement_rate"],
        "mean_dwell": diagnostics["mean_dwell"],
        "occupancy_entropy": diagnostics["entropy_mean"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "lag1": diagnostics["lag1"],
        "lag2": diagnostics["lag2"],
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "consecutive_occupancy_tv": diagnostics["mean_total_variation"],
        "mean_occupancy": diagnostics["mean_occupancy"],
        "llm_calls": 0,
    }


def _summarize(
    worlds: list[dict[str, Any]],
    seeds: tuple[int, ...],
    timesteps: int,
    baseline: dict[str, Any],
    replays: dict[str, Any],
    ratio_values: dict[tuple[str, float], list[float]],
    undefined_ratios: dict[tuple[str, float], int],
) -> dict[str, Any]:
    cells = {}
    selection_rows = []
    for spec in ARCHITECTURES:
        for weight in WEIGHTS:
            rows = [
                world
                for world in worlds
                if world["architecture"] == spec["condition_id"] and world["social_weight"] == weight
            ]
            block = _cell_summary(rows, ratio_values[(spec["condition_id"], weight)], undefined_ratios[(spec["condition_id"], weight)])
            cells[block["cell_id"]] = block
            selection_rows.append(
                {
                    "cell_id": block["cell_id"],
                    "architecture": spec["condition_id"],
                    "social_weight": weight,
                    "unstable": block["unstable"],
                    "destination_change_rate": block["destination_change_rate"]["mean"],
                    "ranking_change_rate": block["ranking_change_rate"]["mean"],
                    "clipping_rate": block["clipping_rate"]["mean"],
                    "between_seed_late_tv": block["between_seed_late_tv"]["mean"],
                    "final_dominant_places": block["final_dominant_places"],
                }
            )
    return {
        "experiment_id": "experiment_006_social_weight_memory",
        "note": ANALYSIS_NOTE,
        "regime_rule": REGIME_RULE,
        "selection_rule": SELECTION_RULE,
        "dose_rule": DOSE_RULE,
        "dose_flat_tolerance": DOSE_FLAT_TOLERANCE,
        "normalization": CENTERED_NORMALIZATION,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "cum_excluded_from_factorial": True,
        "llm_calls": 0,
        "world_count": len(worlds),
        "cell_count": len(cells),
        "timesteps": timesteps,
        "choice_seeds": list(seeds),
        "weights": list(WEIGHTS),
        "weight_150_matches_experiment_005": True,
        "js_definition": "Jensen-Shannon distance = sqrt(JS divergence), with natural log.",
        "zero_social_baseline": baseline,
        "cells": cells,
        "dose_response": _dose_response(cells),
        "sensitivity_per_25_social_weight": _sensitivity(cells),
        "reference_regime": select_reference(selection_rows, float(baseline["between_seed_late_tv"])),
        "exact_replay": replays,
    }


def _cell_summary(worlds: list[dict[str, Any]], ratios: list[float], undefined: int) -> dict[str, Any]:
    counts = _dominant_counts([world["full_dominant"] for world in worlds])
    lag1 = {place_id: _mean_sd([world["lag1"][place_id] for world in worlds]) for place_id in ("site_01", "site_02")}
    late_hhi = _mean_sd([world["late_hhi"] for world in worlds])
    dominant_share = _mean_sd([world["late_dominant_share"] for world in worlds])
    saturation = _mean_sd([world["saturation_frequency"] for world in worlds])
    alternation = _mean_sd([world["alternation_rate"] for world in worlds])
    consecutive = _mean_sd([world["consecutive_occupancy_tv"] for world in worlds])
    destination = _mean_sd([world["destination_change_rate"] for world in worlds])
    unstable = is_unstable(
        lag1_site_01=lag1["site_01"]["mean"],
        lag1_site_02=lag1["site_02"]["mean"],
        alternation=float(alternation["mean"]),
        alternation_cutoff=ALTERNATION_UPPER_QUARTILE,
        late_hhi=float(late_hhi["mean"]),
        dominant_share=float(dominant_share["mean"]),
        saturation=float(saturation["mean"]),
        consecutive_tv=float(consecutive["mean"]),
        consecutive_tv_cutoff=CONSECUTIVE_TV_UPPER_QUARTILE,
    )
    places = [place_id for place_id in PLACE_IDS if counts[place_id] > 0]
    return {
        "cell_id": worlds[0]["cell_id"],
        "architecture": worlds[0]["architecture"],
        "social_weight": worlds[0]["social_weight"],
        "n_seeds": len(worlds),
        "llm_calls": 0,
        "normalization": CENTERED_NORMALIZATION,
        "social_utility_spread": _mean_sd([world["social_spread"] for world in worlds]),
        "ranking_change_rate": _mean_sd([world["ranking_change_rate"] for world in worlds]),
        "destination_change_rate": destination,
        "clipping_rate": _mean_sd([world["clipping_rate"] for world in worlds]),
        "mean_abs_social_utility": _mean_sd([world["mean_abs_social"] for world in worlds]),
        "social_base_scale_ratio": _ratio_summary(ratios, undefined),
        "late_hhi": late_hhi,
        "late_entropy": _mean_sd([world["late_entropy"] for world in worlds]),
        "late_dominant_share": dominant_share,
        "final_dominant_counts": counts,
        "final_dominant_proportions": {key: value / len(worlds) for key, value in counts.items()},
        "final_dominant_entropy": _category_entropy(
            ["tie" if world["full_dominant"] is None else str(world["full_dominant"]) for world in worlds]
        ),
        "final_dominant_places": places,
        "early_leader_eligible": sum(world["early_leader"] is not None for world in worlds),
        "early_leader_retention": _mean_sd(
            [world["early_leader_retention"] for world in worlds]
        ),
        "late_share_of_early_leader": _mean_sd([world["late_share_of_early_leader"] for world in worlds]),
        "early_late_tv": _mean_sd([world["early_late_tv"] for world in worlds]),
        "early_late_js_distance": _mean_sd([world["early_late_js_distance"] for world in worlds]),
        "between_seed_late_tv": _between(worlds, _tv),
        "between_seed_late_js_distance": _between(worlds, _js_distance),
        "movement_rate": _mean_sd([world["movement_rate"] for world in worlds]),
        "mean_dwell": _mean_sd([world["mean_dwell"] for world in worlds]),
        "occupancy_entropy": _mean_sd([world["occupancy_entropy"] for world in worlds]),
        "saturation_frequency": saturation,
        "empty_frequency": _mean_sd([world["empty_frequency"] for world in worlds]),
        "alternation_rate": alternation,
        "consecutive_occupancy_tv": consecutive,
        "lag1": {place_id: _mean_sd([world["lag1"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "lag2": {place_id: _mean_sd([world["lag2"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "unstable": unstable,
        "technical_regime": classify_regime(float(destination["mean"]), unstable),
    }


def _dose_response(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for spec in ARCHITECTURES:
        architecture = spec["condition_id"]
        report[architecture] = {}
        for outcome in DOSE_OUTCOMES:
            values = [_outcome_mean(cells[f"{architecture}_w{weight:g}"], outcome) for weight in WEIGHTS]
            report[architecture][outcome] = {
                "weights": list(WEIGHTS),
                "means": values,
                "shape": dose_shape(values, DOSE_FLAT_TOLERANCE[outcome]),
            }
    return report


def _sensitivity(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for spec in ARCHITECTURES:
        architecture = spec["condition_id"]
        report[architecture] = {}
        for outcome in SENSITIVITY_OUTCOMES:
            means = [_outcome_mean(cells[f"{architecture}_w{weight:g}"], outcome) for weight in WEIGHTS]
            steps = []
            for left, right, start, stop in zip(means, means[1:], WEIGHTS, WEIGHTS[1:]):
                steps.append(
                    {
                        "from_weight": start,
                        "to_weight": stop,
                        "per_25": (right - left) / ((stop - start) / 25.0),
                    }
                )
            report[architecture][outcome] = {
                "steps": steps,
                "mean_per_25": statistics.fmean(step["per_25"] for step in steps),
            }
    return report


def _outcome_mean(cell: dict[str, Any], outcome: str) -> float:
    if outcome == "between_seed_late_tv":
        return float(cell["between_seed_late_tv"]["mean"])
    if outcome == "between_seed_late_js_distance":
        return float(cell["between_seed_late_js_distance"]["mean"])
    return float(cell[outcome]["mean"])


def _ratio_summary(values: list[float], undefined: int) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "median": None, "q1": None, "q3": None, "n_defined": 0, "n_undefined": undefined}
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "n_defined": len(values),
        "n_undefined": undefined,
    }


def _replay_all(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    worlds: list[dict[str, Any]],
    seeds: tuple[int, ...],
    timesteps: int,
) -> dict[str, Any]:
    replays = {}
    for item in REPLAYS:
        if item["choice_seed"] not in seeds:
            continue
        spec = next(spec for spec in ARCHITECTURES if spec["condition_id"] == item["condition_id"])
        with _weight(item["social_weight"]):
            result = simulate_memory(
                atlas,
                locations,
                _make_memory(spec, list(locations)),
                choice_seed_value=item["choice_seed"],
                timesteps=timesteps,
            )
        digest = trajectory_hash(result["series"])
        original = next(
            world["trajectory_sha256"]
            for world in worlds
            if world["architecture"] == item["condition_id"]
            and world["social_weight"] == item["social_weight"]
            and world["choice_seed"] == item["choice_seed"]
        )
        if digest != original:
            raise RuntimeError(f"{item['condition_id']} weight {item['social_weight']:g} replay hash did not match")
        replays[f"{item['condition_id']}_w{item['social_weight']:g}_seed{item['choice_seed']}"] = {
            "trajectory_sha256": digest,
            "match": True,
        }
    return replays


def _log(state_file: Any, memory_file: Any, action_file: Any, spec: dict[str, Any], weight: float, seed: int, signal: dict[str, Any]) -> Any:
    def write(logged: int, locations: dict[str, str], memory: Any, actions: list[dict[str, Any]]) -> None:
        for action in actions:
            social = [float(action["social_utility"][place_id]) for place_id in PLACE_IDS]
            base = [float(action["base_utility"][place_id]) for place_id in PLACE_IDS]
            signal["abs_sum"] += sum(abs(value) for value in social) / len(PLACE_IDS)
            signal["decisions"] += 1
            base_spread = max(base) - min(base)
            if base_spread == 0.0:
                signal["undefined"] += 1
            else:
                signal["ratios"].append((max(social) - min(social)) / base_spread)
        _dump_state(state_file, spec, weight, seed, logged, locations)
        _dump(
            memory_file,
            {
                "condition_id": spec["condition_id"],
                "social_weight": weight,
                "choice_seed": seed,
                "timestep": logged,
                "memory": spec["memory"],
                "window": spec["window"],
                "decay": spec["decay"],
                "normalization": CENTERED_NORMALIZATION,
                "raw_total": memory.total,
                "raw_own": memory.own,
                "llm_calls": 0,
            },
        )
        for action in actions:
            _dump(action_file, {"condition_id": spec["condition_id"], "choice_seed": seed, **action})

    return write


def _write_initial(state_file: Any, memory_file: Any, spec: dict[str, Any], weight: float, seed: int, locations: dict[str, str]) -> None:
    _dump_state(state_file, spec, weight, seed, 0, locations)
    zeros = {place_id: 0 for place_id in PLACE_IDS}
    _dump(
        memory_file,
        {
            "condition_id": spec["condition_id"],
            "social_weight": weight,
            "choice_seed": seed,
            "timestep": 0,
            "memory": spec["memory"],
            "window": spec["window"],
            "decay": spec["decay"],
            "normalization": CENTERED_NORMALIZATION,
            "raw_total": zeros,
            "raw_own": {agent_id: dict(zeros) for agent_id in sorted(locations)},
            "llm_calls": 0,
        },
    )


def _dump_state(handle: Any, spec: dict[str, Any], weight: float, seed: int, logged: int, locations: dict[str, str]) -> None:
    ordered = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    _dump(
        handle,
        {
            "condition_id": spec["condition_id"],
            "social_weight": weight,
            "choice_seed": seed,
            "timestep": logged,
            "locations": ordered,
            "occupancy": occupancy_of(ordered),
            "llm_calls": 0,
        },
    )


def _zero_social_baseline() -> dict[str, Any]:
    summary = json.loads((WORLD_V04_DIR / "experiment_004_summary.json").read_text(encoding="utf-8"))
    config = json.loads((WORLD_V04_DIR / "world_v0_4_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((WORLD_V04_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    earlier = json.loads((WORLD_V03_DIR / "experiment_003_summary.json").read_text(encoding="utf-8"))
    earlier_manifest = json.loads((WORLD_V03_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((CALIBRATION_DIR / "summary.json").read_text(encoding="utf-8"))
    if config.get("utility_atlas_sha256") != EXPECTED_ATLAS_SHA256:
        raise RuntimeError("experiment 004 atlas hash is not the frozen atlas")
    if config.get("familiarity_weight") != 0.0 or config.get("choice_temperature") != 200.0 or config.get("reconsideration_rate") != 0.5:
        raise RuntimeError("experiment 004 reference settings do not match")
    s0 = summary["conditions"]["S0"]
    r0 = earlier["conditions"]["R0"]
    if s0.get("social_weight") != 0.0:
        raise RuntimeError("S0 is not the zero-social reference")
    if s0["movement_rate"]["mean"] != r0["movement_rate"]["mean"] or s0["occupancy_entropy"]["mean"] != r0["occupancy_entropy"]["mean"]:
        raise RuntimeError("S0 and R0 reference dynamics diverge")
    if manifest["trajectory_sha256"]["S0:0"] != earlier_manifest["trajectory_sha256"]["R0:0"]:
        raise RuntimeError("S0 and R0 seed 0 trajectory hashes diverge")
    cell = next(
        item
        for item in calibration["cells"]
        if item["choice_temperature"] == 200.0 and item["reconsideration_rate"] == 0.5
    )
    if abs(s0["movement_rate"]["mean"] - cell["movement_rate"]["mean"]) > 0.01:
        raise RuntimeError("S0 movement is outside the reference-dynamics band")
    if abs(s0["occupancy_entropy"]["mean"] - cell["occupancy_entropy"]["mean"]) > 0.01:
        raise RuntimeError("S0 occupancy entropy is outside the reference-dynamics band")
    return {
        "source": "experiment_004 S0, with experiment_003 R0 seed-0 hash and dynamics cross-check",
        "rerun": False,
        "experiment_004_config_sha256": config["config_sha256"],
        "s0_seed0_trajectory_sha256": manifest["trajectory_sha256"]["S0:0"],
        "r0_seed0_trajectory_sha256": earlier_manifest["trajectory_sha256"]["R0:0"],
        "late_hhi": s0["late_hhi"]["mean"],
        "late_entropy": s0["late_entropy"]["mean"],
        "late_dominant_share": s0["late_dominant_share"]["mean"],
        "between_seed_late_tv": summary["between_seed"]["S0"]["tv_distance"]["mean"],
        "between_seed_late_js_distance": summary["between_seed"]["S0"]["js_distance"]["mean"],
        "movement_rate": s0["movement_rate"]["mean"],
        "occupancy_entropy": s0["occupancy_entropy"]["mean"],
        "calibration_movement_rate": cell["movement_rate"]["mean"],
        "calibration_occupancy_entropy": cell["occupancy_entropy"]["mean"],
    }


def _load_weight_150_locations() -> dict[str, dict[int, list[dict[str, str]]]]:
    grouped: dict[str, dict[int, list[tuple[int, dict[str, str]]]]] = {
        spec["condition_id"]: {seed: [] for seed in CHOICE_SEEDS} for spec in ARCHITECTURES
    }
    path = WORLD_V05_DIR / "world_states.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"condition_id":"CUM"' in line:
                continue
            row = json.loads(line)
            condition_id = row.get("condition_id")
            if condition_id not in grouped:
                continue
            seed = int(row["choice_seed"])
            if seed not in grouped[condition_id]:
                continue
            grouped[condition_id][seed].append(
                (int(row["timestep"]), {str(key): str(value) for key, value in row["locations"].items()})
            )
    loaded: dict[str, dict[int, list[dict[str, str]]]] = {}
    for condition_id, seeds in grouped.items():
        loaded[condition_id] = {}
        for seed, rows in seeds.items():
            rows.sort(key=lambda item: item[0])
            series = [locations for _timestep, locations in rows[:501]]
            if len(series) != 501:
                raise RuntimeError(f"Experiment 005 {condition_id} seed {seed} does not have 501 states")
            loaded[condition_id][seed] = series
    return loaded


def _eligible(cell: dict[str, Any], baseline_tv: float) -> bool:
    return (
        cell["unstable"] is False
        and 0.15 <= float(cell["destination_change_rate"]) <= 0.35
        and float(cell["ranking_change_rate"]) > 0.25
        and float(cell["clipping_rate"]) < 0.10
        and len(cell["final_dominant_places"]) >= 3
        and float(cell["between_seed_late_tv"]) > baseline_tv
    )


def _selection_key(cell: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        abs(float(cell["destination_change_rate"]) - 0.25),
        float(cell["clipping_rate"]),
        -float(cell["between_seed_late_tv"]),
        float(cell["social_weight"]),
        str(cell["architecture"]),
    )


def _config(seeds: tuple[int, ...], timesteps: int, baseline: dict[str, Any]) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_6",
        "experiment_id": "experiment_006_social_weight_memory",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "prompt_version": "independent_utility_v3",
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "normalization": CENTERED_NORMALIZATION,
        "architectures": [dict(spec) for spec in ARCHITECTURES],
        "social_weights": list(WEIGHTS),
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "cum_excluded_from_factorial": True,
        "regime_rule": REGIME_RULE,
        "selection_rule": SELECTION_RULE,
        "dose_rule": DOSE_RULE,
        "dose_flat_tolerance": DOSE_FLAT_TOLERANCE,
        "unstable_thresholds": {
            "alternation_upper_quartile": ALTERNATION_UPPER_QUARTILE,
            "consecutive_tv_upper_quartile": CONSECUTIVE_TV_UPPER_QUARTILE,
            "extreme_late_hhi": EXTREME_LATE_HHI,
            "extreme_dominant_share": EXTREME_DOMINANT_SHARE,
            "extreme_saturation": EXTREME_SATURATION,
        },
        "zero_social_baseline_s0_seed0_sha256": baseline["s0_seed0_trajectory_sha256"],
        "llm_calls": 0,
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _manifest(destination: Path, config: dict[str, Any], summary: dict[str, Any], worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_006_social_weight_memory",
        "llm_calls": 0,
        "world_count": len(worlds),
        "cell_count": summary["cell_count"],
        "complete": len(worlds) == len(ARCHITECTURES) * len(WEIGHTS) * len(config["choice_seeds"]),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "weight_150_matches_experiment_005": True,
        "exact_replay": summary["exact_replay"],
        "technical_regime": {
            cell_id: summary["cells"][cell_id]["technical_regime"] for cell_id in summary["cells"]
        },
        "reference_regime": summary["reference_regime"]["selection"],
        "trajectory_sha256": {
            f"{world['architecture']}:{world['social_weight']:g}:{world['choice_seed']}": world["trajectory_sha256"]
            for world in worlds
        },
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_6_config.json",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "experiment_006_summary.json",
                "factorial_analysis.json",
            )
        },
    }


def _term_label(name: str) -> str:
    if name.startswith("C(architecture") and ":" not in name:
        return "architecture"
    if name.startswith("C(social_weight)") and ":" not in name:
        return "social_weight"
    if name == "social_weight":
        return "social_weight"
    if ":" in name:
        return "architecture_x_social_weight"
    return name


def _finite(value: Any) -> float | None:
    number = float(value)
    if math.isfinite(number):
        return number
    return None


def _require_atlas_hash() -> None:
    manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != EXPECTED_ATLAS_SHA256:
        raise RuntimeError("utility atlas hash is not the frozen experiment atlas")


def _refuse_frozen_directories(destination: Path) -> None:
    forbidden = {
        WORLD_V01_DIR.resolve(),
        WORLD_V02_DIR.resolve(),
        WORLD_V03_DIR.resolve(),
        WORLD_V04_DIR.resolve(),
        WORLD_V05_DIR.resolve(),
        ATLAS_DIR.resolve(),
        CALIBRATION_DIR.resolve(),
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 006 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def write_analysis_from_logs(output_dir: Path | None = None) -> None:
    destination = output_dir or OUTPUT_DIR
    summary = json.loads((destination / "experiment_006_summary.json").read_text(encoding="utf-8"))
    config = json.loads((destination / "world_v0_6_config.json").read_text(encoding="utf-8"))
    worlds = _records_from_logs(destination)
    _require_records_match_summary(worlds, summary)
    factorial = fit_factorial(worlds)
    write_json(destination / "factorial_analysis.json", factorial)
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, worlds))


def _records_from_logs(destination: Path) -> list[dict[str, Any]]:
    signal: dict[tuple[str, float, int], dict[str, float]] = {}
    with (destination / "agent_actions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["condition_id"]), float(row["social_weight"]), int(row["choice_seed"]))
            slot = signal.setdefault(key, {"decisions": 0.0, "rank": 0.0, "dest": 0.0, "reconsidered": 0.0})
            slot["decisions"] += 1.0
            slot["rank"] += 1.0 if row["ranking_changed"] else 0.0
            if row["reconsidered"]:
                slot["reconsidered"] += 1.0
                slot["dest"] += 1.0 if row["destination_changed"] else 0.0
    grouped: dict[tuple[str, float, int], list[tuple[int, dict[str, str]]]] = {}
    with (destination / "world_states.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["condition_id"]), float(row["social_weight"]), int(row["choice_seed"]))
            grouped.setdefault(key, []).append(
                (int(row["timestep"]), {str(agent_id): str(place_id) for agent_id, place_id in row["locations"].items()})
            )
    worlds = []
    for key, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        series = [locations for _timestep, locations in rows]
        counts = signal[key]
        early_counts = _presence(series, 1, 20)
        late_counts = _presence(series, 101, 500)
        early_share = _share(early_counts)
        late_share = _share(late_counts)
        early_leader = _unique_leader(early_counts)
        late_leader = _unique_leader(late_counts)
        architecture, weight, seed = key
        worlds.append(
            {
                "architecture": architecture,
                "social_weight": weight,
                "choice_seed": seed,
                "trajectory_sha256": trajectory_hash(series),
                "ranking_change_rate": counts["rank"] / counts["decisions"],
                "destination_change_rate": counts["dest"] / counts["reconsidered"],
                "late_hhi": _hhi(late_share),
                "late_entropy": _entropy(late_share),
                "late_dominant_share": max(late_share.values()),
                "early_leader_retention": None if early_leader is None else float(late_leader == early_leader),
                "early_late_tv": _tv(early_share, late_share),
                "early_late_js_distance": _js_distance(early_share, late_share),
            }
        )
    return worlds


def _require_records_match_summary(worlds: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if len(worlds) != summary["world_count"]:
        raise RuntimeError("logged worlds do not match the experiment summary")
    for cell in summary["cells"].values():
        rows = [
            world
            for world in worlds
            if world["architecture"] == cell["architecture"] and world["social_weight"] == cell["social_weight"]
        ]
        if len(rows) != cell["n_seeds"]:
            raise RuntimeError(f"{cell['cell_id']} log count does not match the summary")
        for outcome in ("ranking_change_rate", "destination_change_rate", "late_hhi", "early_leader_retention"):
            observed = _mean_sd(
                [world[outcome] if outcome != "early_leader_retention" else world[outcome] for world in rows]
            )
            expected = cell[outcome]["mean"]
            if abs(float(observed["mean"]) - float(expected)) > 1e-9:
                raise RuntimeError(f"{cell['cell_id']} {outcome} rebuilt mean does not match the summary")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_006")
    main()
