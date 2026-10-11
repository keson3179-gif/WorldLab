"""Experiment 014: long-horizon persistence of Experiment 013 history residuals.

Worlds are rerun from t=0 through t=2000. The first 500 steps must match
Experiment 013. Lambda is not refined again. This module does not call a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

import world.experiment_005 as memory_experiment
import world.experiment_013 as experiment_013
from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.dynamics_calibration import OUTPUT_DIR as CALIBRATION_DIR
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_003 import OUTPUT_DIR as WORLD_V03_DIR
from world.experiment_004 import OUTPUT_DIR as WORLD_V04_DIR
from world.experiment_004 import _hhi, _js_distance, _tv, trajectory_hash
from world.experiment_005 import CENTERED_NORMALIZATION, FAMILIARITY_WEIGHT, OUTPUT_DIR as WORLD_V05_DIR, is_unstable
from world.experiment_006 import (
    ALTERNATION_UPPER_QUARTILE,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    OUTPUT_DIR as WORLD_V06_DIR,
)
from world.experiment_007 import OUTPUT_DIR as WORLD_V07_DIR
from world.experiment_008 import OUTPUT_DIR as WORLD_V08_DIR
from world.experiment_009 import OUTPUT_DIR as WORLD_V09_DIR, classify_persistence, memory_payload
from world.experiment_010 import OUTPUT_DIR as WORLD_V10_DIR
from world.experiment_011 import OUTPUT_DIR as WORLD_V11_DIR
from world.experiment_012 import (
    OUTPUT_DIR as WORLD_V12_DIR,
    _anchor_mad,
    _effect,
    _probability_tv,
    _share_of,
    _spearman,
    _trace_mad,
    midpoint_scores,
    reconvergence_time,
    stable_attenuation_time,
)
from world.experiment_013 import (
    BLEND_TOLERANCE,
    BOOTSTRAP_DRAWS,
    HELDOUT_SEEDS,
    HISTORIES,
    OUTPUT_DIR as WORLD_V13_DIR,
    REGIMES,
    simulate_history,
)
from world.geography import PLACE_IDS, occupancy_of
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR, FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_14"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
TIMESTEPS = 2000
LINEAGE_STEPS = 500
INTERVENTION_STEP = 21
ENVIRONMENT_STEP = 22
OVERRIDE_COUNT = 2
ACCUMULATOR_LIMIT = 1_000_000.0
JACCARD_STABLE_MIN = 0.5
SOCIAL_LAMBDAS = (0.3, 0.35, 0.4, 1.0)
S0_LAMBDAS = (0.0, 1.0)
HORIZONS = (("H500", 401, 500), ("H1000", 901, 1000), ("H1500", 1401, 1500), ("H2000", 1901, 2000))
SAFETY_LATE = (1901, 2000)
SAFETY_LINEAGE = (401, 500)
CLASS_CODE = {
    "STABLE_ATTENUATION": "LH-A",
    "DURABLE_PERSISTENCE": "LH-B",
    "TRANSIENT_PERSISTENCE": "LH-C",
    "DELAYED_PERSISTENCE": "LH-D",
    "INTERMITTENT": "LH-E",
}
ROBUST_SENTENCE = (
    "the feedback-autonomy persistence onset classification is robust to extending the simulation horizon "
    "from 500 to 2000 timesteps under the tested architecture"
)
TRANSIENT_SENTENCE = "the apparent 500-step persistent residual is horizon-sensitive and attenuates over the longer 2000-step simulation"
DELAYED_SENTENCE = (
    "history divergence can emerge or re-establish over longer horizons despite attenuation at the original 500-step horizon"
)
INTERMITTENT_SENTENCE = "history persistence is intermittent across the tested horizons"
TURNOVER_SENTENCE = "ensemble-level persistence does not imply persistence of the same stochastic histories across horizons"
SUBSET_SENTENCE = (
    "long-horizon divergence is concentrated in a reproducible subset of stochastic histories within the tested seed cohort"
)
SCOPE_SENTENCE = (
    "A persistent residual through 2000 timesteps is a finite-horizon result. "
    "It is not permanent path dependence, infinite-horizon persistence, an absorbing state, a phase transition, "
    "a critical point, a tipping point, or collective path dependence."
)
HORIZON_RULE = (
    "Primary windows are H500 t=401-500, H1000 t=901-1000, H1500 t=1401-1500, and H2000 t=1901-2000. "
    "Each regime-mean D uses the same 0.025 and 0.10 thresholds. "
    "LH-A: all four attenuated. LH-B: all four persistent. "
    "LH-C: H500 persistent and H2000 attenuated. LH-D: H500 attenuated and H2000 persistent. "
    "Any other mixture is LH-E. LH-C is transient long-horizon persistence, not a proof of dynamical metastability."
)


def _lambda_label(value: float) -> str:
    return f"{value:.2f}"


def lambdas_for(regime_id: str) -> tuple[float, ...]:
    return S0_LAMBDAS if regime_id == "S0" else SOCIAL_LAMBDAS


def long_horizon_class(levels: list[float]) -> str:
    if len(levels) != 4:
        raise RuntimeError("long-horizon classification requires the four fixed windows")
    persistent = [value >= 0.025 for value in levels]
    if not any(persistent):
        return "STABLE_ATTENUATION"
    if all(persistent):
        return "DURABLE_PERSISTENCE"
    if persistent[0] and not persistent[-1]:
        return "TRANSIENT_PERSISTENCE"
    if not persistent[0] and persistent[-1]:
        return "DELAYED_PERSISTENCE"
    return "INTERMITTENT"


def horizon_onset(levels: dict[float, float]) -> dict[str, Any]:
    grid = list(SOCIAL_LAMBDAS)
    persists = [levels[value] >= 0.025 for value in grid]
    seen = False
    for flag in persists:
        if seen and not flag:
            return {"status": "NONMONOTONIC", "horizon_onset_lambda": None, "horizon_onset_band": None}
        if flag:
            seen = True
    if not seen:
        return {"status": "NONE", "horizon_onset_lambda": None, "horizon_onset_band": None}
    index = persists.index(True)
    onset = grid[index]
    if index == 0:
        band = {"lower_exclusive": None, "upper_inclusive": onset, "label": "(-, 0.30]"}
    else:
        previous = grid[index - 1]
        band = {
            "lower_exclusive": previous,
            "upper_inclusive": onset,
            "label": f"({_lambda_label(previous)}, {_lambda_label(onset)}]",
        }
    return {"status": "ONSET", "horizon_onset_lambda": onset, "horizon_onset_band": band}


def _onset_rank(item: dict[str, Any]) -> float | None:
    if item["status"] == "NONMONOTONIC":
        return None
    if item["status"] == "NONE":
        return 2.0
    return float(item["horizon_onset_lambda"])


def onset_stability(items: list[dict[str, Any]]) -> str:
    ranks = [_onset_rank(item) for item in items]
    if any(rank is None for rank in ranks):
        return "ONSET_HORIZON_NONMONOTONIC"
    if len(set(ranks)) == 1:
        return "ONSET_STABLE"
    assert all(rank is not None for rank in ranks)
    ordered = [float(rank) for rank in ranks]
    diffs = [right - left for left, right in zip(ordered, ordered[1:], strict=False)]
    if all(diff >= -1e-12 for diff in diffs):
        return "ONSET_SHIFTS_UP"
    if all(diff <= 1e-12 for diff in diffs):
        return "ONSET_SHIFTS_DOWN"
    return "ONSET_HORIZON_NONMONOTONIC"


def jaccard(left: set[int], right: set[int]) -> float | None:
    if not left and not right:
        return None
    return len(left & right) / len(left | right)


def transition_counts(h500: list[bool], h2000: list[bool]) -> dict[str, int]:
    counts = {"AA": 0, "AP": 0, "PA": 0, "PP": 0}
    for early, late in zip(h500, h2000, strict=True):
        counts[("P" if early else "A") + ("P" if late else "A")] += 1
    return counts


def divergence_duty_cycle(values: list[float]) -> float:
    window = values[501:2001]
    if len(window) != 1500:
        raise RuntimeError("duty cycle requires timesteps 501-2000")
    return statistics.fmean(float(value >= 0.025) for value in window)


def divergence_episodes(values: list[float]) -> dict[str, float | int | None]:
    lengths: list[int] = []
    run = 0
    last = None
    for timestep, value in enumerate(values):
        if value >= 0.025:
            run += 1
            last = timestep
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return {
        "episode_count": len(lengths),
        "mean_episode_length": statistics.fmean(lengths) if lengths else 0.0,
        "max_episode_length": max(lengths) if lengths else 0,
        "last_divergent_t": last,
    }


def terminal_attenuation_time(values: list[float]) -> int | None:
    last_high = None
    for timestep, value in enumerate(values):
        if value >= 0.025:
            last_high = timestep
    if last_high is None:
        return 0
    if last_high == len(values) - 1:
        return None
    return last_high + 1


def terminal_exact_reconvergence_time(values: list[float]) -> int | None:
    last_nonzero = None
    for timestep, value in enumerate(values):
        if value != 0.0:
            last_nonzero = timestep
    if last_nonzero is None:
        return 0
    if last_nonzero == len(values) - 1:
        return None
    return last_nonzero + 1


def descriptive_bins(values: list[float]) -> list[dict[str, float | int]]:
    rows = []
    for start in range(1, TIMESTEPS + 1, 100):
        stop = start + 99
        rows.append({"start": start, "stop": stop, "mean_occupancy_tv": statistics.fmean(values[start : stop + 1])})
    return rows


def kaplan_meier(times: list[int | None], horizon: int = TIMESTEPS) -> list[dict[str, float | int]]:
    events: dict[int, int] = {}
    censored: dict[int, int] = {}
    for time in times:
        if time is None:
            censored[horizon] = censored.get(horizon, 0) + 1
        else:
            events[time] = events.get(time, 0) + 1
    survival = 1.0
    at_risk = len(times)
    points = []
    for time in sorted(set(events) | set(censored)):
        deaths = events.get(time, 0)
        dropped = censored.get(time, 0)
        if deaths and at_risk:
            survival *= 1.0 - deaths / at_risk
            points.append({"t": time, "survival": survival, "events": deaths, "at_risk": at_risk})
        at_risk -= deaths + dropped
    return points


def overall_conclusion(regimes: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    social = [regimes[name] for name in ("RW50", "EXP99")]
    if all(block["onset_stability"] == "ONSET_STABLE" for block in social):
        parts.append(ROBUST_SENTENCE)
    else:
        described = ", ".join(f"{name} is {regimes[name]['onset_stability']}" for name in ("RW50", "EXP99"))
        parts.append("horizon-specific onset is not stable across both social-memory architectures: " + described)
    seen: set[str] = set()
    for name in ("RW50", "EXP99"):
        for level in regimes[name]["levels"].values():
            label = level["long_horizon_class"]
            if label == "TRANSIENT_PERSISTENCE" and label not in seen:
                parts.append(TRANSIENT_SENTENCE)
                seen.add(label)
            elif label == "DELAYED_PERSISTENCE" and label not in seen:
                parts.append(DELAYED_SENTENCE)
                seen.add(label)
            elif label == "INTERMITTENT" and label not in seen:
                parts.append(INTERMITTENT_SENTENCE)
                seen.add(label)
    if any(level.get("turnover_interpretation") for name in ("RW50", "EXP99") for level in regimes[name]["levels"].values()):
        parts.append(TURNOVER_SENTENCE)
    if any(level.get("stable_subset_interpretation") for name in ("RW50", "EXP99") for level in regimes[name]["levels"].values()):
        parts.append(SUBSET_SENTENCE)
    parts.append(SCOPE_SENTENCE)
    return " ".join(parts)


def load_intervention_registry() -> dict[tuple[str, int], dict[str, Any]]:
    plans: dict[tuple[str, int], dict[str, Any]] = {}
    with (WORLD_V13_DIR / "heldout_interventions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            plans[(row["regime"], int(row["choice_seed"]))] = row
    expected = {(spec["regime_id"], seed) for spec in REGIMES for seed in HELDOUT_SEEDS}
    if set(plans) != expected:
        raise RuntimeError("Experiment 013 held-out intervention registry does not cover seeds 50-99")
    for plan in plans.values():
        if int(plan["override_count"]) != OVERRIDE_COUNT or int(plan["intervention_timestep"]) != INTERVENTION_STEP:
            raise RuntimeError("held-out intervention does not keep the Experiment 007 semantics")
        if int(plan["override_consumes_rng"]) != 0:
            raise RuntimeError("held-out intervention consumed RNG")
    return plans


def load_lineage_references() -> dict[str, Any]:
    manifest = json.loads((WORLD_V13_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((WORLD_V13_DIR / "experiment_013_summary.json").read_text(encoding="utf-8"))
    expected = {}
    for regime_id in ("RW50", "EXP99"):
        expected[regime_id] = {
            _lambda_label(value): summary["regimes"][regime_id]["levels"][_lambda_label(value)]["D_lambda"]
            for value in SOCIAL_LAMBDAS
        }
    return {"trajectory_sha256": manifest["trajectory_sha256"], "h500": expected}


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
    if timesteps != TIMESTEPS or seeds != HELDOUT_SEEDS:
        raise RuntimeError("the preregistered run is seeds 50-99 and 2000 timesteps")
    free = shutil.disk_usage(destination if destination.exists() else destination.parent).free
    if free < 40_000_000_000:
        raise RuntimeError("not enough free disk for experiment 014")
    destination.mkdir(parents=True, exist_ok=True)
    source = load_frozen_atlas() if atlas is None else atlas
    _require_atlas_hash()
    references = load_lineage_references()
    plans = load_intervention_registry()
    initial = load_initial_locations()
    config = _config(seeds, timesteps, references)
    write_json(destination / "world_v0_14_config.json", config)
    print("smoke test to t=600", flush=True)
    _smoke(source, initial, plans, references)
    names = (
        "world_states.jsonl",
        "social_memory_states.jsonl",
        "agent_actions.jsonl",
        "long_horizon_donor_social_fields.jsonl",
        "long_horizon_midpoint_fields.jsonl",
        "feedback_blend_audit.jsonl",
        "long_horizon_pair_metrics.jsonl",
    )
    paths = {name: destination / name for name in names}
    for path in paths.values():
        path.write_text("", encoding="utf-8", newline="\n")
    metrics = []
    branch_worlds = []
    with paths["world_states.jsonl"].open("a", encoding="utf-8", newline="\n") as states, paths["social_memory_states.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as memories, paths["agent_actions.jsonl"].open("a", encoding="utf-8", newline="\n") as actions, paths[
        "long_horizon_donor_social_fields.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as donors, paths["long_horizon_midpoint_fields.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as midpoints, paths["feedback_blend_audit.jsonl"].open("a", encoding="utf-8", newline="\n") as audits, paths[
        "long_horizon_pair_metrics.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as pairs:
        for spec in REGIMES:
            for seed in seeds:
                record, worlds = _run_seed(
                    source, initial, spec, seed, plans[(spec["regime_id"], seed)], references, states, memories, actions, donors, midpoints, audits, pairs
                )
                metrics.append(record)
                branch_worlds.extend(worlds)
                print(
                    f"{spec['regime_id']} seed={seed} "
                    + " ".join(
                        f"{item['lambda_label']}:H500={item['horizons']['H500']['D_horizon']:.4f}"
                        for item in record["lambdas"]
                    ),
                    flush=True,
                )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    _require_h500(metrics, references)
    replays = _replay(source, initial, plans, metrics)
    summary = _summarize(metrics, branch_worlds, replays, references)
    write_json(destination / "experiment_014_summary.json", summary)
    write_json(destination / "horizon_persistence_analysis.json", _horizon_analysis(summary))
    write_json(destination / "seed_transition_analysis.json", _transition_analysis(summary))
    write_json(destination / "divergence_episode_analysis.json", _episode_analysis(summary))
    write_json(destination / "terminal_attenuation_analysis.json", _terminal_analysis(summary))
    write_json(destination / "regime_comparison.json", _compare(summary))
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, metrics))
    return destination


def _smoke(atlas, initial, plans, references) -> None:
    for regime_id, lambdas in (("RW50", (0.4,)), ("S0", S0_LAMBDAS)):
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        plan = plans[(regime_id, 50)]
        donors, _numeric = _collect_donors(atlas, initial, spec, 50, plan, timesteps=600, references=references)
        by_history = {history: [] for history in HISTORIES}
        for lambda_value in lambdas:
            for history in HISTORIES:
                result = simulate_history(
                    atlas,
                    initial,
                    spec,
                    choice_seed_value=50,
                    history=history,
                    lambda_value=lambda_value,
                    plan=plan,
                    donors=donors,
                    timesteps=600,
                )
                _require_prefix(spec, 50, history, lambda_value, result, references)
                by_history[history].append(result["canonical_dynamics_sha256"])
                if result["max_blend_error"] > BLEND_TOLERANCE:
                    raise RuntimeError("smoke blend audit failed")
        if regime_id == "S0" and any(len(set(values)) != 1 for values in by_history.values()):
            raise RuntimeError("S0 smoke histories are not lambda-equivalent")


def _run_seed(atlas, initial, spec, seed, plan, references, states, memories, actions, donor_handle, midpoint_handle, audits, pairs):
    lambdas = lambdas_for(spec["regime_id"])
    donors, numeric = _collect_donors(atlas, initial, spec, seed, plan, timesteps=TIMESTEPS, states=states, memories=memories, actions=actions, references=references)
    _write_donors(donor_handle, spec, seed, donors)
    _write_midpoint(midpoint_handle, spec, seed, donors)
    runs: dict[tuple[float, str], dict[str, Any]] = {}
    worlds = []
    for history in HISTORIES:
        result = donors.pop(("result", history))
        _write_lambda_audit(audits, spec, seed, history, 1.0, result, donors)
        runs[(1.0, history)] = result
        worlds.append(_branch_safety(spec, seed, history, 1.0, result))
    for lambda_value in lambdas:
        if lambda_value == 1.0:
            continue
        for history in HISTORIES:
            result = _run_logged(
                atlas, initial, spec, seed, history, lambda_value, plan, donors, references, states, memories, actions, audits
            )
            runs[(lambda_value, history)] = result
            worlds.append(_branch_safety(spec, seed, history, lambda_value, result))
    _require_seed_gates(spec, seed, runs, lambdas)
    record = _seed_metrics(spec, seed, runs, numeric, pairs)
    for result in runs.values():
        for key in ("probabilities", "series", "used_trace", "endo_trace", "mid_trace", "endogenous_field", "memory_hashes"):
            result.pop(key, None)
    return record, worlds


def _collect_donors(atlas, initial, spec, seed, plan, timesteps, states=None, memories=None, actions=None, references=None):
    donors: dict[Any, Any] = {}
    numeric = _empty_numeric()
    for history in HISTORIES:
        memory_hashes: dict[int, str] = {}

        def write(logged, current, memory, step_actions, _audit, history=history, memory_hashes=memory_hashes):
            if states is not None and memories is not None and actions is not None:
                _write_step(states, memories, actions, spec, seed, history, 1.0, logged, current, memory, step_actions)
            _observe_memory(spec, memory, numeric)
            if logged >= ENVIRONMENT_STEP:
                memory_hashes[logged] = canonical_sha256(memory_payload(memory))

        result = simulate_history(
            atlas,
            initial,
            spec,
            choice_seed_value=seed,
            history=history,
            lambda_value=1.0,
            plan=plan,
            donors=None,
            timesteps=timesteps,
            require_donors=False,
            on_step=write if states is not None else None,
        )
        if references is not None and timesteps >= LINEAGE_STEPS:
            _require_prefix(spec, seed, history, 1.0, result, references)
        if result["used_trace"] != result["endo_trace"]:
            raise RuntimeError("lambda 1 used field is not the endogenous field")
        _observe_scores(result["endogenous_field"], numeric)
        result["memory_hashes"] = memory_hashes
        donors[history] = result["endogenous_field"]
        donors[("result", history)] = result
    if numeric["nonfinite"] or numeric["exploded"]:
        raise RuntimeError(f"{spec['regime_id']} seed {seed} failed the numerical stability audit")
    return donors, numeric


def _run_logged(atlas, initial, spec, seed, history, lambda_value, plan, donors, references, states, memories, actions, audits):
    def write(logged, current, memory, step_actions, audit):
        _write_step(states, memories, actions, spec, seed, history, lambda_value, logged, current, memory, step_actions)
        if audit is not None:
            _dump(audits, _compact_audit(spec, seed, history, lambda_value, logged, audit))

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
        on_step=write,
    )
    _require_prefix(spec, seed, history, lambda_value, result, references)
    if result["llm_calls"] != 0:
        raise RuntimeError("experiment 014 called a model")
    return result


def _require_prefix(spec, seed, history, lambda_value, result, references) -> None:
    prefix = trajectory_hash(result["series"][: LINEAGE_STEPS + 1])
    key = f"{spec['regime_id']}:{seed}:{history}:{_lambda_label(lambda_value)}"
    if prefix != references["trajectory_sha256"][key]:
        raise RuntimeError(f"{key} first-500 trajectory does not match Experiment 013")
    result["prefix_sha256"] = prefix


def _require_seed_gates(spec, seed, runs, lambdas) -> None:
    base = runs[(1.0, "CONTROL")]
    for (lambda_value, history), result in runs.items():
        if result["series"][:INTERVENTION_STEP] != base["series"][:INTERVENTION_STEP]:
            raise RuntimeError(f"{spec['regime_id']} seed {seed} diverged before t=21")
        if result["rng_draw_counts"] != base["rng_draw_counts"]:
            raise RuntimeError("lambda or history changed the RNG schedule")
        if result["max_blend_error"] > BLEND_TOLERANCE:
            raise RuntimeError("blend audit tolerance failed")
        if result["series"][: ENVIRONMENT_STEP] != runs[(1.0, history)]["series"][:ENVIRONMENT_STEP]:
            raise RuntimeError("blending changed the pre-t22 history")
        if history == "PERTURBED" and result["override_count"] != OVERRIDE_COUNT:
            raise RuntimeError("perturbed history does not have exactly two overrides")
        if history == "CONTROL" and result["override_count"] != 0:
            raise RuntimeError("control history has an override")
        if lambda_value == 1.0 and result["used_trace"] != result["endo_trace"]:
            raise RuntimeError("lambda 1 used field is not endogenous")
    if spec["regime_id"] == "S0":
        if runs[(0.0, "CONTROL")]["used_trace"] != runs[(0.0, "PERTURBED")]["used_trace"]:
            raise RuntimeError("S0 lambda 0 histories do not share the decision field")
        for history in HISTORIES:
            if runs[(0.0, history)]["canonical_dynamics_sha256"] != runs[(1.0, history)]["canonical_dynamics_sha256"]:
                raise RuntimeError("S0 lambda levels are not dynamically equivalent through t=2000")
    if set(lambdas) != {key[0] for key in runs}:
        raise RuntimeError("a preregistered lambda was not run")


def _seed_metrics(spec, seed, runs, numeric, pairs) -> dict[str, Any]:
    levels = []
    for lambda_value in lambdas_for(spec["regime_id"]):
        contrast = _series_contrast(runs[(lambda_value, "PERTURBED")], runs[(lambda_value, "CONTROL")])
        if len(contrast["occupancy_tv"]) != TIMESTEPS + 1:
            raise RuntimeError("metric window requires the 2000-step run")
        horizons = {}
        for name, start, stop in HORIZONS:
            horizons[name] = {
                "D_horizon": statistics.fmean(contrast["occupancy_tv"][start : stop + 1]),
                "late_js": statistics.fmean(contrast["js_distance"][start : stop + 1]),
                "late_agent_disagreement": statistics.fmean(contrast["agent_disagreement"][start : stop + 1]),
                "late_probability_tv": statistics.fmean(contrast["probability_tv"][start : stop + 1]),
                "late_field_mad": statistics.fmean(contrast["field_mad"][start : stop + 1]),
                "late_endogenous_field_mad": statistics.fmean(contrast["endogenous_field_mad"][start : stop + 1]),
                "late_midpoint_mad": statistics.fmean(contrast["midpoint_mad"][start : stop + 1]),
            }
            horizons[name]["persistence_class"] = classify_persistence(horizons[name]["D_horizon"])
            horizons[name]["seed_persistent"] = horizons[name]["D_horizon"] >= 0.025
        attenuation = terminal_attenuation_time(contrast["occupancy_tv"])
        reconverged = terminal_exact_reconvergence_time(contrast["occupancy_tv"])
        for timestep, value in enumerate(contrast["occupancy_tv"]):
            _dump(
                pairs,
                {
                    "regime": spec["regime_id"],
                    "choice_seed": seed,
                    "lambda": lambda_value,
                    "timestep": timestep,
                    "occupancy_tv": value,
                    "js_distance": contrast["js_distance"][timestep],
                    "agent_disagreement": contrast["agent_disagreement"][timestep],
                    "probability_tv": contrast["probability_tv"][timestep],
                    "field_mad": contrast["field_mad"][timestep],
                    "endogenous_field_mad": contrast["endogenous_field_mad"][timestep],
                    "midpoint_mad": contrast["midpoint_mad"][timestep],
                },
            )
        levels.append(
            {
                "lambda": lambda_value,
                "lambda_label": _lambda_label(lambda_value),
                "horizons": horizons,
                "descriptive_bins": descriptive_bins(contrast["occupancy_tv"]),
                "divergence_duty_cycle": divergence_duty_cycle(contrast["occupancy_tv"]),
                "episodes": divergence_episodes(contrast["occupancy_tv"]),
                "terminal_attenuation_time": attenuation,
                "terminal_attenuation_right_censored": attenuation is None,
                "terminal_exact_reconvergence_time": reconverged,
                "terminal_exact_reconvergence_right_censored": reconverged is None,
                "stable_attenuation_time_20": stable_attenuation_time(contrast["occupancy_tv"]),
                "time_to_reconvergence_20": reconvergence_time(contrast["occupancy_tv"]),
                "control_trajectory_sha256": runs[(lambda_value, "CONTROL")]["trajectory_sha256"],
                "perturbed_trajectory_sha256": runs[(lambda_value, "PERTURBED")]["trajectory_sha256"],
                "prefix_sha256": {
                    "CONTROL": runs[(lambda_value, "CONTROL")]["prefix_sha256"],
                    "PERTURBED": runs[(lambda_value, "PERTURBED")]["prefix_sha256"],
                },
            }
        )
    return {
        "regime_id": spec["regime_id"],
        "choice_seed": seed,
        "lambdas": levels,
        "numerical": numeric if spec["regime_id"] == "EXP99" else None,
        "llm_calls": 0,
    }


def _series_contrast(perturbed, control) -> dict[str, list[float]]:
    occupancy_tv, js_distance, disagreement = [], [], []
    agents = sorted(perturbed["series"][0])
    population = len(agents)
    for timestep in range(len(perturbed["series"])):
        occupancy_tv.append(_tv(_share_of(perturbed["series"][timestep]), _share_of(control["series"][timestep])))
        js_distance.append(_js_distance(_share_of(perturbed["series"][timestep]), _share_of(control["series"][timestep])))
        disagreement.append(
            sum(perturbed["series"][timestep][agent_id] != control["series"][timestep][agent_id] for agent_id in agents) / population
        )
    return {
        "occupancy_tv": occupancy_tv,
        "js_distance": js_distance,
        "agent_disagreement": disagreement,
        "probability_tv": _probability_tv(perturbed["probabilities"], control["probabilities"]),
        "field_mad": [_trace_mad(left, right) for left, right in zip(perturbed["used_trace"], control["used_trace"], strict=True)],
        "endogenous_field_mad": [_trace_mad(left, right) for left, right in zip(perturbed["endo_trace"], control["endo_trace"], strict=True)],
        "midpoint_mad": [
            statistics.fmean(
                (
                    _anchor_mad(perturbed["used_trace"][timestep], perturbed["mid_trace"][timestep]),
                    _anchor_mad(control["used_trace"][timestep], control["mid_trace"][timestep]),
                )
            )
            for timestep in range(len(perturbed["series"]))
        ],
    }


def _require_h500(metrics, references) -> None:
    for regime_id in ("RW50", "EXP99"):
        rows = [row for row in metrics if row["regime_id"] == regime_id]
        for lambda_value in SOCIAL_LAMBDAS:
            label = _lambda_label(lambda_value)
            actual = statistics.fmean(_lambda_row(row, lambda_value)["horizons"]["H500"]["D_horizon"] for row in rows)
            expected = references["h500"][regime_id][label]
            if actual != expected:
                raise RuntimeError(f"{regime_id} {label} H500 {actual} does not reproduce Experiment 013 {expected}")


def _summarize(metrics, worlds, replays, references) -> dict[str, Any]:
    regimes = {spec["regime_id"]: _regime_summary(spec, [row for row in metrics if row["regime_id"] == spec["regime_id"]], worlds) for spec in REGIMES}
    for regime_id in ("RW50", "EXP99"):
        block = regimes[regime_id]
        onsets = [horizon_onset({value: block["levels"][_lambda_label(value)]["horizons"][name]["D_horizon"] for value in SOCIAL_LAMBDAS}) for name, _start, _stop in HORIZONS]
        block["horizon_onsets"] = {name: item for (name, _start, _stop), item in zip(HORIZONS, onsets, strict=True)}
        block["onset_stability"] = onset_stability(onsets)
        for label, level in block["levels"].items():
            counts = level["transition"]
            both_persistent = level["horizons"]["H500"]["D_horizon"] >= 0.025 and level["horizons"]["H2000"]["D_horizon"] >= 0.025
            level["turnover_interpretation"] = bool(both_persistent and counts["PP"] < counts["PA"] + counts["AP"])
            overlap = level["persistent_seed_jaccard"]
            level["stable_subset_interpretation"] = bool(
                counts["PP"] > counts["PA"] + counts["AP"] and overlap is not None and overlap >= JACCARD_STABLE_MIN
            )
    conclusion_input = {name: regimes[name] for name in ("RW50", "EXP99")}
    numeric = _numeric_summary([row["numerical"] for row in metrics if row["regime_id"] == "EXP99"])
    return {
        "experiment_id": "experiment_014_long_horizon_persistence",
        "note": SCOPE_SENTENCE,
        "horizon_rule": HORIZON_RULE,
        "h500_reference": references["h500"],
        "conclusion": overall_conclusion(conclusion_input),
        "gates": {
            "first_500_lineage": True,
            "h500_replication": True,
            "s0_lambda_equivalence": True,
            "blend_identity": True,
            "lambda1_endogenous_field": True,
            "numerical_stability": numeric["passed"],
            "pairs_checked": len(metrics),
        },
        "numerical_stability": numeric,
        "llm_calls": 0,
        "world_count": len(worlds),
        "seed_count": len(HELDOUT_SEEDS),
        "timesteps": TIMESTEPS,
        "regimes": regimes,
        "exact_replay": replays,
    }


def _regime_summary(spec, rows, worlds) -> dict[str, Any]:
    levels = {}
    for lambda_value in lambdas_for(spec["regime_id"]):
        label = _lambda_label(lambda_value)
        selected = [_lambda_row(row, lambda_value) for row in rows]
        horizons = {}
        for name, _start, _stop in HORIZONS:
            values = [item["horizons"][name]["D_horizon"] for item in selected]
            mean_value = statistics.fmean(values)
            horizons[name] = {
                "D_horizon": mean_value,
                "persistence_class": classify_persistence(mean_value),
                "seed_persistent_rate": statistics.fmean(float(item["horizons"][name]["seed_persistent"]) for item in selected),
                "late_tv": _dist(values),
                "late_js": _dist([item["horizons"][name]["late_js"] for item in selected]),
                "late_agent_disagreement": _dist([item["horizons"][name]["late_agent_disagreement"] for item in selected]),
                "late_probability_tv": _dist([item["horizons"][name]["late_probability_tv"] for item in selected]),
                "late_field_mad": _dist([item["horizons"][name]["late_field_mad"] for item in selected]),
                "late_endogenous_field_mad": _dist([item["horizons"][name]["late_endogenous_field_mad"] for item in selected]),
                "late_midpoint_mad": _dist([item["horizons"][name]["late_midpoint_mad"] for item in selected]),
            }
        flags_500 = [item["horizons"]["H500"]["seed_persistent"] for item in selected]
        flags_2000 = [item["horizons"]["H2000"]["seed_persistent"] for item in selected]
        counts = transition_counts(flags_500, flags_2000)
        early = {int(rows[index]["choice_seed"]) for index, flag in enumerate(flags_500) if flag}
        late = {int(rows[index]["choice_seed"]) for index, flag in enumerate(flags_2000) if flag}
        horizon_values = [horizons[name]["D_horizon"] for name, _start, _stop in HORIZONS]
        label_class = long_horizon_class(horizon_values)
        levels[label] = {
            "lambda": lambda_value,
            "horizons": horizons,
            "long_horizon_class": label_class,
            "long_horizon_code": CLASS_CODE[label_class],
            "transition": counts,
            "transition_proportion": {key: value / len(rows) for key, value in counts.items()},
            "persistent_seed_jaccard": jaccard(early, late),
            "divergence_duty_cycle": _dist([item["divergence_duty_cycle"] for item in selected]),
            "episode_count": _dist([item["episodes"]["episode_count"] for item in selected]),
            "mean_episode_length": _dist([item["episodes"]["mean_episode_length"] for item in selected]),
            "max_episode_length": _dist([item["episodes"]["max_episode_length"] for item in selected]),
            "last_divergent_t": _dist([item["episodes"]["last_divergent_t"] for item in selected]),
            "terminal_attenuation": _event_summary([item["terminal_attenuation_time"] for item in selected]),
            "terminal_exact_reconvergence": _event_summary([item["terminal_exact_reconvergence_time"] for item in selected]),
            "stable_attenuation_time_20": _dist([item["stable_attenuation_time_20"] for item in selected]),
            "stable_attenuation_rate_20": statistics.fmean(float(item["stable_attenuation_time_20"] is not None) for item in selected),
            "time_to_reconvergence_20": _dist([item["time_to_reconvergence_20"] for item in selected]),
            "descriptive_bins": [
                {"start": start, "stop": start + 99, "mean_occupancy_tv": statistics.fmean(item["descriptive_bins"][index]["mean_occupancy_tv"] for item in selected)}
                for index, start in enumerate(range(1, TIMESTEPS + 1, 100))
            ],
            "horizon_contrasts": {
                name: _effect([item["horizons"][name]["D_horizon"] - item["horizons"]["H500"]["D_horizon"] for item in selected])
                for name in ("H1000", "H1500", "H2000")
            },
            "spearman_horizon_index": _spearman(
                [horizon for item in selected for horizon in (500, 1000, 1500, 2000)],
                [item["horizons"][name]["D_horizon"] for item in selected for name, _start, _stop in HORIZONS],
            ),
            "safety": {
                history: _safety_pair(worlds, spec["regime_id"], history, lambda_value) for history in HISTORIES
            },
        }
    return {"n_seeds": len(rows), "levels": levels, "llm_calls": 0}


def _branch_safety(spec, seed, history, lambda_value, result) -> dict[str, Any]:
    diagnostics = measure_world(result["series"], moves=result["moves"], timesteps=TIMESTEPS)
    return {
        "regime_id": spec["regime_id"],
        "history": history,
        "lambda": lambda_value,
        "choice_seed": seed,
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "consecutive_occupancy_tv": diagnostics["mean_total_variation"],
        "lag1": diagnostics["lag1"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "late_1901_2000": _late_share(result["series"], *SAFETY_LATE),
        "late_401_500": _late_share(result["series"], *SAFETY_LINEAGE),
        "llm_calls": 0,
    }


def _late_share(series, start: int, stop: int) -> dict[str, float]:
    counts = {place_id: 0 for place_id in PLACE_IDS}
    for step in series[start : stop + 1]:
        for place_id in step.values():
            counts[place_id] += 1
    total = sum(counts.values())
    share = {place_id: counts[place_id] / total for place_id in PLACE_IDS}
    return {"hhi": _hhi(share), "dominant_share": max(share.values())}


def _safety_pair(worlds, regime_id, history, lambda_value) -> dict[str, Any]:
    selected = [world for world in worlds if world["regime_id"] == regime_id and world["history"] == history and world["lambda"] == lambda_value]
    return {
        "primary_1901_2000": _safety_summary(selected, "late_1901_2000"),
        "lineage_401_500": _safety_summary(selected, "late_401_500"),
    }


def _safety_summary(worlds, window_key: str) -> dict[str, Any]:
    lag1 = {place_id: statistics.fmean(world["lag1"][place_id] for world in worlds) for place_id in ("site_01", "site_02")}
    late_hhi = statistics.fmean(world[window_key]["hhi"] for world in worlds)
    dominant = statistics.fmean(world[window_key]["dominant_share"] for world in worlds)
    saturation = statistics.fmean(world["saturation_frequency"] for world in worlds)
    alternation = statistics.fmean(world["alternation_rate"] for world in worlds)
    consecutive = statistics.fmean(world["consecutive_occupancy_tv"] for world in worlds)
    return {
        "n": len(worlds),
        "alternation_rate": alternation,
        "consecutive_occupancy_tv": consecutive,
        "lag1_site_01": lag1["site_01"],
        "lag1_site_02": lag1["site_02"],
        "late_hhi": late_hhi,
        "late_dominant_share": dominant,
        "saturation_frequency": saturation,
        "window": window_key,
        "unstable": is_unstable(
            lag1_site_01=lag1["site_01"],
            lag1_site_02=lag1["site_02"],
            alternation=alternation,
            alternation_cutoff=ALTERNATION_UPPER_QUARTILE,
            late_hhi=late_hhi,
            dominant_share=dominant,
            saturation=saturation,
            consecutive_tv=consecutive,
            consecutive_tv_cutoff=CONSECUTIVE_TV_UPPER_QUARTILE,
        ),
    }


def _event_summary(values: list[int | None]) -> dict[str, Any]:
    observed = [int(value) for value in values if value is not None]
    return {
        "event_rate": len(observed) / len(values) if values else None,
        "right_censored_count": sum(value is None for value in values),
        "median_among_observed": statistics.median(observed) if observed else None,
        "kaplan_meier": kaplan_meier(values),
    }


def _dist(values: list[float | int | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"mean": None, "sd": None, "median": None, "q1": None, "q3": None, "min": None, "max": None, "n": 0}
    quarters = statistics.quantiles(finite, n=4, method="inclusive") if len(finite) >= 2 else [finite[0], finite[0], finite[0]]
    return {
        "mean": statistics.fmean(finite),
        "sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "median": statistics.median(finite),
        "q1": quarters[0],
        "q3": quarters[2],
        "min": min(finite),
        "max": max(finite),
        "n": len(finite),
    }


def _lambda_row(row, lambda_value: float) -> dict[str, Any]:
    return next(item for item in row["lambdas"] if item["lambda"] == lambda_value)


def _empty_numeric() -> dict[str, Any]:
    return {"nonfinite": 0, "exact_zero": 0, "clip_boundary": 0, "score_count": 0, "max_abs": 0.0, "exploded": False}


def _observe_memory(spec, memory, bucket) -> None:
    if spec["memory"] != "exponential":
        return
    payload = memory_payload(memory)
    values = list(payload["total"].values())
    for own in payload["own"].values():
        values.extend(own.values())
    for value in values:
        if not math.isfinite(float(value)):
            bucket["nonfinite"] += 1
        magnitude = abs(float(value))
        bucket["max_abs"] = max(bucket["max_abs"], magnitude)
        if float(value) == 0.0:
            bucket["exact_zero"] += 1
        if magnitude > ACCUMULATOR_LIMIT:
            bucket["exploded"] = True


def _observe_scores(field, bucket) -> None:
    for scores_by_agent in field.values():
        for scores in scores_by_agent.values():
            for value in scores.values():
                bucket["score_count"] += 1
                if not math.isfinite(value):
                    bucket["nonfinite"] += 1
                if value == 2.0 or value == -2.0:
                    bucket["clip_boundary"] += 1
                if value < -2.0 or value > 2.0:
                    raise RuntimeError("normalized social score left the clipped range")


def _numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_abs = max(row["max_abs"] for row in rows)
    nonfinite = sum(row["nonfinite"] for row in rows)
    exploded = any(row["exploded"] for row in rows)
    score_count = sum(row["score_count"] for row in rows)
    clip_count = sum(row["clip_boundary"] for row in rows)
    return {
        "passed": nonfinite == 0 and not exploded and max_abs <= ACCUMULATOR_LIMIT,
        "nonfinite_count": nonfinite,
        "exploded": exploded,
        "max_abs_accumulator": max_abs,
        "exact_zero_count": sum(row["exact_zero"] for row in rows),
        "clip_boundary_count": clip_count,
        "clip_boundary_frequency": clip_count / score_count if score_count else None,
        "stabilization_change": False,
    }


def _replay(atlas, initial, plans, metrics) -> dict[str, Any]:
    requested = (
        ("RW50", "PERTURBED", 0.4, 50),
        ("RW50", "CONTROL", 0.35, 75),
        ("EXP99", "PERTURBED", 0.35, 50),
        ("EXP99", "CONTROL", 0.4, 99),
        ("S0", "CONTROL", 0.0, 50),
        ("S0", "CONTROL", 1.0, 50),
    )
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    found = {}
    for regime_id, history, lambda_value, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        key = (regime_id, seed)
        if key not in cache:
            donors, _numeric = _collect_donors(atlas, initial, spec, seed, plans[key], timesteps=TIMESTEPS)
            cache[key] = donors
        result = simulate_history(
            atlas,
            initial,
            spec,
            choice_seed_value=seed,
            history=history,
            lambda_value=lambda_value,
            plan=plans[key],
            donors=cache[key],
            timesteps=TIMESTEPS,
        )
        stored = _lambda_row(next(row for row in metrics if row["regime_id"] == regime_id and row["choice_seed"] == seed), lambda_value)
        expected = stored["control_trajectory_sha256"] if history == "CONTROL" else stored["perturbed_trajectory_sha256"]
        if result["trajectory_sha256"] != expected:
            raise RuntimeError(f"{regime_id} {history} lambda {lambda_value} seed {seed} replay mismatch")
        found[f"{regime_id}_{history}_lambda_{_lambda_label(lambda_value)}_seed{seed}"] = {
            "trajectory_sha256": result["trajectory_sha256"],
            "match": True,
        }
    return found


def _config(seeds, timesteps, references) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_14",
        "experiment_id": "experiment_014_long_horizon_persistence",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "normalization": CENTERED_NORMALIZATION,
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "social_lambdas": list(SOCIAL_LAMBDAS),
        "s0_lambdas": list(S0_LAMBDAS),
        "horizons": [{"name": name, "start": start, "stop": stop} for name, start, stop in HORIZONS],
        "intervention_source": "data/worlds/world_v0_13/heldout_interventions.jsonl",
        "lineage_source": "data/worlds/world_v0_13/run_manifest.json",
        "experiment_013_h500": references["h500"],
        "horizon_rule": HORIZON_RULE,
        "jaccard_stable_min": JACCARD_STABLE_MIN,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "safety_primary_window": list(SAFETY_LATE),
        "safety_lineage_window": list(SAFETY_LINEAGE),
        "unstable_thresholds": {
            "alternation_upper_quartile": ALTERNATION_UPPER_QUARTILE,
            "consecutive_tv_upper_quartile": CONSECUTIVE_TV_UPPER_QUARTILE,
            "extreme_late_hhi": EXTREME_LATE_HHI,
            "extreme_dominant_share": EXTREME_DOMINANT_SHARE,
            "extreme_saturation": EXTREME_SATURATION,
        },
        "scope": SCOPE_SENTENCE,
        "llm_calls": 0,
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _horizon_analysis(summary) -> dict[str, Any]:
    return {
        "experiment_id": summary["experiment_id"],
        "rule": HORIZON_RULE,
        "conclusion": summary["conclusion"],
        "regimes": {
            regime_id: {
                "onset_stability": block.get("onset_stability"),
                "horizon_onsets": block.get("horizon_onsets"),
                "levels": {
                    label: {
                        "long_horizon_class": level["long_horizon_class"],
                        "long_horizon_code": level["long_horizon_code"],
                        "D_horizon": {name: level["horizons"][name]["D_horizon"] for name in level["horizons"]},
                        "persistence_class": {name: level["horizons"][name]["persistence_class"] for name in level["horizons"]},
                        "seed_persistent_rate": {name: level["horizons"][name]["seed_persistent_rate"] for name in level["horizons"]},
                        "horizon_contrasts": level["horizon_contrasts"],
                        "late_field_mad": {name: level["horizons"][name]["late_field_mad"]["mean"] for name in level["horizons"]},
                    }
                    for label, level in block["levels"].items()
                },
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


def _transition_analysis(summary) -> dict[str, Any]:
    return {
        "experiment_id": summary["experiment_id"],
        "conclusion": summary["conclusion"],
        "regimes": {
            regime_id: {
                label: {
                    "transition": level["transition"],
                    "transition_proportion": level["transition_proportion"],
                    "persistent_seed_jaccard": level["persistent_seed_jaccard"],
                    "turnover_interpretation": level.get("turnover_interpretation"),
                    "stable_subset_interpretation": level.get("stable_subset_interpretation"),
                }
                for label, level in block["levels"].items()
            }
            for regime_id, block in summary["regimes"].items()
            if regime_id in ("RW50", "EXP99")
        },
    }


def _episode_analysis(summary) -> dict[str, Any]:
    return {
        "experiment_id": summary["experiment_id"],
        "duty_cycle_window": [501, 2000],
        "regimes": {
            regime_id: {
                label: {
                    "divergence_duty_cycle": level["divergence_duty_cycle"],
                    "episode_count": level["episode_count"],
                    "mean_episode_length": level["mean_episode_length"],
                    "max_episode_length": level["max_episode_length"],
                    "last_divergent_t": level["last_divergent_t"],
                }
                for label, level in block["levels"].items()
            }
            for regime_id, block in summary["regimes"].items()
            if regime_id in ("RW50", "EXP99")
        },
    }


def _terminal_analysis(summary) -> dict[str, Any]:
    return {
        "experiment_id": summary["experiment_id"],
        "note": "Null event times are right-censored at the observed horizon. They are not coded as 2000. The 20-step metric is separate and is not the long-horizon conclusion.",
        "regimes": {
            regime_id: {
                label: {
                    "terminal_attenuation": level["terminal_attenuation"],
                    "terminal_exact_reconvergence": level["terminal_exact_reconvergence"],
                    "stable_attenuation_rate_20": level["stable_attenuation_rate_20"],
                    "stable_attenuation_time_20": level["stable_attenuation_time_20"],
                }
                for label, level in block["levels"].items()
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


def _compare(summary) -> dict[str, Any]:
    return {
        "experiment_id": summary["experiment_id"],
        "note": "RW50 and EXP99 differences are architecture sensitivity, not a short-memory versus long-memory causal continuum.",
        "conclusion": summary["conclusion"],
        "regimes": {
            regime_id: {
                "onset_stability": block.get("onset_stability"),
                "long_horizon_class": {label: level["long_horizon_class"] for label, level in block["levels"].items()},
                "D_H2000": {label: level["horizons"]["H2000"]["D_horizon"] for label, level in block["levels"].items()},
                "duty_cycle_mean": {label: level["divergence_duty_cycle"]["mean"] for label, level in block["levels"].items()},
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


def _manifest(destination, config, summary, metrics) -> dict[str, Any]:
    trajectories = {}
    for row in metrics:
        for level in row["lambdas"]:
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:CONTROL:{level['lambda_label']}"] = level["control_trajectory_sha256"]
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:PERTURBED:{level['lambda_label']}"] = level["perturbed_trajectory_sha256"]
    return {
        "experiment_id": summary["experiment_id"],
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "seed_count": summary["seed_count"],
        "timesteps": TIMESTEPS,
        "complete": summary["world_count"] == 1000,
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "gates": summary["gates"],
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": trajectories,
        "files": {
            name: _file_sha256(destination / name)
            for name in (
                "world_v0_14_config.json",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "long_horizon_donor_social_fields.jsonl",
                "long_horizon_midpoint_fields.jsonl",
                "feedback_blend_audit.jsonl",
                "long_horizon_pair_metrics.jsonl",
                "horizon_persistence_analysis.json",
                "seed_transition_analysis.json",
                "divergence_episode_analysis.json",
                "terminal_attenuation_analysis.json",
                "regime_comparison.json",
                "experiment_014_summary.json",
            )
        },
    }


def _write_donors(handle, spec, seed, donors) -> None:
    for history, donor_name in (("CONTROL", "CONTROL_ENDOGENOUS"), ("PERTURBED", "PERTURBED_ENDOGENOUS")):
        for timestep in range(ENVIRONMENT_STEP, TIMESTEPS + 1):
            for agent_id in sorted(donors[history][timestep]):
                scores = donors[history][timestep][agent_id]
                _dump(
                    handle,
                    {
                        "regime": spec["regime_id"],
                        "choice_seed": seed,
                        "donor": donor_name,
                        "timestep": timestep,
                        "agent_id": agent_id,
                        "scores": scores,
                        "llm_calls": 0,
                    },
                )


def _write_midpoint(handle, spec, seed, donors) -> None:
    for timestep in range(ENVIRONMENT_STEP, TIMESTEPS + 1):
        control = donors["CONTROL"][timestep]
        perturbed = donors["PERTURBED"][timestep]
        for agent_id in sorted(control):
            scores = midpoint_scores(control[agent_id], perturbed[agent_id])
            if any(value < -2.0 or value > 2.0 for value in scores.values()):
                raise RuntimeError("midpoint left the convex range")
            _dump(
                handle,
                {
                    "regime": spec["regime_id"],
                    "choice_seed": seed,
                    "timestep": timestep,
                    "agent_id": agent_id,
                    "scores": scores,
                    "renormalized": False,
                    "recentered": False,
                    "reclipped": False,
                    "llm_calls": 0,
                },
            )


def _write_lambda_audit(handle, spec, seed, history, lambda_value, result, donors) -> None:
    for timestep in range(ENVIRONMENT_STEP, TIMESTEPS + 1):
        endogenous = result["endogenous_field"][timestep]
        control = donors["CONTROL"][timestep]
        perturbed = donors["PERTURBED"][timestep]
        max_error = 0.0
        checked = 0
        for agent_id in endogenous:
            for place_id in PLACE_IDS:
                midpoint = 0.5 * (control[agent_id][place_id] + perturbed[agent_id][place_id])
                expected = (1.0 - lambda_value) * midpoint + lambda_value * endogenous[agent_id][place_id]
                max_error = max(max_error, abs(endogenous[agent_id][place_id] - expected))
                checked += 1
        if not math.isfinite(max_error) or max_error > BLEND_TOLERANCE:
            raise RuntimeError("lambda 1 blend identity failed")
        _dump(
            handle,
            {
                "regime": spec["regime_id"],
                "choice_seed": seed,
                "lambda": lambda_value,
                "history": history,
                "timestep": timestep,
                "max_absolute_error": max_error,
                "checked_agent_place_count": checked,
                "identity": "(1-lambda)*midpoint_score + lambda*endogenous_score",
                "shadow_memory_hash": result["memory_hashes"][timestep],
                "decision_field_hash": canonical_sha256(endogenous),
                "llm_calls": 0,
            },
        )


def _compact_audit(spec, seed, history, lambda_value, logged, audit) -> dict[str, Any]:
    max_error = 0.0
    checked = 0
    for agent_id, used_scores in audit["used"].items():
        for place_id, used in used_scores.items():
            expected = (1.0 - lambda_value) * audit["midpoint"][agent_id][place_id] + lambda_value * audit["endogenous"][agent_id][place_id]
            max_error = max(max_error, abs(used - expected))
            checked += 1
    if not math.isfinite(max_error) or max_error > BLEND_TOLERANCE:
        raise RuntimeError("compact blend audit exceeded tolerance")
    return {
        "regime": spec["regime_id"],
        "choice_seed": seed,
        "lambda": lambda_value,
        "history": history,
        "timestep": logged,
        "max_absolute_error": max_error,
        "checked_agent_place_count": checked,
        "identity": "(1-lambda)*midpoint_score + lambda*endogenous_score",
        "shadow_memory_hash": audit["memory_hash"],
        "decision_field_hash": audit["decision_hash"],
        "llm_calls": 0,
    }


def _write_step(states, memories, actions, spec, seed, history, lambda_value, logged, current, memory, step_actions) -> None:
    if logged == 1:
        _dump_state(states, spec, seed, history, lambda_value, 0, {action["agent_id"]: action["origin"] for action in step_actions})
    _dump_state(states, spec, seed, history, lambda_value, logged, current)
    payload = memory_payload(memory)
    record = {
        "regime_id": spec["regime_id"],
        "history": history,
        "lambda": lambda_value,
        "choice_seed": seed,
        "timestep": logged,
        "architecture": payload["architecture"],
        "llm_calls": 0,
    }
    if payload["architecture"] == "rolling":
        record["window"] = payload["window"]
        record["commit_count"] = len(payload["commits"])
        record["total"] = payload["total"]
    else:
        record["decay"] = payload["decay"]
        record["total"] = payload["total"]
        record["own"] = payload["own"]
    _dump(memories, record)
    for action in step_actions:
        _dump(
            actions,
            {
                "regime_id": spec["regime_id"],
                "choice_seed": seed,
                "lambda": lambda_value,
                "history": history,
                "timestep": action["timestep"],
                "agent_id": action["agent_id"],
                "origin": action["origin"],
                "destination": action["destination"],
                "stayed": action["stayed"],
                "reconsidered": action["reconsidered"],
                "score_source": action["score_source"],
                "intervention_applied": action["intervention_applied"],
                "rng_draws": action["rng_draws"],
                "llm_calls": 0,
            },
        )


def _dump_state(handle, spec, seed, history, lambda_value, logged, locations) -> None:
    if not locations:
        return
    ordered = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    _dump(
        handle,
        {
            "regime_id": spec["regime_id"],
            "history": history,
            "lambda": lambda_value,
            "choice_seed": seed,
            "timestep": logged,
            "locations": ordered,
            "occupancy": occupancy_of(ordered),
            "llm_calls": 0,
        },
    )


def _require_atlas_hash() -> None:
    manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != EXPECTED_ATLAS_SHA256:
        raise RuntimeError("utility atlas hash is not the frozen experiment atlas")


HASH_CHUNK_BYTES = 1024 * 1024


def _file_sha256(path: Path, chunk_size: int = HASH_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_nonempty_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("experiment 014 refuses to write into a non-empty output directory")


def _refuse_frozen_directories(destination: Path) -> None:
    forbidden = {
        path.resolve()
        for path in (
            WORLD_V01_DIR,
            WORLD_V02_DIR,
            WORLD_V03_DIR,
            WORLD_V04_DIR,
            WORLD_V05_DIR,
            WORLD_V06_DIR,
            WORLD_V07_DIR,
            WORLD_V08_DIR,
            WORLD_V09_DIR,
            WORLD_V10_DIR,
            WORLD_V11_DIR,
            WORLD_V12_DIR,
            WORLD_V13_DIR,
            ATLAS_DIR,
            CALIBRATION_DIR,
        )
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 014 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_014")
    main()
