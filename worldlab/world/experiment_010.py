"""Experiment 010: closed-loop social feedback necessity.

The t=21 perturbation is copied from Experiment 007. From t=22, two ablations
sever either branch-specific social feedback or all social influence.
This module does not call a model and does not retune rules after seeing results.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import world.experiment_005 as memory_experiment
from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import OUTPUT_DIR as CALIBRATION_DIR
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_003 import OUTPUT_DIR as WORLD_V03_DIR
from world.experiment_004 import (
    OUTPUT_DIR as WORLD_V04_DIR,
    _entropy,
    _hhi,
    _js_distance,
    _select,
    _share,
    _tv,
    trajectory_hash,
)
from world.experiment_005 import (
    CENTERED_NORMALIZATION,
    FAMILIARITY_WEIGHT,
    OUTPUT_DIR as WORLD_V05_DIR,
    DecayMemory,
    RollingMemory,
    _argmax,
    _utilities,
    centered_scores,
    is_unstable,
)
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
from world.experiment_009 import (
    OUTPUT_DIR as WORLD_V09_DIR,
    classify_persistence,
    load_intervention_plans,
    memory_distance,
    memory_payload,
    raw_memory_vector,
)
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_10"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
EXPECTED_S0_SEED0 = "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
INTERVENTION_STEP = 21
ABLATION_STEP = 22
OVERRIDE_COUNT = 2
INITIAL_TV = 0.10
ATTENUATION_RUN = 20
RECONVERGENCE_RUN = 20
ANALYSIS_BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 10000
BRANCHES = ("CONTROL_CLOSED", "PERTURBED_CLOSED", "PERTURBED_CONTROL_FIELD", "PERTURBED_SOCIAL_OFF")
PERTURBED_BRANCHES = ("PERTURBED_CLOSED", "PERTURBED_CONTROL_FIELD", "PERTURBED_SOCIAL_OFF")
REGIMES = (
    {"regime_id": "S0", "memory": "rolling", "window": 50, "decay": None, "social_weight": 0.0},
    {"regime_id": "RW50", "memory": "rolling", "window": 50, "decay": None, "social_weight": 100.0},
    {"regime_id": "EXP99", "memory": "exponential", "window": None, "decay": 0.99, "social_weight": 100.0},
)
WINDOWS = (
    ("intervention", 21, 21),
    ("early", 21, 50),
    ("early_mid", 51, 100),
    ("mid", 101, 200),
    ("late_mid", 201, 300),
    ("late", 301, 400),
    ("final_late", 401, 500),
)
LINEAGE_BRANCH = {"CONTROL_CLOSED": "CONTROL", "PERTURBED_CLOSED": "PERTURBED"}
ANALYSIS_NOTE = (
    "From t=22, control-field replay keeps social influence but replaces the branch social scores "
    "with the matched control field. Social-off removes ongoing social influence. "
    "These are closure ablations inside the frozen simulation. They do not establish collective path dependence, "
    "human social feedback, real-world causality, emergence, or self-organization."
)
CLOSURE_RULE = (
    "F means control-field late TV is at least 0.025. O means social-off late TV is at least 0.025. "
    "Closed-loop persistence is required before a closure type is assigned. "
    "Type A: F false and O false. Type B: F true and O false. Type C: F false and O true. Type D: F true and O true. "
    "Retention fractions are descriptive ratios of regime-mean late TV, not mediation shares."
)
CONCLUSION_RULE = (
    "If a hard gate fails: Experiment 010 closure classification is invalid pending feedback-path implementation audit. "
    "If a social closed-loop branch is attenuated: Experiment 010 mechanism classification invalid. "
    "If RW50 and EXP99 closure types differ: the necessity of endogenous feedback closure depends on social-memory architecture. "
    "Type A: persistent collective residuals require branch-endogenous social feedback closure under the tested simulation architecture. "
    "Type B: ongoing social influence sustains persistent collective divergence even without branch-endogenous feedback closure. "
    "Type C: persistent divergence survives removal of social influence, while replay of the control social field suppresses that residual under the tested architecture. "
    "Type D: persistent collective divergence survives removal of ongoing endogenous social feedback closure after the initial perturbation."
)
ATTENUATION_RULE = (
    "Stable attenuation is the first completion of 20 consecutive timesteps with occupancy TV below 0.025, "
    "counted only from t=22. The recorded time is the timestep that completes that run. Otherwise null."
)
RECONVERGENCE_RULE = (
    "Reconvergence is 20 consecutive timesteps with occupancy TV exactly 0, counted only from t=21. "
    "It does not replace the persistence classification."
)
GATE_FAILURE_CONCLUSION = (
    "Experiment 010 closure classification is invalid pending feedback-path implementation audit"
)
CLOSED_FAILURE_CONCLUSION = "Experiment 010 mechanism classification invalid"


@contextmanager
def _weight(value: float) -> Iterator[None]:
    original = memory_experiment.SOCIAL_WEIGHT
    memory_experiment.SOCIAL_WEIGHT = float(value)
    try:
        yield
    finally:
        memory_experiment.SOCIAL_WEIGHT = original


def classify_closure(closed_class: str, field_late_tv: float, off_late_tv: float) -> str | None:
    if closed_class == "attenuated":
        return None
    field_persists = field_late_tv >= 0.025
    off_persists = off_late_tv >= 0.025
    if not field_persists and not off_persists:
        return "A"
    if field_persists and not off_persists:
        return "B"
    if not field_persists and off_persists:
        return "C"
    return "D"


def retention_fraction(component: float, closed: float) -> float | None:
    if closed == 0.0:
        return None
    return component / closed


def overall_conclusion(closures: dict[str, str | None] | None, *, gates_ok: bool) -> str:
    if not gates_ok or not closures:
        return GATE_FAILURE_CONCLUSION
    if any(closures[name] is None for name in ("RW50", "EXP99")):
        return CLOSED_FAILURE_CONCLUSION
    if closures["RW50"] != closures["EXP99"]:
        return "the necessity of endogenous feedback closure depends on social-memory architecture"
    sentences = {
        "A": "persistent collective residuals require branch-endogenous social feedback closure under the tested simulation architecture",
        "B": "ongoing social influence sustains persistent collective divergence even without branch-endogenous feedback closure",
        "C": "persistent divergence survives removal of social influence, while replay of the control social field suppresses that residual under the tested architecture",
        "D": "persistent collective divergence survives removal of ongoing endogenous social feedback closure after the initial perturbation",
    }
    return sentences[str(closures["RW50"])]


def stable_attenuation_time(values: list[float]) -> int | None:
    run = 0
    for timestep in range(ABLATION_STEP, len(values)):
        if values[timestep] < 0.025:
            run += 1
            if run >= ATTENUATION_RUN:
                return timestep
        else:
            run = 0
    return None


def reconvergence_time(values: list[float]) -> int | None:
    run = 0
    for timestep in range(INTERVENTION_STEP, len(values)):
        if values[timestep] == 0.0:
            run += 1
            if run >= RECONVERGENCE_RUN:
                return timestep
        else:
            run = 0
    return None


def simulate_branch(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    spec: dict[str, Any],
    *,
    choice_seed_value: int,
    branch: str,
    plan: dict[str, Any],
    timesteps: int = TIMESTEPS,
    social_field: dict[int, dict[str, dict[str, float]]] | None = None,
    on_step: Any = None,
) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise RuntimeError("unknown closure branch")
    if branch == "PERTURBED_CONTROL_FIELD" and social_field is None:
        raise RuntimeError("control-field replay requires the control social field")
    matrix = distance_matrix()
    places = place_index()
    weight = float(spec["social_weight"])
    memory = _make_memory(spec, list(locations))
    current = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    series = [dict(current)]
    memory_trace = [raw_memory_vector(memory)]
    collected: dict[int, dict[str, dict[str, float]]] = {}
    score_gap = [0.0]
    utility_gap = [0.0]
    probability_trace = [{}]
    audits = []
    dynamics = hashlib.sha256()
    with _weight(weight):
        for timestep in range(timesteps):
            logged = timestep + 1
            occupancy = occupancy_of(current)
            decisions: dict[str, str] = {}
            actions = []
            step_scores: dict[str, dict[str, float]] = {}
            step_gap = 0.0
            step_utility_gap = 0.0
            for agent_id in sorted(current):
                origin = current[agent_id]
                _unclipped, endogenous = centered_scores(memory.loo_raw(agent_id))
                bases, totals, social_term = _utilities(
                    atlas, matrix, places, agent_id, origin, occupancy, endogenous
                )
                score_source = "endogenous"
                used_scores = {place_id: float(endogenous[place_id]) for place_id in PLACE_IDS}
                if branch == "PERTURBED_CONTROL_FIELD" and logged >= ABLATION_STEP:
                    used_scores = {
                        place_id: float(social_field[logged][agent_id][place_id]) for place_id in PLACE_IDS
                    }
                    social_term = {place_id: weight * used_scores[place_id] for place_id in PLACE_IDS}
                    totals = {place_id: bases[place_id] + social_term[place_id] for place_id in PLACE_IDS}
                    score_source = "control_field"
                    if not _scores_equal(used_scores, social_field[logged][agent_id]):
                        raise RuntimeError("control-field score was recomputed from recipient state")
                elif branch == "PERTURBED_SOCIAL_OFF" and logged >= ABLATION_STEP:
                    social_term = {place_id: 0.0 for place_id in PLACE_IDS}
                    totals = {place_id: bases[place_id] for place_id in PLACE_IDS}
                    score_source = "social_off"
                social_term = {place_id: _positive_zero(social_term[place_id]) for place_id in PLACE_IDS}
                totals = {place_id: bases[place_id] + social_term[place_id] for place_id in PLACE_IDS}
                if branch == "PERTURBED_SOCIAL_OFF" and logged >= ABLATION_STEP:
                    if any(value != 0.0 for value in social_term.values()):
                        raise RuntimeError("social-off utility was not exactly zero")
                if social_field is not None and logged in social_field:
                    reference_term = {
                        place_id: weight * float(social_field[logged][agent_id][place_id]) for place_id in PLACE_IDS
                    }
                    step_gap = max(step_gap, _max_abs(used_scores, social_field[logged][agent_id]))
                    step_utility_gap = max(step_utility_gap, _max_abs(social_term, reference_term))
                does_reconsider = agent_reconsiders(
                    choice_seed_value, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE
                )
                ranking_changed = _argmax(totals) != _argmax(bases)
                if not does_reconsider:
                    destination = origin
                    probabilities = {place_id: 1.0 if place_id == origin else 0.0 for place_id in PLACE_IDS}
                    destination_changed = False
                else:
                    draw = random.Random(choice_seed(choice_seed_value, timestep, agent_id)).random()
                    destination, probabilities = _select(
                        list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw
                    )
                    _base_destination, _base_probabilities = _select(
                        list(PLACE_IDS), [bases[place_id] for place_id in PLACE_IDS], draw
                    )
                    destination_changed = destination != _base_destination
                decisions[agent_id] = destination
                step_scores[agent_id] = used_scores
                actions.append(
                    {
                        "timestep": logged,
                        "agent_id": agent_id,
                        "origin": origin,
                        "destination": destination,
                        "proposed_destination": destination,
                        "stayed": destination == origin,
                        "reconsidered": does_reconsider,
                        "social_weight": weight,
                        "effective_social_weight": 0.0 if score_source == "social_off" else weight,
                        "familiarity_weight": FAMILIARITY_WEIGHT,
                        "normalization": CENTERED_NORMALIZATION,
                        "branch": branch,
                        "score_source": score_source,
                        "intervention_applied": False,
                        "base_utility": bases,
                        "popularity_score": used_scores,
                        "social_utility": social_term,
                        "total_utility": totals,
                        "choice_probabilities": probabilities,
                        "occupancy_before": occupancy,
                        "ranking_changed": ranking_changed,
                        "destination_changed": destination_changed,
                        "rng_draws": 2 if does_reconsider else 1,
                        "llm_calls": 0,
                    }
                )
            if logged == INTERVENTION_STEP and branch in PERTURBED_BRANCHES:
                chosen = set(plan["selected_agents"])
                for agent_id in plan["selected_agents"]:
                    decisions[agent_id] = plan["target_place"]
                for action in actions:
                    if action["agent_id"] in chosen:
                        action["destination"] = plan["target_place"]
                        action["stayed"] = action["destination"] == action["origin"]
                        action["intervention_applied"] = True
                if sum(bool(action["intervention_applied"]) for action in actions) != OVERRIDE_COUNT:
                    raise RuntimeError("perturbation did not override exactly two agents")
            elif logged == INTERVENTION_STEP:
                pass
            current = {agent_id: decisions[agent_id] for agent_id in sorted(decisions)}
            memory.commit(decisions)
            series.append(dict(current))
            memory_trace.append(raw_memory_vector(memory))
            score_gap.append(step_gap)
            utility_gap.append(step_utility_gap)
            probability_trace.append({action["agent_id"]: dict(action["choice_probabilities"]) for action in actions})
            if branch == "CONTROL_CLOSED" and logged >= ABLATION_STEP:
                collected[logged] = step_scores
            if branch == "PERTURBED_CONTROL_FIELD" and logged >= ABLATION_STEP:
                reference_hash = canonical_sha256(social_field[logged])
                decision_hash = canonical_sha256(step_scores)
                audits.append(
                    {
                        "regime": spec["regime_id"],
                        "choice_seed": choice_seed_value,
                        "timestep": logged,
                        "decision_field_hash": decision_hash,
                        "control_reference_field_hash": reference_hash,
                        "shadow_memory_hash": canonical_sha256(memory_payload(memory)),
                        "field_match": decision_hash == reference_hash,
                        "llm_calls": 0,
                    }
                )
                if decision_hash != reference_hash:
                    raise RuntimeError(f"{spec['regime_id']} seed {choice_seed_value} control field mismatch at t={logged}")
            _extend_hash(dynamics, _step_payload(logged, current, actions, memory))
            if on_step is not None:
                on_step(logged, current, memory, actions, collected if logged >= ABLATION_STEP else None)
    return {
        "series": series,
        "moves": _movement(series),
        "memory_trace": memory_trace,
        "social_field": collected,
        "score_gap": score_gap,
        "decision_field_gap": utility_gap,
        "probabilities": probability_trace,
        "audits": audits,
        "override_steps": [INTERVENTION_STEP] if branch in PERTURBED_BRANCHES else [],
        "canonical_dynamics_sha256": dynamics.hexdigest(),
        "trajectory_sha256": trajectory_hash(series),
        "llm_calls": 0,
    }


def run_experiment(
    atlas: FrozenUtilityAtlas | None = None,
    *,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = CHOICE_SEEDS,
    timesteps: int = TIMESTEPS,
) -> Path:
    if timesteps != TIMESTEPS or seeds != CHOICE_SEEDS:
        raise RuntimeError("the preregistered run is 50 seeds and 500 timesteps")
    destination = output_dir or OUTPUT_DIR
    _refuse_frozen_directories(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = load_frozen_atlas() if atlas is None else atlas
    _require_atlas_hash()
    initial = load_initial_locations()
    plans = load_intervention_plans()
    references = _reference_hashes()
    carriers = _exp009_carriers()
    config = _config(seeds, timesteps, carriers)
    write_json(destination / "world_v0_10_config.json", config)
    names = (
        "world_states.jsonl",
        "social_memory_states.jsonl",
        "agent_actions.jsonl",
        "control_social_field.jsonl",
        "control_field_replay_audit.jsonl",
        "feedback_ablation_metrics.jsonl",
    )
    paths = {name: destination / name for name in names}
    for path in paths.values():
        path.write_text("", encoding="utf-8", newline="\n")
    metrics = []
    branch_worlds = []
    with paths["world_states.jsonl"].open("a", encoding="utf-8", newline="\n") as states, paths[
        "social_memory_states.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as memories, paths["agent_actions.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as actions, paths["control_social_field.jsonl"].open("a", encoding="utf-8", newline="\n") as fields, paths[
        "control_field_replay_audit.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as audits, paths["feedback_ablation_metrics.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as rows:
        for spec in REGIMES:
            for seed in seeds:
                runs = _run_seed(
                    source, initial, spec, seed, plans[(spec["regime_id"], seed)], references, states, memories, actions, fields, audits
                )
                record = _seed_metrics(spec, seed, runs, plans[(spec["regime_id"], seed)])
                for result in runs.values():
                    result.pop("memory_trace", None)
                    result.pop("social_field", None)
                    result.pop("probabilities", None)
                metrics.append(record)
                for branch, result in runs.items():
                    branch_worlds.append(_branch_safety(spec, seed, branch, result))
                _dump(rows, record)
                print(
                    f"{spec['regime_id']} seed={seed} closed={record['D_closed']:.4f} "
                    f"field={record['D_control_field']:.4f} off={record['D_social_off']:.4f}",
                    flush=True,
                )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    replays = _replay(source, initial, plans, metrics)
    summary = _summarize(metrics, branch_worlds, replays, carriers)
    write_json(destination / "experiment_010_summary.json", summary)
    write_json(destination / "regime_comparison.json", _compare(metrics, summary))
    write_json(destination / "closure_analysis.json", _closure(summary))
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, metrics))
    return destination


def _run_seed(atlas, initial, spec, seed, plan, references, states, memories, actions, fields, audits) -> dict[str, dict[str, Any]]:
    runs = {}
    control = _run_logged(
        atlas, initial, spec, seed, "CONTROL_CLOSED", plan, None, states, memories, actions, fields
    )
    expected = references[f"{spec['regime_id']}:{seed}:CONTROL"]
    if control["trajectory_sha256"] != expected:
        raise RuntimeError(f"{spec['regime_id']} CONTROL_CLOSED seed {seed} does not match Experiment 007")
    runs["CONTROL_CLOSED"] = control
    field = control["social_field"]
    for branch in PERTURBED_BRANCHES:
        result = _run_logged(
            atlas, initial, spec, seed, branch, plan, field, states, memories, actions, None
        )
        if branch == "PERTURBED_CLOSED":
            perturbed_expected = references[f"{spec['regime_id']}:{seed}:PERTURBED"]
            if result["trajectory_sha256"] != perturbed_expected:
                raise RuntimeError(f"{spec['regime_id']} PERTURBED_CLOSED seed {seed} does not match Experiment 007")
        if branch == "PERTURBED_CONTROL_FIELD":
            if not result["audits"] or any(not row["field_match"] for row in result["audits"]):
                raise RuntimeError(f"{spec['regime_id']} seed {seed} control-field replay failed")
            for row in result["audits"]:
                _dump(audits, row)
        runs[branch] = result
    _require_seed_gates(spec, seed, runs)
    return runs


def _run_logged(atlas, initial, spec, seed, branch, plan, field, states, memories, actions, fields) -> dict[str, Any]:
    _write_initial(states, memories, spec, seed, branch, initial)

    def write(logged, current, memory, step_actions, collected):
        _dump_state(states, spec, seed, branch, logged, current)
        payload = memory_payload(memory)
        _dump(
            memories,
            {
                "regime_id": spec["regime_id"],
                "branch": branch,
                "choice_seed": seed,
                "timestep": logged,
                "social_weight": spec["social_weight"],
                "normalization": CENTERED_NORMALIZATION,
                "architecture": payload["architecture"],
                "raw_total": payload["total"],
                "raw_own": payload["own"],
                "commit_count": len(payload["commits"]) if payload["architecture"] == "rolling" else None,
                "decision_relevant": branch != "PERTURBED_CONTROL_FIELD" or logged < ABLATION_STEP,
                "llm_calls": 0,
            },
        )
        for action in step_actions:
            _dump(actions, {"regime_id": spec["regime_id"], "choice_seed": seed, **action})
        if fields is not None and collected is not None and logged >= ABLATION_STEP:
            for agent_id in sorted(collected[logged]):
                _dump(
                    fields,
                    {
                        "regime": spec["regime_id"],
                        "choice_seed": seed,
                        "timestep": logged,
                        "agent_id": agent_id,
                        "scores": collected[logged][agent_id],
                    },
                )

    result = simulate_branch(
        atlas,
        initial,
        spec,
        choice_seed_value=seed,
        branch=branch,
        plan=plan,
        social_field=field,
        on_step=write,
    )
    if result["llm_calls"] != 0:
        raise RuntimeError("experiment 010 called a model")
    return result


def _require_seed_gates(spec, seed, runs) -> None:
    control = runs["CONTROL_CLOSED"]["series"][:INTERVENTION_STEP]
    perturbed_prefix = runs["PERTURBED_CLOSED"]["series"][: ABLATION_STEP]
    for branch, result in runs.items():
        if result["series"][:INTERVENTION_STEP] != control:
            raise RuntimeError(f"{spec['regime_id']} seed {seed} {branch} diverged before t=21")
    for branch in PERTURBED_BRANCHES:
        if runs[branch]["series"][:ABLATION_STEP] != perturbed_prefix:
            raise RuntimeError(f"{branch} t=21 intervention does not match the closed perturbation")
        if runs[branch]["override_steps"] != [INTERVENTION_STEP]:
            raise RuntimeError(f"{branch} intervention was not confined to t=21")
    if spec["regime_id"] == "S0":
        closed_hash = runs["PERTURBED_CLOSED"]["canonical_dynamics_sha256"]
        for branch in ("PERTURBED_CONTROL_FIELD", "PERTURBED_SOCIAL_OFF"):
            if runs[branch]["canonical_dynamics_sha256"] != closed_hash:
                raise RuntimeError("S0 ablation changed a path other than the social term")
    if spec["regime_id"] == "S0" and seed == 0 and runs["CONTROL_CLOSED"]["trajectory_sha256"] != EXPECTED_S0_SEED0:
        raise RuntimeError("S0 seed 0 hash gate failed")


def _seed_metrics(spec, seed, runs, plan) -> dict[str, Any]:
    control_series = runs["CONTROL_CLOSED"]["series"]
    control_trace = runs["CONTROL_CLOSED"]["memory_trace"]
    branches = {}
    for branch, result in runs.items():
        tv, js, disagreement = [], [], []
        agents = sorted(control_series[0])
        population = len(agents)
        for timestep in range(len(result["series"])):
            tv.append(_tv(_share_of(result["series"][timestep]), _share_of(control_series[timestep])))
            js.append(_js_distance(_share_of(result["series"][timestep]), _share_of(control_series[timestep])))
            disagreement.append(
                sum(result["series"][timestep][agent_id] != control_series[timestep][agent_id] for agent_id in agents) / population
            )
        shadow = [memory_distance(left, right) for left, right in zip(result["memory_trace"], control_trace, strict=True)]
        probability_tv = _probability_tv(result["probabilities"], runs["CONTROL_CLOSED"]["probabilities"])
        if len(tv) != TIMESTEPS + 1:
            raise RuntimeError("metric window requires the 500-step run")
        late_tv = statistics.fmean(tv[401:501])
        attenuated_at = stable_attenuation_time(tv)
        reconverged_at = reconvergence_time(tv)
        branches[branch] = {
            "late_tv": late_tv,
            "persistence_class": classify_persistence(late_tv),
            "windows": {
                name: {
                    "occupancy_tv": statistics.fmean(tv[start : stop + 1]),
                    "js_distance": statistics.fmean(js[start : stop + 1]),
                    "agent_disagreement": statistics.fmean(disagreement[start : stop + 1]),
                    "probability_tv": statistics.fmean(probability_tv[start : stop + 1]),
                }
                for name, start, stop in WINDOWS
            },
            "post_t22_choice_disagreement_rate": statistics.fmean(disagreement[ABLATION_STEP : TIMESTEPS + 1]),
            "stable_attenuation_time": attenuated_at,
            "ever_reconverged": reconverged_at is not None,
            "time_to_reconvergence": reconverged_at,
            "first_choice_divergence_t": _first_location_divergence(result["series"], control_series),
            "first_agent_location_divergence_t": _first_location_divergence(result["series"], control_series),
            "first_occupancy_tv_ge_025_t": _first_true(tv, lambda value: value >= 0.025),
            "occupancy_tv": tv,
            "js_distance": js,
            "agent_disagreement": disagreement,
            "shadow_memory_divergence": shadow,
            "decision_field_divergence": result["decision_field_gap"],
            "probability_tv": probability_tv,
            "trajectory_sha256": result["trajectory_sha256"],
        }
    closed = branches["PERTURBED_CLOSED"]["late_tv"]
    field = branches["PERTURBED_CONTROL_FIELD"]["late_tv"]
    off = branches["PERTURBED_SOCIAL_OFF"]["late_tv"]
    return {
        "regime_id": spec["regime_id"],
        "social_weight": spec["social_weight"],
        "choice_seed": seed,
        "target_place": plan["target_place"],
        "selected_agents": list(plan["selected_agents"]),
        "D_closed": closed,
        "D_control_field": field,
        "D_social_off": off,
        "closure_removal_effect": closed - field,
        "social_influence_removal_effect": closed - off,
        "branches": branches,
        "llm_calls": 0,
    }


def _branch_safety(spec, seed, branch, result) -> dict[str, Any]:
    diagnostics = measure_world(result["series"], moves=result["moves"], timesteps=TIMESTEPS)
    late_share = _share(_presence(result["series"], 401, 500))
    return {
        "regime_id": spec["regime_id"],
        "branch": branch,
        "choice_seed": seed,
        "movement_rate": diagnostics["movement_rate"],
        "mean_dwell": diagnostics["mean_dwell"],
        "occupancy_entropy": diagnostics["entropy_mean"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "consecutive_occupancy_tv": diagnostics["mean_total_variation"],
        "lag1": diagnostics["lag1"],
        "lag2": diagnostics["lag2"],
        "late_hhi": _hhi(late_share),
        "late_entropy": _entropy(late_share),
        "late_dominant_share": max(late_share.values()),
        "llm_calls": 0,
    }


def _summarize(metrics, worlds, replays, carriers) -> dict[str, Any]:
    regimes = {spec["regime_id"]: _regime_summary(spec, [row for row in metrics if row["regime_id"] == spec["regime_id"]], worlds) for spec in REGIMES}
    closures = {}
    for regime_id in ("RW50", "EXP99"):
        block = regimes[regime_id]
        closures[regime_id] = classify_closure(
            block["branches"]["PERTURBED_CLOSED"]["persistence_class"],
            block["D_control_field"],
            block["D_social_off"],
        )
        block["closure_type_exp010"] = closures[regime_id]
    return {
        "experiment_id": "experiment_010_feedback_closure",
        "note": ANALYSIS_NOTE,
        "closure_rule": CLOSURE_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "attenuation_rule": ATTENUATION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
        "carrier_type_exp009": carriers,
        "closure_type_exp010": closures,
        "conclusion": overall_conclusion(closures, gates_ok=True),
        "gates": {
            "control_lineage": True,
            "perturbed_lineage": True,
            "s0_equivalence": True,
            "control_field_replay": True,
            "social_off_zero": True,
            "pairs_checked": len(metrics),
        },
        "normalization": CENTERED_NORMALIZATION,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "llm_calls": 0,
        "world_count": len(worlds),
        "seed_count": len(metrics),
        "timesteps": TIMESTEPS,
        "choice_seeds": list(CHOICE_SEEDS),
        "js_definition": "Jensen-Shannon distance = sqrt(JS divergence), with natural log.",
        "regimes": regimes,
        "exact_replay": replays,
    }


def _regime_summary(spec, rows, worlds) -> dict[str, Any]:
    closed = statistics.fmean(row["D_closed"] for row in rows)
    field = statistics.fmean(row["D_control_field"] for row in rows)
    off = statistics.fmean(row["D_social_off"] for row in rows)
    return {
        "social_weight": spec["social_weight"],
        "n_seeds": len(rows),
        "llm_calls": 0,
        "D_closed": closed,
        "D_control_field": field,
        "D_social_off": off,
        "control_field_retention_fraction": retention_fraction(field, closed),
        "social_off_retention_fraction": retention_fraction(off, closed),
        "removal_contrasts": {
            "closure_removal_effect": _effect([row["closure_removal_effect"] for row in rows]),
            "social_influence_removal_effect": _effect([row["social_influence_removal_effect"] for row in rows]),
        },
        "branches": {branch: _branch_summary(rows, worlds, spec["regime_id"], branch) for branch in BRANCHES},
    }


def _branch_summary(rows, worlds, regime_id, branch) -> dict[str, Any]:
    late = [row["branches"][branch]["late_tv"] for row in rows]
    return {
        "late_tv": _dist(late),
        "regime_late_tv": statistics.fmean(late),
        "late_tv_ci95": list(_bootstrap_ci(late)),
        "persistence_class": classify_persistence(statistics.fmean(late)),
        "windows": {
            name: {
                metric: _dist([row["branches"][branch]["windows"][name][metric] for row in rows])
                for metric in ("occupancy_tv", "js_distance", "agent_disagreement", "probability_tv")
            }
            for name, _start, _stop in WINDOWS
        },
        "post_t22_choice_disagreement_rate": _dist(
            [row["branches"][branch]["post_t22_choice_disagreement_rate"] for row in rows]
        ),
        "stable_attenuation_time": _dist([row["branches"][branch]["stable_attenuation_time"] for row in rows]),
        "stable_attenuation_rate": statistics.fmean(float(row["branches"][branch]["stable_attenuation_time"] is not None) for row in rows),
        "ever_reconverged_rate": statistics.fmean(float(row["branches"][branch]["ever_reconverged"]) for row in rows),
        "time_to_reconvergence": _dist([row["branches"][branch]["time_to_reconvergence"] for row in rows]),
        "first_choice_divergence_t": _dist([row["branches"][branch]["first_choice_divergence_t"] for row in rows]),
        "first_occupancy_tv_ge_025_t": _dist([row["branches"][branch]["first_occupancy_tv_ge_025_t"] for row in rows]),
        "late_shadow_divergence": _dist(
            [statistics.fmean(row["branches"][branch]["shadow_memory_divergence"][401:501]) for row in rows]
        ),
        "late_decision_field_divergence": _dist(
            [statistics.fmean(row["branches"][branch]["decision_field_divergence"][401:501]) for row in rows]
        ),
        "safety": _safety_summary([world for world in worlds if world["regime_id"] == regime_id and world["branch"] == branch]),
    }


def _safety_summary(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    lag1 = {place_id: statistics.fmean(world["lag1"][place_id] for world in worlds) for place_id in ("site_01", "site_02")}
    late_hhi = statistics.fmean(world["late_hhi"] for world in worlds)
    dominant = statistics.fmean(world["late_dominant_share"] for world in worlds)
    saturation = statistics.fmean(world["saturation_frequency"] for world in worlds)
    alternation = statistics.fmean(world["alternation_rate"] for world in worlds)
    consecutive = statistics.fmean(world["consecutive_occupancy_tv"] for world in worlds)
    return {
        "n": len(worlds),
        "movement_rate": _dist([world["movement_rate"] for world in worlds]),
        "mean_dwell": _dist([world["mean_dwell"] for world in worlds]),
        "occupancy_entropy": _dist([world["occupancy_entropy"] for world in worlds]),
        "saturation_frequency": _dist([world["saturation_frequency"] for world in worlds]),
        "empty_frequency": _dist([world["empty_frequency"] for world in worlds]),
        "alternation_rate": _dist([world["alternation_rate"] for world in worlds]),
        "consecutive_occupancy_tv": _dist([world["consecutive_occupancy_tv"] for world in worlds]),
        "lag1_site_01": lag1["site_01"],
        "lag1_site_02": lag1["site_02"],
        "late_hhi": late_hhi,
        "late_dominant_share": dominant,
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


def _compare(metrics, summary) -> dict[str, Any]:
    by_regime = {spec["regime_id"]: [row for row in metrics if row["regime_id"] == spec["regime_id"]] for spec in REGIMES}
    return {
        "experiment_id": "experiment_010_feedback_closure",
        "note": "Paired by choice seed. Closure class uses regime-mean late TV, not a p-value. Intervals use 10000 paired resamples.",
        "conclusion": summary["conclusion"],
        "carrier_type_exp009": summary["carrier_type_exp009"],
        "closure_type_exp010": summary["closure_type_exp010"],
        "paired_contrasts": {
            "RW50_minus_S0_closed": _paired(by_regime["S0"], by_regime["RW50"], "D_closed"),
            "EXP99_minus_S0_closed": _paired(by_regime["S0"], by_regime["EXP99"], "D_closed"),
            "EXP99_minus_RW50_closed": _paired(by_regime["RW50"], by_regime["EXP99"], "D_closed"),
        },
        "mean_trajectory": {
            regime_id: {
                branch: {
                    "occupancy_tv": _column_mean(rows, branch, "occupancy_tv"),
                    "shadow_memory_divergence": _column_mean(rows, branch, "shadow_memory_divergence"),
                    "decision_field_divergence": _column_mean(rows, branch, "decision_field_divergence"),
                }
                for branch in PERTURBED_BRANCHES
            }
            for regime_id, rows in by_regime.items()
        },
    }


def _closure(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_010_feedback_closure",
        "closure_rule": CLOSURE_RULE,
        "conclusion": summary["conclusion"],
        "carrier_type_exp009": summary["carrier_type_exp009"],
        "closure_type_exp010": summary["closure_type_exp010"],
        "gates": summary["gates"],
        "regimes": {
            regime_id: {
                "D_closed": block["D_closed"],
                "D_control_field": block["D_control_field"],
                "D_social_off": block["D_social_off"],
                "control_field_retention_fraction": block["control_field_retention_fraction"],
                "social_off_retention_fraction": block["social_off_retention_fraction"],
                "removal_contrasts": block["removal_contrasts"],
                "closure_type_exp010": block.get("closure_type_exp010"),
                "persistence_class": {
                    branch: block["branches"][branch]["persistence_class"] for branch in PERTURBED_BRANCHES
                },
                "stable_attenuation_time": {
                    branch: block["branches"][branch]["stable_attenuation_time"] for branch in PERTURBED_BRANCHES
                },
                "ever_reconverged_rate": {
                    branch: block["branches"][branch]["ever_reconverged_rate"] for branch in PERTURBED_BRANCHES
                },
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


def _paired(left_rows, right_rows, metric: str) -> dict[str, Any]:
    left = {row["choice_seed"]: row for row in left_rows}
    right = {row["choice_seed"]: row for row in right_rows}
    diffs = [right[seed][metric] - left[seed][metric] for seed in sorted(left)]
    return _effect(diffs)


def _effect(diffs: list[float]) -> dict[str, float | int | None]:
    mean = statistics.fmean(diffs)
    scale = statistics.stdev(diffs) if len(diffs) > 1 else None
    low, high = _bootstrap_ci(diffs)
    return {
        "mean_difference": mean,
        "sd": scale,
        "cohens_dz": None if not scale else mean / scale,
        "ci95_low": low,
        "ci95_high": high,
        "n_pairs": len(diffs),
    }


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    sampler = random.Random(ANALYSIS_BOOTSTRAP_SEED)
    means = []
    size = len(values)
    for _draw in range(BOOTSTRAP_DRAWS):
        means.append(statistics.fmean(values[sampler.randrange(size)] for _index in range(size)))
    means.sort()
    return means[int(0.025 * (BOOTSTRAP_DRAWS - 1))], means[int(0.975 * (BOOTSTRAP_DRAWS - 1))]


def _column_mean(rows, branch: str, key: str) -> list[float]:
    width = len(rows[0]["branches"][branch][key])
    return [statistics.fmean(row["branches"][branch][key][index] for row in rows) for index in range(width)]


def _replay(atlas, initial, plans, metrics) -> dict[str, Any]:
    requested = (("S0", "CONTROL_CLOSED", 0), ("RW50", "PERTURBED_CLOSED", 0), ("EXP99", "PERTURBED_CONTROL_FIELD", 0))
    found = {}
    for regime_id, branch, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        plan = plans[(regime_id, seed)]
        field = None
        if branch == "PERTURBED_CONTROL_FIELD":
            field = simulate_branch(
                atlas, initial, spec, choice_seed_value=seed, branch="CONTROL_CLOSED", plan=plan
            )["social_field"]
        result = simulate_branch(
            atlas, initial, spec, choice_seed_value=seed, branch=branch, plan=plan, social_field=field
        )
        original = next(
            row["branches"][branch]["trajectory_sha256"]
            for row in metrics
            if row["regime_id"] == regime_id and row["choice_seed"] == seed
        )
        if result["trajectory_sha256"] != original:
            raise RuntimeError(f"{regime_id} {branch} seed {seed} replay mismatch")
        found[f"{regime_id}_{branch}_seed{seed}"] = {"trajectory_sha256": result["trajectory_sha256"], "match": True}
    return found


def _probability_tv(branch_probs, control_probs) -> list[float]:
    series = [0.0]
    for timestep in range(1, len(branch_probs)):
        agents = sorted(branch_probs[timestep])
        series.append(statistics.fmean(_prob_tv(branch_probs[timestep][agent_id], control_probs[timestep][agent_id]) for agent_id in agents))
    return series


def _prob_tv(left: dict[str, float], right: dict[str, float]) -> float:
    return 0.5 * sum(abs(left[place_id] - right[place_id]) for place_id in PLACE_IDS)


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _scores_equal(left: dict[str, float], right: dict[str, float]) -> bool:
    return all(left[place_id] == float(right[place_id]) for place_id in PLACE_IDS)


def _max_abs(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(float(left[place_id]) - float(right[place_id])) for place_id in PLACE_IDS)


def _step_payload(logged, locations, actions, memory) -> dict[str, Any]:
    return {
        "timestep": logged,
        "locations": {agent_id: locations[agent_id] for agent_id in sorted(locations)},
        "occupancy": occupancy_of(locations),
        "agents": [
            {
                "agent_id": action["agent_id"],
                "reconsidered": action["reconsidered"],
                "destination": action["destination"],
                "probabilities": action["choice_probabilities"],
                "social_utility": action["social_utility"],
                "rng_draws": action["rng_draws"],
            }
            for action in actions
        ],
        "rng_draw_count": sum(action["rng_draws"] for action in actions),
        "social_memory": memory_payload(memory),
    }


def _extend_hash(hasher, payload: dict[str, Any]) -> None:
    hasher.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    hasher.update(b"\n")


def _movement(series: list[dict[str, str]]) -> int:
    return sum(previous[agent_id] != nxt[agent_id] for previous, nxt in zip(series, series[1:]) for agent_id in previous)


def _make_memory(spec: dict[str, Any], agent_ids: list[str]) -> RollingMemory | DecayMemory:
    if spec["memory"] == "rolling":
        return RollingMemory(agent_ids, int(spec["window"]))
    return DecayMemory(agent_ids, float(spec["decay"]))


def _reference_hashes() -> dict[str, str]:
    manifest = json.loads((WORLD_V07_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = manifest["trajectory_sha256"]
    if hashes["S0:0:CONTROL"] != EXPECTED_S0_SEED0:
        raise RuntimeError("frozen S0 seed 0 hash is not the preregistered value")
    return hashes


def _exp009_carriers() -> dict[str, str]:
    summary = json.loads((WORLD_V09_DIR / "experiment_009_summary.json").read_text(encoding="utf-8"))
    return {"RW50": summary["carrier_type"]["RW50"], "EXP99": summary["carrier_type"]["EXP99"]}


def _first_true(values: list[float], predicate) -> int | None:
    for timestep, value in enumerate(values):
        if predicate(value):
            return timestep
    return None


def _first_location_divergence(series, control) -> int | None:
    for timestep, (left, right) in enumerate(zip(series, control, strict=True)):
        if any(left[agent_id] != right[agent_id] for agent_id in left):
            return timestep
    return None


def _presence(series, start: int, stop: int) -> dict[str, int]:
    counts = {place_id: 0 for place_id in PLACE_IDS}
    for step in series[start : stop + 1]:
        for place_id in step.values():
            counts[place_id] += 1
    return counts


def _share_of(locations: dict[str, str]) -> dict[str, float]:
    counts = occupancy_of(locations)
    population = len(locations)
    return {place_id: counts[place_id] / population for place_id in PLACE_IDS}


def _dist(values: list[float | int | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"mean": None, "sd": None, "median": None, "q1": None, "q3": None, "n": 0}
    quarters = statistics.quantiles(finite, n=4, method="inclusive") if len(finite) >= 2 else [finite[0], finite[0], finite[0]]
    return {
        "mean": statistics.fmean(finite),
        "sd": statistics.stdev(finite) if len(finite) > 1 else None,
        "median": statistics.median(finite),
        "q1": quarters[0],
        "q3": quarters[2],
        "n": len(finite),
    }


def _config(seeds, timesteps, carriers) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_10",
        "experiment_id": "experiment_010_feedback_closure",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "normalization": CENTERED_NORMALIZATION,
        "regimes": [dict(spec) for spec in REGIMES],
        "branches": list(BRANCHES),
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "intervention_step": INTERVENTION_STEP,
        "ablation_step": ABLATION_STEP,
        "intervention_source": "data/worlds/world_v0_7/interventions.jsonl",
        "override_count": OVERRIDE_COUNT,
        "windows": [{"name": name, "start": start, "stop": stop} for name, start, stop in WINDOWS],
        "closure_rule": CLOSURE_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "attenuation_rule": ATTENUATION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
        "carrier_type_exp009": carriers,
        "unstable_thresholds": {
            "alternation_upper_quartile": ALTERNATION_UPPER_QUARTILE,
            "consecutive_tv_upper_quartile": CONSECUTIVE_TV_UPPER_QUARTILE,
            "extreme_late_hhi": EXTREME_LATE_HHI,
            "extreme_dominant_share": EXTREME_DOMINANT_SHARE,
            "extreme_saturation": EXTREME_SATURATION,
        },
        "llm_calls": 0,
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _manifest(destination, config, summary, metrics) -> dict[str, Any]:
    trajectories = {}
    for row in metrics:
        for branch in BRANCHES:
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:{branch}"] = row["branches"][branch]["trajectory_sha256"]
    return {
        "experiment_id": "experiment_010_feedback_closure",
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "seed_count": summary["seed_count"],
        "complete": summary["world_count"] == len(REGIMES) * len(CHOICE_SEEDS) * len(BRANCHES),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "s0_seed0_sha256": EXPECTED_S0_SEED0,
        "gates": summary["gates"],
        "carrier_type_exp009": summary["carrier_type_exp009"],
        "closure_type_exp010": summary["closure_type_exp010"],
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": trajectories,
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_10_config.json",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "control_social_field.jsonl",
                "control_field_replay_audit.jsonl",
                "feedback_ablation_metrics.jsonl",
                "closure_analysis.json",
                "experiment_010_summary.json",
                "regime_comparison.json",
            )
        },
    }


def _write_initial(states, memories, spec, seed, branch, locations) -> None:
    _dump_state(states, spec, seed, branch, 0, locations)
    memory = _make_memory(spec, list(locations))
    payload = memory_payload(memory)
    _dump(
        memories,
        {
            "regime_id": spec["regime_id"],
            "branch": branch,
            "choice_seed": seed,
            "timestep": 0,
            "social_weight": spec["social_weight"],
            "architecture": payload["architecture"],
            "raw_total": payload["total"],
            "raw_own": payload["own"],
            "commit_count": 0 if payload["architecture"] == "rolling" else None,
            "decision_relevant": True,
            "llm_calls": 0,
        },
    )


def _dump_state(handle, spec, seed, branch, logged, locations) -> None:
    ordered = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    _dump(
        handle,
        {
            "regime_id": spec["regime_id"],
            "branch": branch,
            "social_weight": spec["social_weight"],
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
            ATLAS_DIR,
            CALIBRATION_DIR,
        )
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 010 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_010")
    main()
