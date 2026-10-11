"""Experiment 011: symmetric post-perturbation social environments.

Each seed has a control history and a perturbed history. From t=22 both
histories enter the same environment: endogenous closure, a shared control
donor field, a shared perturbed donor field, or social-off. This module does
not call a model and does not retune rules after seeing results.
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
    memory_payload,
)
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_11"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
WORLD_V10_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_10"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
EXPECTED_S0_SEED0 = "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
EXPECTED_RW50_PERTURBED_SEED0 = "57a31820b38472a41a15a79dd49af2bbd8012cab13e672bf6190c8953e2a5939"
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
INTERVENTION_STEP = 21
ENVIRONMENT_STEP = 22
OVERRIDE_COUNT = 2
ATTENUATION_RUN = 20
RECONVERGENCE_RUN = 20
ANALYSIS_BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 10000
BRANCHES = (
    "CONTROL_ENDOGENOUS",
    "PERTURBED_ENDOGENOUS",
    "CONTROL_COMMON_CONTROL_FIELD",
    "PERTURBED_COMMON_CONTROL_FIELD",
    "CONTROL_COMMON_PERTURBED_FIELD",
    "PERTURBED_COMMON_PERTURBED_FIELD",
    "CONTROL_SOCIAL_OFF",
    "PERTURBED_SOCIAL_OFF",
)
ENVIRONMENTS = (
    "ENDOGENOUS_CLOSED",
    "COMMON_CONTROL_FIELD",
    "COMMON_PERTURBED_FIELD",
    "SOCIAL_OFF",
)
ENVIRONMENT_PAIR = {
    "ENDOGENOUS_CLOSED": ("CONTROL_ENDOGENOUS", "PERTURBED_ENDOGENOUS"),
    "COMMON_CONTROL_FIELD": ("CONTROL_COMMON_CONTROL_FIELD", "PERTURBED_COMMON_CONTROL_FIELD"),
    "COMMON_PERTURBED_FIELD": ("CONTROL_COMMON_PERTURBED_FIELD", "PERTURBED_COMMON_PERTURBED_FIELD"),
    "SOCIAL_OFF": ("CONTROL_SOCIAL_OFF", "PERTURBED_SOCIAL_OFF"),
}
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
ANALYSIS_NOTE = (
    "Experiment 010 established the preregistered asymmetric closure contrast. "
    "Experiment 011 separately identifies early-history persistence under symmetric post-t22 environments. "
    "Matched comparisons stay inside one environment. They do not establish collective path dependence, "
    "real-world social synchronization, human conformity, tourist validity, emergence, or self-organization."
)
OFF_RULE = (
    "D_social_off is the mean matched occupancy TV of PERTURBED_SOCIAL_OFF versus CONTROL_SOCIAL_OFF "
    "during t=401-500. OFF-A: below 0.025. OFF-B: at least 0.025 and below 0.10. OFF-C: at least 0.10. "
    "The Experiment 010 contrast of PERTURBED_SOCIAL_OFF versus CONTROL_ENDOGENOUS is diagnostic only."
)
SYNC_RULE = (
    "C means D_common_control is at least 0.025. P means D_common_perturbed is at least 0.025. "
    "S1: C false and P false. S2: C false and P true. S3: C true and P false. S4: C true and P true."
)
CONCLUSION_RULE = (
    "If symmetric social-off late TV is below 0.025: when post-t22 social influence is removed symmetrically "
    "from both histories, the early perturbation attenuates under the tested architecture. "
    "If it is at least 0.025: early history remains consequential even under a symmetric post-t22 social-off environment. "
    "If both common fields attenuate: shared social fields synchronize alternative histories independently of donor history "
    "under the tested architecture. If only one attenuates: synchronization depends on donor-field content. "
    "If both persist: a shared social field alone is insufficient to erase history-dependent divergence."
)
ATTENUATION_RULE = (
    "Stable attenuation is the first completion of 20 consecutive timesteps with history-pair occupancy TV below 0.025, "
    "counted only from t=22. The recorded time is the timestep that completes that run. Otherwise null."
)
RECONVERGENCE_RULE = (
    "Reconvergence is 20 consecutive timesteps with history-pair occupancy TV exactly 0, counted only from t=21. "
    "It does not replace the persistence classification."
)
RELATIONSHIP = (
    "Experiment 010 established the preregistered asymmetric closure contrast. "
    "Experiment 011 separately identifies early-history persistence under symmetric post-t22 environments."
)
ASYMMETRIC_REINTERPRETATION = (
    "the amplified SOCIAL_OFF contrast in Experiment 010 reflected the combined effect of early-history perturbation "
    "and asymmetric post-t22 social treatment, rather than isolated history persistence under social-off dynamics"
)
OFF_SENTENCE = {
    "OFF-A": "the early perturbation attenuates when both histories evolve under the same post-t22 social-off environment",
    "OFF-B": "the early perturbation leaves a persistent residual even when both histories evolve without ongoing social influence",
    "OFF-C": "the early perturbation produces amplified long-run divergence under a symmetric social-off environment",
}
SYNC_SENTENCE = {
    "S1": "a shared nonzero social field synchronizes alternative histories regardless of whether that field is generated by the control or perturbed donor trajectory",
    "S2": "synchronization depends on the specific content of the control social field rather than common-field exposure alone",
    "S3": "synchronization depends on the specific content of the perturbed social field rather than common-field exposure alone",
    "S4": "sharing the same social field is insufficient to erase history-dependent divergence under the tested architecture",
}
SECTION35_OFF = {
    "OFF-A": "when post-t22 social influence is removed symmetrically from both histories, the early perturbation attenuates under the tested architecture",
    "OFF-B": "early history remains consequential even under a symmetric post-t22 social-off environment",
    "OFF-C": "early history remains consequential even under a symmetric post-t22 social-off environment",
}
SECTION35_SYNC = {
    "S1": "shared social fields synchronize alternative histories independently of donor history under the tested architecture",
    "S2": "synchronization depends on donor-field content",
    "S3": "synchronization depends on donor-field content",
    "S4": "a shared social field alone is insufficient to erase history-dependent divergence",
}


@contextmanager
def _weight(value: float) -> Iterator[None]:
    original = memory_experiment.SOCIAL_WEIGHT
    memory_experiment.SOCIAL_WEIGHT = float(value)
    try:
        yield
    finally:
        memory_experiment.SOCIAL_WEIGHT = original


def history_of(branch: str) -> str:
    if branch.startswith("CONTROL_"):
        return "CONTROL"
    if branch.startswith("PERTURBED_"):
        return "PERTURBED"
    raise RuntimeError(f"unknown branch {branch}")


def environment_of(branch: str) -> str:
    if branch.endswith("_ENDOGENOUS"):
        return "endogenous"
    if branch.endswith("_COMMON_CONTROL_FIELD"):
        return "common_control"
    if branch.endswith("_COMMON_PERTURBED_FIELD"):
        return "common_perturbed"
    if branch.endswith("_SOCIAL_OFF"):
        return "social_off"
    raise RuntimeError(f"unknown branch {branch}")


def classify_off(late_tv: float) -> str:
    if late_tv < 0.025:
        return "OFF-A"
    if late_tv < 0.10:
        return "OFF-B"
    return "OFF-C"


def classify_sync(common_control_tv: float, common_perturbed_tv: float) -> str:
    control_persists = common_control_tv >= 0.025
    perturbed_persists = common_perturbed_tv >= 0.025
    if not control_persists and not perturbed_persists:
        return "S1"
    if not control_persists and perturbed_persists:
        return "S2"
    if control_persists and not perturbed_persists:
        return "S3"
    return "S4"


def overall_conclusion(off_types: dict[str, str], sync_types: dict[str, str]) -> str:
    clauses = []
    same = off_types["RW50"] == off_types["EXP99"] and sync_types["RW50"] == sync_types["EXP99"]
    if same:
        clauses.append(OFF_SENTENCE[off_types["RW50"]])
        clauses.append(SECTION35_OFF[off_types["RW50"]])
        clauses.append(SYNC_SENTENCE[sync_types["RW50"]])
        clauses.append(SECTION35_SYNC[sync_types["RW50"]])
    else:
        for regime_id in ("RW50", "EXP99"):
            clauses.append(
                f"{regime_id}: {OFF_SENTENCE[off_types[regime_id]]} {SECTION35_OFF[off_types[regime_id]]} "
                f"{SYNC_SENTENCE[sync_types[regime_id]]} {SECTION35_SYNC[sync_types[regime_id]]}"
            )
    clauses.append(RELATIONSHIP)
    attenuated = [regime_id for regime_id in ("RW50", "EXP99") if off_types[regime_id] == "OFF-A"]
    if len(attenuated) == 2:
        clauses.append(ASYMMETRIC_REINTERPRETATION)
    elif len(attenuated) == 1:
        clauses.append(f"{attenuated[0]}: {ASYMMETRIC_REINTERPRETATION}")
    return " ".join(clauses)


def stable_attenuation_time(values: list[float]) -> int | None:
    run = 0
    for timestep in range(ENVIRONMENT_STEP, len(values)):
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
    donor_field: dict[int, dict[str, dict[str, float]]] | None = None,
    on_step: Any = None,
) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise RuntimeError("unknown experiment 011 branch")
    environment = environment_of(branch)
    history = history_of(branch)
    if environment in {"common_control", "common_perturbed"} and donor_field is None:
        raise RuntimeError("common-field replay requires a materialized donor field")
    matrix = distance_matrix()
    places = place_index()
    weight = float(spec["social_weight"])
    memory = _make_memory(spec, list(locations))
    current = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    series = [dict(current)]
    collected: dict[int, dict[str, dict[str, float]]] = {}
    probability_trace: list[dict[str, dict[str, float]]] = [{}]
    field_hashes: dict[int, str] = {}
    shadow_score_hashes: dict[int, str] = {}
    shadow_memory_hashes: dict[int, str] = {}
    rng_draw_counts = [0]
    social_utility_absmax = 0.0
    shadow_scores_differed = False
    dynamics = hashlib.sha256()
    with _weight(weight):
        for timestep in range(timesteps):
            logged = timestep + 1
            occupancy = occupancy_of(current)
            decisions: dict[str, str] = {}
            actions = []
            step_scores: dict[str, dict[str, float]] = {}
            endogenous_scores: dict[str, dict[str, float]] = {}
            step_rng = 0
            for agent_id in sorted(current):
                origin = current[agent_id]
                _unclipped, endogenous = centered_scores(memory.loo_raw(agent_id))
                bases, _totals, social_term = _utilities(
                    atlas, matrix, places, agent_id, origin, occupancy, endogenous
                )
                used_scores = {place_id: float(endogenous[place_id]) for place_id in PLACE_IDS}
                endogenous_scores[agent_id] = dict(used_scores)
                score_source = "endogenous"
                if environment in {"common_control", "common_perturbed"} and logged >= ENVIRONMENT_STEP:
                    donor_scores = donor_field[logged][agent_id]
                    used_scores = {place_id: float(donor_scores[place_id]) for place_id in PLACE_IDS}
                    social_term = {place_id: weight * used_scores[place_id] for place_id in PLACE_IDS}
                    score_source = "donor_field"
                    if not _scores_equal(used_scores, donor_scores):
                        raise RuntimeError("donor field was recomputed from recipient state")
                elif environment == "social_off" and logged >= ENVIRONMENT_STEP:
                    social_term = {place_id: 0.0 for place_id in PLACE_IDS}
                    score_source = "social_off"
                social_term = {place_id: _positive_zero(social_term[place_id]) for place_id in PLACE_IDS}
                totals = {place_id: bases[place_id] + social_term[place_id] for place_id in PLACE_IDS}
                if environment == "social_off" and logged >= ENVIRONMENT_STEP:
                    if any(value != 0.0 for value in social_term.values()):
                        raise RuntimeError("symmetric social-off utility was not exactly zero")
                does_reconsider = agent_reconsiders(
                    choice_seed_value, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE
                )
                ranking_changed = _argmax(totals) != _argmax(bases)
                if not does_reconsider:
                    destination = origin
                    probabilities = {place_id: 1.0 if place_id == origin else 0.0 for place_id in PLACE_IDS}
                    destination_changed = False
                    step_rng += 1
                else:
                    draw = random.Random(choice_seed(choice_seed_value, timestep, agent_id)).random()
                    destination, probabilities = _select(
                        list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw
                    )
                    _base_destination, _base_probabilities = _select(
                        list(PLACE_IDS), [bases[place_id] for place_id in PLACE_IDS], draw
                    )
                    destination_changed = destination != _base_destination
                    step_rng += 2
                decisions[agent_id] = destination
                step_scores[agent_id] = used_scores
                if logged >= ENVIRONMENT_STEP:
                    social_utility_absmax = max(social_utility_absmax, max(abs(value) for value in social_term.values()))
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
                        "history": history,
                        "environment": environment,
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
            if logged == INTERVENTION_STEP and history == "PERTURBED":
                chosen = set(plan["selected_agents"])
                if len(chosen) != OVERRIDE_COUNT:
                    raise RuntimeError("perturbation does not select exactly two agents")
                for agent_id in plan["selected_agents"]:
                    decisions[agent_id] = plan["target_place"]
                for action in actions:
                    if action["agent_id"] in chosen:
                        action["destination"] = plan["target_place"]
                        action["stayed"] = action["destination"] == action["origin"]
                        action["intervention_applied"] = True
                if sum(bool(action["intervention_applied"]) for action in actions) != OVERRIDE_COUNT:
                    raise RuntimeError("perturbation did not override exactly two agents")
            elif logged == INTERVENTION_STEP and any(action["intervention_applied"] for action in actions):
                raise RuntimeError("control history received an override")
            current = {agent_id: decisions[agent_id] for agent_id in sorted(decisions)}
            memory.commit(decisions)
            series.append(dict(current))
            probability_trace.append({action["agent_id"]: dict(action["choice_probabilities"]) for action in actions})
            rng_draw_counts.append(step_rng)
            if logged >= ENVIRONMENT_STEP:
                decision_hash = canonical_sha256(step_scores)
                shadow_hash = canonical_sha256(endogenous_scores)
                field_hashes[logged] = decision_hash
                shadow_score_hashes[logged] = shadow_hash
                shadow_memory_hashes[logged] = canonical_sha256(memory_payload(memory))
                if shadow_hash != decision_hash:
                    shadow_scores_differed = True
                if environment == "endogenous":
                    collected[logged] = step_scores
                elif environment in {"common_control", "common_perturbed"} and decision_hash != canonical_sha256(donor_field[logged]):
                    raise RuntimeError(f"{branch} donor mismatch at t={logged}")
            _extend_hash(dynamics, _step_payload(logged, current, actions, memory))
            if on_step is not None:
                on_step(logged, current, memory, actions, collected if environment == "endogenous" else None)
    return {
        "series": series,
        "moves": _movement(series),
        "social_field": collected,
        "probabilities": probability_trace,
        "field_hashes": field_hashes,
        "shadow_score_hashes": shadow_score_hashes,
        "shadow_memory_hashes": shadow_memory_hashes,
        "rng_draw_counts": rng_draw_counts,
        "social_utility_absmax": social_utility_absmax,
        "shadow_scores_differed": shadow_scores_differed,
        "override_steps": [INTERVENTION_STEP] if history == "PERTURBED" else [],
        "override_count": OVERRIDE_COUNT if history == "PERTURBED" else 0,
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
    closure_010 = _exp010_closure()
    config = _config(seeds, timesteps, closure_010)
    write_json(destination / "world_v0_11_config.json", config)
    names = (
        "world_states.jsonl",
        "social_memory_states.jsonl",
        "agent_actions.jsonl",
        "donor_social_fields.jsonl",
        "symmetric_environment_audit.jsonl",
        "matched_history_metrics.jsonl",
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
    ) as actions, paths["donor_social_fields.jsonl"].open("a", encoding="utf-8", newline="\n") as donors, paths[
        "symmetric_environment_audit.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as audits, paths["matched_history_metrics.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as rows:
        for spec in REGIMES:
            for seed in seeds:
                runs = _run_seed(
                    source,
                    initial,
                    spec,
                    seed,
                    plans[(spec["regime_id"], seed)],
                    references,
                    states,
                    memories,
                    actions,
                    donors,
                    audits,
                )
                record = _seed_metrics(spec, seed, runs, plans[(spec["regime_id"], seed)])
                metrics.append(record)
                for branch, result in runs.items():
                    branch_worlds.append(_branch_safety(spec, seed, branch, result))
                    result.pop("probabilities", None)
                    result.pop("social_field", None)
                    result.pop("series", None)
                _dump(rows, record)
                print(
                    f"{spec['regime_id']} seed={seed} endogenous={record['D_endogenous']:.4f} "
                    f"common_control={record['D_common_control']:.4f} "
                    f"common_perturbed={record['D_common_perturbed']:.4f} "
                    f"social_off={record['D_social_off']:.4f}",
                    flush=True,
                )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    replays = _replay(source, initial, plans, metrics)
    summary = _summarize(metrics, branch_worlds, replays, closure_010)
    write_json(destination / "experiment_011_summary.json", summary)
    write_json(destination / "regime_comparison.json", _compare(metrics, summary))
    write_json(destination / "symmetric_social_off_analysis.json", _social_off_analysis(summary))
    write_json(destination / "common_field_synchronization.json", _synchronization(summary))
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, metrics))
    return destination


def _run_seed(atlas, initial, spec, seed, plan, references, states, memories, actions, donors, audits) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    control = _run_logged(
        atlas, initial, spec, seed, "CONTROL_ENDOGENOUS", plan, None, states, memories, actions, donors
    )
    if control["trajectory_sha256"] != references[f"{spec['regime_id']}:{seed}:CONTROL"]:
        raise RuntimeError(f"{spec['regime_id']} CONTROL_ENDOGENOUS seed {seed} does not match Experiment 007")
    perturbed = _run_logged(
        atlas, initial, spec, seed, "PERTURBED_ENDOGENOUS", plan, None, states, memories, actions, donors
    )
    if perturbed["trajectory_sha256"] != references[f"{spec['regime_id']}:{seed}:PERTURBED"]:
        raise RuntimeError(f"{spec['regime_id']} PERTURBED_ENDOGENOUS seed {seed} does not match Experiment 007")
    runs["CONTROL_ENDOGENOUS"] = control
    runs["PERTURBED_ENDOGENOUS"] = perturbed
    donors_by_environment = {
        "common_control": control["social_field"],
        "common_perturbed": perturbed["social_field"],
        "social_off": None,
    }
    for branch in BRANCHES:
        if environment_of(branch) == "endogenous":
            continue
        runs[branch] = _run_logged(
            atlas,
            initial,
            spec,
            seed,
            branch,
            plan,
            donors_by_environment[environment_of(branch)],
            states,
            memories,
            actions,
            None,
        )
    _require_seed_gates(spec, seed, runs)
    _write_audits(audits, spec, seed, runs)
    return runs


def _run_logged(atlas, initial, spec, seed, branch, plan, donor, states, memories, actions, donors) -> dict[str, Any]:
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
                "decision_relevant": environment_of(branch) == "endogenous" or logged < ENVIRONMENT_STEP,
                "llm_calls": 0,
            },
        )
        for action in step_actions:
            _dump(actions, {"regime_id": spec["regime_id"], "choice_seed": seed, **action})
        if donors is not None and collected is not None and logged >= ENVIRONMENT_STEP:
            for agent_id in sorted(collected[logged]):
                _dump(
                    donors,
                    {
                        "regime": spec["regime_id"],
                        "choice_seed": seed,
                        "donor": branch,
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
        donor_field=donor,
        on_step=write,
    )
    if result["llm_calls"] != 0:
        raise RuntimeError("experiment 011 called a model")
    return result


def _require_seed_gates(spec, seed, runs) -> None:
    prefix = runs["CONTROL_ENDOGENOUS"]["series"][:INTERVENTION_STEP]
    for branch, result in runs.items():
        if result["series"][:INTERVENTION_STEP] != prefix:
            raise RuntimeError(f"{spec['regime_id']} seed {seed} {branch} diverged before t=21")
        if result["rng_draw_counts"] != runs["CONTROL_ENDOGENOUS"]["rng_draw_counts"]:
            raise RuntimeError(f"{branch} did not keep the matched RNG schedule")
    for environment, (control_name, perturbed_name) in ENVIRONMENT_PAIR.items():
        control = runs[control_name]
        perturbed = runs[perturbed_name]
        if control["override_count"] != 0 or control["override_steps"] != []:
            raise RuntimeError("control history has a t=21 override")
        if perturbed["override_count"] != OVERRIDE_COUNT or perturbed["override_steps"] != [INTERVENTION_STEP]:
            raise RuntimeError("perturbed history does not have exactly two t=21 overrides")
        if control["series"][:ENVIRONMENT_STEP] != runs["CONTROL_ENDOGENOUS"]["series"][:ENVIRONMENT_STEP]:
            raise RuntimeError(f"{control_name} changed history before the environment")
        if perturbed["series"][:ENVIRONMENT_STEP] != runs["PERTURBED_ENDOGENOUS"]["series"][:ENVIRONMENT_STEP]:
            raise RuntimeError(f"{perturbed_name} changed history before the environment")
        if environment == "SOCIAL_OFF":
            if control["social_utility_absmax"] != 0.0 or perturbed["social_utility_absmax"] != 0.0:
                raise RuntimeError("symmetric social-off utility was not zero")
        if environment in {"COMMON_CONTROL_FIELD", "COMMON_PERTURBED_FIELD"}:
            donor_name = "CONTROL_ENDOGENOUS" if environment == "COMMON_CONTROL_FIELD" else "PERTURBED_ENDOGENOUS"
            donor_hashes = runs[donor_name]["field_hashes"]
            if control["field_hashes"] != perturbed["field_hashes"] or control["field_hashes"] != donor_hashes:
                raise RuntimeError(f"{environment} recipients do not share the donor field")
    if spec["regime_id"] == "S0":
        for history in ("CONTROL", "PERTURBED"):
            names = [branch for branch in BRANCHES if history_of(branch) == history]
            expected = runs[names[0]]["canonical_dynamics_sha256"]
            if any(runs[branch]["canonical_dynamics_sha256"] != expected for branch in names):
                raise RuntimeError(f"S0 {history} environments are not dynamically equivalent")


def _write_audits(handle, spec, seed, runs) -> None:
    for environment, donor_name in (
        ("COMMON_CONTROL_FIELD", "CONTROL_ENDOGENOUS"),
        ("COMMON_PERTURBED_FIELD", "PERTURBED_ENDOGENOUS"),
    ):
        control_name, perturbed_name = ENVIRONMENT_PAIR[environment]
        control = runs[control_name]
        perturbed = runs[perturbed_name]
        donor = runs[donor_name]
        for timestep in sorted(donor["field_hashes"]):
            control_hash = control["field_hashes"][timestep]
            perturbed_hash = perturbed["field_hashes"][timestep]
            donor_hash = donor["field_hashes"][timestep]
            _dump(
                handle,
                {
                    "regime": spec["regime_id"],
                    "choice_seed": seed,
                    "timestep": timestep,
                    "environment": environment,
                    "donor": donor_name,
                    "control_decision_field_hash": control_hash,
                    "perturbed_decision_field_hash": perturbed_hash,
                    "donor_field_hash": donor_hash,
                    "recipients_match": control_hash == perturbed_hash,
                    "donor_match": control_hash == donor_hash and perturbed_hash == donor_hash,
                    "control_shadow_memory_hash": control["shadow_memory_hashes"][timestep],
                    "perturbed_shadow_memory_hash": perturbed["shadow_memory_hashes"][timestep],
                    "control_shadow_score_hash": control["shadow_score_hashes"][timestep],
                    "perturbed_shadow_score_hash": perturbed["shadow_score_hashes"][timestep],
                    "shadow_entered_decision": control_hash != donor_hash or perturbed_hash != donor_hash,
                    "llm_calls": 0,
                },
            )
    control_off = runs["CONTROL_SOCIAL_OFF"]
    perturbed_off = runs["PERTURBED_SOCIAL_OFF"]
    endogenous_draws = runs["CONTROL_ENDOGENOUS"]["rng_draw_counts"]
    for timestep in sorted(control_off["field_hashes"]):
        _dump(
            handle,
            {
                "regime": spec["regime_id"],
                "choice_seed": seed,
                "timestep": timestep,
                "environment": "SOCIAL_OFF",
                "control_effective_social_weight": 0.0,
                "perturbed_effective_social_weight": 0.0,
                "control_social_utility_zero": control_off["social_utility_absmax"] == 0.0,
                "perturbed_social_utility_zero": perturbed_off["social_utility_absmax"] == 0.0,
                "control_rng_draws": control_off["rng_draw_counts"][timestep],
                "perturbed_rng_draws": perturbed_off["rng_draw_counts"][timestep],
                "rng_draws_match_endogenous_control": (
                    control_off["rng_draw_counts"][timestep] == endogenous_draws[timestep]
                    and perturbed_off["rng_draw_counts"][timestep] == endogenous_draws[timestep]
                ),
                "structural_parameters_match": True,
                "llm_calls": 0,
            },
        )


def _seed_metrics(spec, seed, runs, plan) -> dict[str, Any]:
    environments = {}
    for environment, (control_name, perturbed_name) in ENVIRONMENT_PAIR.items():
        contrast = _contrast(runs[perturbed_name], runs[control_name])
        late_tv = statistics.fmean(contrast["occupancy_tv"][401:501])
        environments[environment] = {
            **contrast,
            "late_tv": late_tv,
            "persistence_class": classify_persistence(late_tv),
            "windows": _windows(contrast),
            "post_t22_choice_disagreement_rate": statistics.fmean(contrast["agent_disagreement"][ENVIRONMENT_STEP : TIMESTEPS + 1]),
            "stable_attenuation_time": stable_attenuation_time(contrast["occupancy_tv"]),
            "ever_reconverged": reconvergence_time(contrast["occupancy_tv"]) is not None,
            "time_to_reconvergence": reconvergence_time(contrast["occupancy_tv"]),
            "control_trajectory_sha256": runs[control_name]["trajectory_sha256"],
            "perturbed_trajectory_sha256": runs[perturbed_name]["trajectory_sha256"],
        }
    entrainment = {
        "COMMON_CONTROL_FIELD": _entrainment(runs, "CONTROL_ENDOGENOUS", "CONTROL_COMMON_CONTROL_FIELD", "PERTURBED_COMMON_CONTROL_FIELD"),
        "COMMON_PERTURBED_FIELD": _entrainment(runs, "PERTURBED_ENDOGENOUS", "CONTROL_COMMON_PERTURBED_FIELD", "PERTURBED_COMMON_PERTURBED_FIELD"),
    }
    asymmetric = _contrast(runs["PERTURBED_SOCIAL_OFF"], runs["CONTROL_ENDOGENOUS"])
    asymmetric_late = statistics.fmean(asymmetric["occupancy_tv"][401:501])
    endogenous = environments["ENDOGENOUS_CLOSED"]["late_tv"]
    common_control = environments["COMMON_CONTROL_FIELD"]["late_tv"]
    common_perturbed = environments["COMMON_PERTURBED_FIELD"]["late_tv"]
    social_off = environments["SOCIAL_OFF"]["late_tv"]
    return {
        "regime_id": spec["regime_id"],
        "social_weight": spec["social_weight"],
        "choice_seed": seed,
        "target_place": plan["target_place"],
        "selected_agents": list(plan["selected_agents"]),
        "D_endogenous": endogenous,
        "D_common_control": common_control,
        "D_common_perturbed": common_perturbed,
        "D_social_off": social_off,
        "asymmetric_social_off_contrast_exp010": asymmetric_late,
        "symmetric_social_off_contrast_exp011": social_off,
        "endogenous_minus_common_control": endogenous - common_control,
        "endogenous_minus_common_perturbed": endogenous - common_perturbed,
        "endogenous_minus_social_off": endogenous - social_off,
        "common_control_minus_common_perturbed": common_control - common_perturbed,
        "environments": environments,
        "entrainment": entrainment,
        "asymmetric_occupancy_tv": asymmetric["occupancy_tv"],
        "llm_calls": 0,
    }


def _contrast(perturbed, control) -> dict[str, Any]:
    return _series_contrast(perturbed["series"], control["series"], perturbed["probabilities"], control["probabilities"])


def _series_contrast(left_series, right_series, left_probs, right_probs) -> dict[str, list[float]]:
    occupancy_tv, js_distance, disagreement = [], [], []
    agents = sorted(left_series[0])
    population = len(agents)
    for timestep in range(len(left_series)):
        occupancy_tv.append(_tv(_share_of(left_series[timestep]), _share_of(right_series[timestep])))
        js_distance.append(_js_distance(_share_of(left_series[timestep]), _share_of(right_series[timestep])))
        disagreement.append(
            sum(left_series[timestep][agent_id] != right_series[timestep][agent_id] for agent_id in agents) / population
        )
    return {
        "occupancy_tv": occupancy_tv,
        "js_distance": js_distance,
        "agent_disagreement": disagreement,
        "probability_tv": _probability_tv(left_probs, right_probs),
    }


def _entrainment(runs, donor_name, control_name, perturbed_name) -> dict[str, Any]:
    donor = runs[donor_name]
    control = _series_contrast(runs[control_name]["series"], donor["series"], runs[control_name]["probabilities"], donor["probabilities"])
    perturbed = _series_contrast(runs[perturbed_name]["series"], donor["series"], runs[perturbed_name]["probabilities"], donor["probabilities"])
    return {
        "history_pair_tv": _series_contrast(
            runs[perturbed_name]["series"], runs[control_name]["series"], runs[perturbed_name]["probabilities"], runs[control_name]["probabilities"]
        )["occupancy_tv"],
        "control_recipient_to_donor_tv": control["occupancy_tv"],
        "perturbed_recipient_to_donor_tv": perturbed["occupancy_tv"],
        "control_recipient_to_donor_disagreement": control["agent_disagreement"],
        "perturbed_recipient_to_donor_disagreement": perturbed["agent_disagreement"],
        "control_exact_donor_replay": runs[control_name]["trajectory_sha256"] == donor["trajectory_sha256"],
        "perturbed_exact_donor_replay": runs[perturbed_name]["trajectory_sha256"] == donor["trajectory_sha256"],
        "late_control_to_donor_tv": statistics.fmean(control["occupancy_tv"][401:501]),
        "late_perturbed_to_donor_tv": statistics.fmean(perturbed["occupancy_tv"][401:501]),
    }


def _windows(contrast) -> dict[str, dict[str, float]]:
    return {
        name: {
            "occupancy_tv": statistics.fmean(contrast["occupancy_tv"][start : stop + 1]),
            "js_distance": statistics.fmean(contrast["js_distance"][start : stop + 1]),
            "agent_disagreement": statistics.fmean(contrast["agent_disagreement"][start : stop + 1]),
            "probability_tv": statistics.fmean(contrast["probability_tv"][start : stop + 1]),
        }
        for name, start, stop in WINDOWS
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


def _summarize(metrics, worlds, replays, closure_010) -> dict[str, Any]:
    regimes = {
        spec["regime_id"]: _regime_summary(spec, [row for row in metrics if row["regime_id"] == spec["regime_id"]], worlds)
        for spec in REGIMES
    }
    off_types = {}
    sync_types = {}
    for regime_id in ("RW50", "EXP99"):
        block = regimes[regime_id]
        off_types[regime_id] = classify_off(block["D_social_off"])
        sync_types[regime_id] = classify_sync(block["D_common_control"], block["D_common_perturbed"])
        block["off_class_exp011"] = off_types[regime_id]
        block["synchronization_type_exp011"] = sync_types[regime_id]
        block["off_conclusion"] = OFF_SENTENCE[off_types[regime_id]]
        block["synchronization_conclusion"] = SYNC_SENTENCE[sync_types[regime_id]]
    return {
        "experiment_id": "experiment_011_symmetric_social_environment",
        "note": ANALYSIS_NOTE,
        "off_rule": OFF_RULE,
        "synchronization_rule": SYNC_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "attenuation_rule": ATTENUATION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
        "closure_type_exp010": closure_010,
        "off_class_exp011": off_types,
        "synchronization_type_exp011": sync_types,
        "conclusion": overall_conclusion(off_types, sync_types),
        "gates": {
            "control_lineage": True,
            "perturbed_lineage": True,
            "s0_equivalence": True,
            "common_control_field": True,
            "common_perturbed_field": True,
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
    values = {
        key: statistics.fmean(row[key] for row in rows)
        for key in ("D_endogenous", "D_common_control", "D_common_perturbed", "D_social_off", "asymmetric_social_off_contrast_exp010")
    }
    return {
        "social_weight": spec["social_weight"],
        "n_seeds": len(rows),
        "llm_calls": 0,
        **values,
        "symmetric_social_off_contrast_exp011": values["D_social_off"],
        "persistence_class": {
            "ENDOGENOUS_CLOSED": classify_persistence(values["D_endogenous"]),
            "COMMON_CONTROL_FIELD": classify_persistence(values["D_common_control"]),
            "COMMON_PERTURBED_FIELD": classify_persistence(values["D_common_perturbed"]),
            "SOCIAL_OFF": classify_persistence(values["D_social_off"]),
        },
        "paired_contrasts": {
            "endogenous_minus_common_control": _effect([row["endogenous_minus_common_control"] for row in rows]),
            "endogenous_minus_common_perturbed": _effect([row["endogenous_minus_common_perturbed"] for row in rows]),
            "endogenous_minus_social_off": _effect([row["endogenous_minus_social_off"] for row in rows]),
            "common_control_minus_common_perturbed": _effect([row["common_control_minus_common_perturbed"] for row in rows]),
            "asymmetric_minus_symmetric_social_off": _effect(
                [row["asymmetric_social_off_contrast_exp010"] - row["D_social_off"] for row in rows]
            ),
        },
        "environments": {name: _environment_summary(rows, name) for name in ENVIRONMENTS},
        "entrainment": {name: _entrainment_summary(rows, name) for name in ("COMMON_CONTROL_FIELD", "COMMON_PERTURBED_FIELD")},
        "safety": {branch: _safety_summary([world for world in worlds if world["regime_id"] == spec["regime_id"] and world["branch"] == branch]) for branch in BRANCHES},
    }


def _environment_summary(rows, environment: str) -> dict[str, Any]:
    late = [row["environments"][environment]["late_tv"] for row in rows]
    return {
        "late_tv": _dist(late),
        "regime_late_tv": statistics.fmean(late),
        "persistence_class": classify_persistence(statistics.fmean(late)),
        "windows": {
            name: {
                metric: _dist([row["environments"][environment]["windows"][name][metric] for row in rows])
                for metric in ("occupancy_tv", "js_distance", "agent_disagreement", "probability_tv")
            }
            for name, _start, _stop in WINDOWS
        },
        "post_t22_choice_disagreement_rate": _dist(
            [row["environments"][environment]["post_t22_choice_disagreement_rate"] for row in rows]
        ),
        "stable_attenuation_time": _dist([row["environments"][environment]["stable_attenuation_time"] for row in rows]),
        "stable_attenuation_rate": statistics.fmean(
            float(row["environments"][environment]["stable_attenuation_time"] is not None) for row in rows
        ),
        "ever_reconverged_rate": statistics.fmean(float(row["environments"][environment]["ever_reconverged"]) for row in rows),
        "time_to_reconvergence": _dist([row["environments"][environment]["time_to_reconvergence"] for row in rows]),
    }


def _entrainment_summary(rows, environment: str) -> dict[str, Any]:
    return {
        "late_control_to_donor_tv": _dist([row["entrainment"][environment]["late_control_to_donor_tv"] for row in rows]),
        "late_perturbed_to_donor_tv": _dist([row["entrainment"][environment]["late_perturbed_to_donor_tv"] for row in rows]),
        "control_exact_donor_replay_rate": statistics.fmean(
            float(row["entrainment"][environment]["control_exact_donor_replay"]) for row in rows
        ),
        "perturbed_exact_donor_replay_rate": statistics.fmean(
            float(row["entrainment"][environment]["perturbed_exact_donor_replay"]) for row in rows
        ),
        "windows": {
            name: {
                "control_recipient_to_donor_tv": statistics.fmean(
                    statistics.fmean(row["entrainment"][environment]["control_recipient_to_donor_tv"][start : stop + 1]) for row in rows
                ),
                "perturbed_recipient_to_donor_tv": statistics.fmean(
                    statistics.fmean(row["entrainment"][environment]["perturbed_recipient_to_donor_tv"][start : stop + 1]) for row in rows
                ),
                "history_pair_tv": statistics.fmean(
                    statistics.fmean(row["entrainment"][environment]["history_pair_tv"][start : stop + 1]) for row in rows
                ),
            }
            for name, start, stop in WINDOWS
        },
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
        "experiment_id": "experiment_011_symmetric_social_environment",
        "note": "Paired by choice seed inside matched environments. Classification uses regime-mean late TV, not a p-value.",
        "conclusion": summary["conclusion"],
        "closure_type_exp010": summary["closure_type_exp010"],
        "off_class_exp011": summary["off_class_exp011"],
        "synchronization_type_exp011": summary["synchronization_type_exp011"],
        "paired_contrasts": {
            regime_id: summary["regimes"][regime_id]["paired_contrasts"] for regime_id in ("RW50", "EXP99")
        },
        "cross_regime": {
            "EXP99_minus_RW50_endogenous": _paired(by_regime["RW50"], by_regime["EXP99"], "D_endogenous"),
            "EXP99_minus_RW50_social_off": _paired(by_regime["RW50"], by_regime["EXP99"], "D_social_off"),
        },
    }


def _social_off_analysis(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_011_symmetric_social_environment",
        "definition": "symmetric_social_off_contrast_exp011 compares PERTURBED_SOCIAL_OFF with CONTROL_SOCIAL_OFF. asymmetric_social_off_contrast_exp010 compares PERTURBED_SOCIAL_OFF with CONTROL_ENDOGENOUS and does not determine classification.",
        "closure_type_exp010": summary["closure_type_exp010"],
        "off_class_exp011": summary["off_class_exp011"],
        "conclusion": summary["conclusion"],
        "regimes": {
            regime_id: {
                "D_social_off": block["D_social_off"],
                "asymmetric_social_off_contrast_exp010": block["asymmetric_social_off_contrast_exp010"],
                "symmetric_social_off_contrast_exp011": block["symmetric_social_off_contrast_exp011"],
                "off_class_exp011": block.get("off_class_exp011"),
                "off_conclusion": block.get("off_conclusion"),
                "persistence_class": block["persistence_class"]["SOCIAL_OFF"],
                "paired_difference": block["paired_contrasts"]["asymmetric_minus_symmetric_social_off"],
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


def _synchronization(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_011_symmetric_social_environment",
        "synchronization_rule": SYNC_RULE,
        "synchronization_type_exp011": summary["synchronization_type_exp011"],
        "conclusion": summary["conclusion"],
        "regimes": {
            regime_id: {
                "D_common_control": block["D_common_control"],
                "D_common_perturbed": block["D_common_perturbed"],
                "D_endogenous": block["D_endogenous"],
                "synchronization_type_exp011": block.get("synchronization_type_exp011"),
                "synchronization_conclusion": block.get("synchronization_conclusion"),
                "persistence_class": {
                    "COMMON_CONTROL_FIELD": block["persistence_class"]["COMMON_CONTROL_FIELD"],
                    "COMMON_PERTURBED_FIELD": block["persistence_class"]["COMMON_PERTURBED_FIELD"],
                },
                "entrainment": block["entrainment"],
                "stable_attenuation_time": {
                    name: block["environments"][name]["stable_attenuation_time"]
                    for name in ("COMMON_CONTROL_FIELD", "COMMON_PERTURBED_FIELD")
                },
                "ever_reconverged_rate": {
                    name: block["environments"][name]["ever_reconverged_rate"]
                    for name in ("COMMON_CONTROL_FIELD", "COMMON_PERTURBED_FIELD", "ENDOGENOUS_CLOSED", "SOCIAL_OFF")
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
        "sd": 0.0 if scale == 0 else scale,
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


def _replay(atlas, initial, plans, metrics) -> dict[str, Any]:
    requested = (
        ("S0", "CONTROL_ENDOGENOUS", None),
        ("RW50", "PERTURBED_ENDOGENOUS", None),
        ("EXP99", "PERTURBED_COMMON_CONTROL_FIELD", "CONTROL_ENDOGENOUS"),
    )
    found = {}
    for regime_id, branch, donor_branch in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        plan = plans[(regime_id, 0)]
        donor = None
        if donor_branch is not None:
            donor = simulate_branch(
                atlas, initial, spec, choice_seed_value=0, branch=donor_branch, plan=plan
            )["social_field"]
        result = simulate_branch(
            atlas, initial, spec, choice_seed_value=0, branch=branch, plan=plan, donor_field=donor
        )
        original = next(
            row["environments"][_environment_name(branch)]["perturbed_trajectory_sha256" if history_of(branch) == "PERTURBED" else "control_trajectory_sha256"]
            for row in metrics
            if row["regime_id"] == regime_id and row["choice_seed"] == 0
        )
        if result["trajectory_sha256"] != original:
            raise RuntimeError(f"{regime_id} {branch} seed 0 replay mismatch")
        found[f"{regime_id}_{branch}_seed0"] = {"trajectory_sha256": result["trajectory_sha256"], "match": True}
    if found["S0_CONTROL_ENDOGENOUS_seed0"]["trajectory_sha256"] != EXPECTED_S0_SEED0:
        raise RuntimeError("S0 control replay is not the frozen hash")
    if found["RW50_PERTURBED_ENDOGENOUS_seed0"]["trajectory_sha256"] != EXPECTED_RW50_PERTURBED_SEED0:
        raise RuntimeError("RW50 perturbed replay is not the frozen hash")
    return found


def _environment_name(branch: str) -> str:
    return {
        "endogenous": "ENDOGENOUS_CLOSED",
        "common_control": "COMMON_CONTROL_FIELD",
        "common_perturbed": "COMMON_PERTURBED_FIELD",
        "social_off": "SOCIAL_OFF",
    }[environment_of(branch)]


def _probability_tv(branch_probs, control_probs) -> list[float]:
    series = [0.0]
    for timestep in range(1, len(branch_probs)):
        agents = sorted(branch_probs[timestep])
        series.append(
            statistics.fmean(_prob_tv(branch_probs[timestep][agent_id], control_probs[timestep][agent_id]) for agent_id in agents)
        )
    return series


def _prob_tv(left: dict[str, float], right: dict[str, float]) -> float:
    return 0.5 * sum(abs(left[place_id] - right[place_id]) for place_id in PLACE_IDS)


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _scores_equal(left: dict[str, float], right: dict[str, float]) -> bool:
    return all(left[place_id] == float(right[place_id]) for place_id in PLACE_IDS)


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
    if hashes["RW50:0:PERTURBED"] != EXPECTED_RW50_PERTURBED_SEED0:
        raise RuntimeError("frozen RW50 perturbed seed 0 hash is not the preregistered value")
    return hashes


def _exp010_closure() -> dict[str, str]:
    summary = json.loads((WORLD_V10_DIR / "experiment_010_summary.json").read_text(encoding="utf-8"))
    return {"RW50": summary["closure_type_exp010"]["RW50"], "EXP99": summary["closure_type_exp010"]["EXP99"]}


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
        "sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "median": statistics.median(finite),
        "q1": quarters[0],
        "q3": quarters[2],
        "n": len(finite),
    }


def _config(seeds, timesteps, closure_010) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_11",
        "experiment_id": "experiment_011_symmetric_social_environment",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "normalization": CENTERED_NORMALIZATION,
        "regimes": [dict(spec) for spec in REGIMES],
        "branches": list(BRANCHES),
        "environments": list(ENVIRONMENTS),
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "intervention_step": INTERVENTION_STEP,
        "environment_step": ENVIRONMENT_STEP,
        "intervention_source": "data/worlds/world_v0_7/interventions.jsonl",
        "override_count": OVERRIDE_COUNT,
        "windows": [{"name": name, "start": start, "stop": stop} for name, start, stop in WINDOWS],
        "off_rule": OFF_RULE,
        "synchronization_rule": SYNC_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "attenuation_rule": ATTENUATION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
        "closure_type_exp010": closure_010,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
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
        for environment, (control_name, perturbed_name) in ENVIRONMENT_PAIR.items():
            block = row["environments"][environment]
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:{control_name}"] = block["control_trajectory_sha256"]
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:{perturbed_name}"] = block["perturbed_trajectory_sha256"]
    return {
        "experiment_id": "experiment_011_symmetric_social_environment",
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "seed_count": summary["seed_count"],
        "complete": summary["world_count"] == len(REGIMES) * len(CHOICE_SEEDS) * len(BRANCHES),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "s0_seed0_sha256": EXPECTED_S0_SEED0,
        "gates": summary["gates"],
        "closure_type_exp010": summary["closure_type_exp010"],
        "off_class_exp011": summary["off_class_exp011"],
        "synchronization_type_exp011": summary["synchronization_type_exp011"],
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": trajectories,
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_11_config.json",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "donor_social_fields.jsonl",
                "symmetric_environment_audit.jsonl",
                "matched_history_metrics.jsonl",
                "symmetric_social_off_analysis.json",
                "common_field_synchronization.json",
                "experiment_011_summary.json",
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
            WORLD_V10_DIR,
            ATLAS_DIR,
            CALIBRATION_DIR,
        )
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 011 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_011")
    main()
