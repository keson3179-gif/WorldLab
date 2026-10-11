"""Experiment 013: held-out feedback-autonomy onset replication.

Seeds 50-99 repeat the Experiment 012 blend on a finer grid inside (0.20, 0.40].
Lambda remains a simulation control parameter. This module does not call a model
and does not retune the grid, thresholds, or intervention after seeing results.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from contextlib import contextmanager
from pathlib import Path
from random import Random
from typing import Any, Iterator

import world.experiment_005 as memory_experiment
import world.experiment_007 as intervention_experiment
import world.experiment_012 as experiment_012
from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import OUTPUT_DIR as CALIBRATION_DIR
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_003 import OUTPUT_DIR as WORLD_V03_DIR
from world.experiment_004 import OUTPUT_DIR as WORLD_V04_DIR
from world.experiment_004 import _hhi, _js_distance, _select, _tv, trajectory_hash
from world.experiment_005 import (
    CENTERED_NORMALIZATION,
    FAMILIARITY_WEIGHT,
    OUTPUT_DIR as WORLD_V05_DIR,
    _utilities,
    centered_scores,
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
from world.experiment_007 import intervention_hash, plan_intervention
from world.experiment_008 import OUTPUT_DIR as WORLD_V08_DIR
from world.experiment_009 import OUTPUT_DIR as WORLD_V09_DIR
from world.experiment_009 import classify_persistence, memory_payload
from world.experiment_010 import OUTPUT_DIR as WORLD_V10_DIR
from world.experiment_011 import OUTPUT_DIR as WORLD_V11_DIR
from world.experiment_012 import (
    OUTPUT_DIR as WORLD_V12_DIR,
    _anchor_mad,
    _make_memory,
    _movement,
    _positive_zero,
    _probability_tv,
    _safety_summary,
    _share_of,
    _step_payload,
    _trace_mad,
    blend_scores,
    midpoint_scores,
    reconvergence_time,
    stable_attenuation_time,
)
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_13"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
HELDOUT_SEEDS = tuple(range(50, 100))
PRIOR_SEEDS = tuple(range(50))
TIMESTEPS = 500
INTERVENTION_STEP = 21
ENVIRONMENT_STEP = 22
FIELD_STEPS = tuple(range(ENVIRONMENT_STEP, TIMESTEPS + 1))
OVERRIDE_COUNT = 2
ANALYSIS_BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 10000
BLEND_TOLERANCE = 1e-12
LAMBDAS = (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 1.0)
REFINEMENT_LAMBDAS = (0.2, 0.25, 0.3, 0.35, 0.4)
ANCHOR_LAMBDAS = (0.2, 0.4, 1.0)
HISTORIES = ("CONTROL", "PERTURBED")
REGIMES = experiment_012.REGIMES
WINDOWS = experiment_012.WINDOWS
REPLAYS = (
    ("S0", "CONTROL", 1.0, 50),
    ("RW50", "PERTURBED", 0.3, 50),
    ("EXP99", "CONTROL", 0.35, 50),
    ("RW50", "CONTROL", 1.0, 99),
    ("EXP99", "PERTURBED", 0.25, 75),
    ("S0", "PERTURBED", 0.0, 99),
)
MIDPOINT_NOTE = experiment_012.MIDPOINT_NOTE
AUTONOMY_NOTE = (
    "Lambda is a simulation control parameter mixing a shared midpoint field with branch-specific "
    "endogenous social scores. It is not human autonomy, psychological independence, or individualism. "
    "Held-out seeds are a stochastic replication set relative to Experiment 012, not independent model validation, "
    "external validation, or new population validation. "
    "These runs do not establish a critical lambda, a phase transition, a critical point, a tipping point, "
    "a universal threshold, or collective path dependence."
)
COARSE_RULE = (
    "Coarse replication uses only lambda 0.20 and 0.40 on held-out seeds. "
    "COARSE_REPLICATED: D_0.20 < 0.025 and D_0.40 >= 0.025. "
    "SHIFTED_BELOW: D_0.20 >= 0.025. "
    "SHIFTED_ABOVE: D_0.40 < 0.025. "
    "GLOBAL_NONMONOTONIC: D_0.40 >= 0.025 and D_1.00 < 0.025. "
    "Replication is this classification structure, not a nonsignificant magnitude difference."
)
REFINEMENT_RULE = (
    "Refined onset is assigned only after COARSE_REPLICATED. "
    "On lambda 0.20, 0.25, 0.30, 0.35, 0.40, the onset is the first level whose regime-mean late TV is at least 0.025 "
    "when every higher refinement level through 0.40 also stays at least 0.025. "
    "The band is (previous tested lambda, onset lambda]. "
    "If persistence is followed by attenuation inside the refinement grid, the pattern is NONMONOTONIC and no onset is assigned. "
    "Lambda 1.00 does not define the refined band. No interpolation, logistic fit, or change-point search is used."
)
CONCLUSION_RULE = (
    "If both social regimes are COARSE_REPLICATED, the Experiment 012 coarse pattern replicated on held-out seeds, "
    "and each refined band is reported separately. "
    "If the refined bands differ, the held-out onset interval differs between the tested social-memory architectures. "
    "If only one regime replicates, replication depends on social-memory architecture. "
    "If neither replicates, the Experiment 012 onset interval did not replicate on the held-out stochastic seed set. "
    "The strongest description is a held-out feedback-autonomy persistence onset interval within the frozen simulation architecture."
)
BOTH_REPLICATED = (
    "the coarse feedback-autonomy persistence pattern from Experiment 012 replicated on held-out stochastic seeds in both social-memory regimes"
)
ONE_REPLICATED = "held-out replication of the Experiment 012 onset band depends on social-memory architecture"
NONE_REPLICATED = "the Experiment 012 onset interval did not replicate on the held-out stochastic seed set"
BANDS_DIFFER = "the held-out feedback-autonomy onset interval differs between the two tested social-memory architectures"
REFINEMENT_NONMONOTONIC = (
    "history persistence depends non-monotonically on feedback autonomy within the refinement grid, so no single refined onset is assigned"
)
HELDOUT_NOTE = (
    "Seeds 50-99 are a held-out stochastic replication set relative to Experiment 012. "
    "The architecture, agents, places, Atlas, and intervention rule are unchanged. "
    "This is not independent model validation, external validation, or new population validation."
)


@contextmanager
def _weight(value: float) -> Iterator[None]:
    original = memory_experiment.SOCIAL_WEIGHT
    memory_experiment.SOCIAL_WEIGHT = float(value)
    try:
        yield
    finally:
        memory_experiment.SOCIAL_WEIGHT = original


def _lambda_label(value: float) -> str:
    return f"{value:.2f}"


def coarse_status(d20: float, d40: float, d1: float) -> str:
    if d40 >= 0.025 and d1 < 0.025:
        return "GLOBAL_NONMONOTONIC"
    if d20 >= 0.025:
        return "SHIFTED_BELOW"
    if d40 < 0.025:
        return "SHIFTED_ABOVE"
    return "COARSE_REPLICATED"


def refine_onset(levels: dict[float, float]) -> dict[str, Any]:
    grid = list(REFINEMENT_LAMBDAS)
    persists = [levels[value] >= 0.025 for value in grid]
    seen = False
    for flag in persists:
        if seen and not flag:
            return {"refinement_pattern": "NONMONOTONIC", "refined_onset_lambda": None, "refined_onset_band": None}
        if flag:
            seen = True
    if not seen:
        return {"refinement_pattern": "NONMONOTONIC", "refined_onset_lambda": None, "refined_onset_band": None}
    index = persists.index(True)
    onset = grid[index]
    band = {
        "lower_exclusive": grid[index - 1],
        "upper_inclusive": onset,
        "label": f"({_lambda_label(grid[index - 1])}, {_lambda_label(onset)}]",
    }
    return {"refinement_pattern": "REFINED_GRADED_ONSET", "refined_onset_lambda": onset, "refined_onset_band": band}


def classify_regime(levels: dict[float, float]) -> dict[str, Any]:
    status = coarse_status(levels[0.2], levels[0.4], levels[1.0])
    result: dict[str, Any] = {
        "coarse_replication_status": status,
        "refinement_pattern": None,
        "refined_onset_lambda": None,
        "refined_onset_band": None,
        "refined_onset_claim": None,
    }
    if status != "COARSE_REPLICATED":
        return result
    refined = refine_onset(levels)
    result.update(refined)
    if refined["refinement_pattern"] == "REFINED_GRADED_ONSET":
        result["refined_onset_claim"] = refined["refined_onset_band"]["label"]
    return result


def overall_conclusion(classified: dict[str, dict[str, Any]]) -> str:
    replicated = [name for name in ("RW50", "EXP99") if classified[name]["coarse_replication_status"] == "COARSE_REPLICATED"]
    if not replicated:
        return NONE_REPLICATED
    if len(replicated) == 1:
        return ONE_REPLICATED
    sentence = BOTH_REPLICATED
    if any(classified[name]["refinement_pattern"] == "NONMONOTONIC" for name in ("RW50", "EXP99")):
        return sentence + ". " + REFINEMENT_NONMONOTONIC
    bands = [classified[name]["refined_onset_band"]["label"] for name in ("RW50", "EXP99")]
    detail = f"RW50 refined onset band is {bands[0]}. EXP99 refined onset band is {bands[1]}."
    if bands[0] != bands[1]:
        detail += " " + BANDS_DIFFER
    return sentence + ". " + detail


def post_t22_redivergence_onset(values: list[float]) -> int | None:
    dropped = None
    for timestep in range(ENVIRONMENT_STEP, len(values)):
        if values[timestep] < 0.025:
            dropped = timestep
            break
    if dropped is None:
        return None
    for timestep in range(dropped + 1, len(values)):
        if values[timestep] >= 0.025:
            return timestep
    return None


def audit_experiment_012_seeds() -> dict[str, Any]:
    config = json.loads((WORLD_V12_DIR / "world_v0_12_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((WORLD_V12_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    config_seeds = tuple(int(seed) for seed in config["choice_seeds"])
    metric_seeds: set[int] = set()
    with (WORLD_V12_DIR / "autonomy_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                metric_seeds.add(int(json.loads(line)["choice_seed"]))
    trajectory_seeds = {int(key.split(":")[1]) for key in manifest["trajectory_sha256"]}
    heldout_present = bool(set(range(50, 100)) & (set(config_seeds) | metric_seeds | trajectory_seeds))
    passed = config_seeds == PRIOR_SEEDS and metric_seeds == set(PRIOR_SEEDS) and trajectory_seeds == set(PRIOR_SEEDS) and not heldout_present
    return {
        "experiment_012_config_seeds": list(config_seeds),
        "experiment_012_metric_seeds": sorted(metric_seeds),
        "experiment_012_trajectory_seeds": sorted(trajectory_seeds),
        "substantive_sets_checked": ["D_lambda", "onset classification", "adjacent contrasts", "Spearman"],
        "seeds_50_to_99_in_experiment_012": heldout_present,
        "experiment_013_substantive_seeds": list(HELDOUT_SEEDS),
        "passed": passed,
        "note": HELDOUT_NOTE,
    }


def simulate_history(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    spec: dict[str, Any],
    *,
    choice_seed_value: int,
    history: str,
    lambda_value: float,
    plan: dict[str, Any] | None,
    donors: dict[str, dict[int, dict[str, dict[str, float]]]] | None,
    timesteps: int = TIMESTEPS,
    on_step: Any = None,
    require_donors: bool = True,
) -> dict[str, Any]:
    if history not in HISTORIES:
        raise RuntimeError("unknown history")
    if lambda_value not in LAMBDAS:
        raise RuntimeError("lambda is outside the fixed grid")
    if not require_donors and lambda_value != 1.0:
        raise RuntimeError("donor collection is only defined for lambda 1")
    matrix = distance_matrix()
    places = place_index()
    weight = float(spec["social_weight"])
    memory = _make_memory(spec, list(locations))
    current = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    series = [dict(current)]
    probability_trace: list[dict[str, dict[str, float]]] = [{}]
    used_trace: list[Any] = [None]
    endo_trace: list[Any] = [None]
    mid_trace: list[Any] = [None]
    rng_draw_counts = [0]
    endogenous_field: dict[int, dict[str, dict[str, float]]] = {}
    captured_plan: dict[str, Any] | None = None
    max_blend_error = 0.0
    lambda0_field_mismatch = False
    donor_mismatch = 0.0
    dynamics = hashlib.sha256()
    with _weight(weight):
        for timestep in range(timesteps):
            logged = timestep + 1
            occupancy = occupancy_of(current)
            decisions: dict[str, str] = {}
            actions = []
            step_scores: dict[str, dict[str, float]] = {}
            endogenous_scores: dict[str, dict[str, float]] = {}
            midpoint: dict[str, dict[str, float]] = {}
            step_rng = 0
            step_error = 0.0
            for agent_id in sorted(current):
                origin = current[agent_id]
                _unclipped, endogenous = centered_scores(memory.loo_raw(agent_id))
                bases, _totals, social_term = _utilities(atlas, matrix, places, agent_id, origin, occupancy, endogenous)
                endogenous_agent = {place_id: float(endogenous[place_id]) for place_id in PLACE_IDS}
                endogenous_scores[agent_id] = endogenous_agent
                used_scores = dict(endogenous_agent)
                score_source = "endogenous"
                if logged >= ENVIRONMENT_STEP and require_donors:
                    assert donors is not None
                    control_donor = donors["CONTROL"][logged][agent_id]
                    perturbed_donor = donors["PERTURBED"][logged][agent_id]
                    midpoint_agent = midpoint_scores(control_donor, perturbed_donor)
                    if any(value < -2.0 or value > 2.0 for value in midpoint_agent.values()):
                        raise RuntimeError("midpoint left the convex range")
                    midpoint[agent_id] = midpoint_agent
                    if lambda_value == 1.0:
                        used_scores = dict(endogenous_agent)
                        score_source = "endogenous"
                    else:
                        used_scores = blend_scores(midpoint_agent, endogenous_agent, lambda_value)
                        social_term = {place_id: weight * used_scores[place_id] for place_id in PLACE_IDS}
                        score_source = "shared_midpoint" if lambda_value == 0.0 else "blend"
                    for place_id in PLACE_IDS:
                        expected = (1.0 - lambda_value) * midpoint_agent[place_id] + lambda_value * endogenous_agent[place_id]
                        error = abs(used_scores[place_id] - expected)
                        step_error = max(step_error, error)
                        if error > BLEND_TOLERANCE:
                            raise RuntimeError(f"blend identity failed at t={logged} {agent_id} {place_id}")
                        if lambda_value == 1.0:
                            reference = donors[history][logged][agent_id][place_id]
                            donor_mismatch = max(donor_mismatch, abs(endogenous_agent[place_id] - float(reference)))
                social_term = {place_id: _positive_zero(social_term[place_id]) for place_id in PLACE_IDS}
                totals = {place_id: bases[place_id] + social_term[place_id] for place_id in PLACE_IDS}
                does_reconsider = agent_reconsiders(choice_seed_value, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE)
                if not does_reconsider:
                    destination = origin
                    probabilities = {place_id: 1.0 if place_id == origin else 0.0 for place_id in PLACE_IDS}
                    step_rng += 1
                else:
                    draw = Random(choice_seed(choice_seed_value, timestep, agent_id)).random()
                    destination, probabilities = _select(list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw)
                    _base_destination, _unused = _select(list(PLACE_IDS), [bases[place_id] for place_id in PLACE_IDS], draw)
                    step_rng += 2
                decisions[agent_id] = destination
                step_scores[agent_id] = used_scores
                actions.append(
                    {
                        "timestep": logged,
                        "agent_id": agent_id,
                        "origin": origin,
                        "destination": destination,
                        "stayed": destination == origin,
                        "reconsidered": does_reconsider,
                        "score_source": score_source,
                        "intervention_applied": False,
                        "social_utility": social_term,
                        "choice_probabilities": probabilities,
                        "rng_draws": 2 if does_reconsider else 1,
                        "llm_calls": 0,
                    }
                )
            if logged == INTERVENTION_STEP:
                live = plan_intervention(spec["regime_id"], choice_seed_value, occupancy, decisions)
                captured_plan = {
                    **live,
                    "occupancy": dict(occupancy),
                    "proposed_destinations": dict(decisions),
                    "override_consumes_rng": 0,
                }
                if plan is not None and (
                    live["target_place"] != plan["target_place"]
                    or list(live["selected_agents"]) != list(plan["selected_agents"])
                    or list(live["original_proposed_destinations"]) != list(plan["original_proposed_destinations"])
                ):
                    raise RuntimeError("live intervention does not match the held-out registry")
                if history == "PERTURBED":
                    chosen = set(live["selected_agents"])
                    if len(chosen) != OVERRIDE_COUNT:
                        raise RuntimeError("perturbation does not select exactly two agents")
                    for agent_id in live["selected_agents"]:
                        decisions[agent_id] = live["target_place"]
                    for action in actions:
                        if action["agent_id"] in chosen:
                            action["destination"] = live["target_place"]
                            action["stayed"] = action["destination"] == action["origin"]
                            action["intervention_applied"] = True
                    if sum(bool(action["intervention_applied"]) for action in actions) != OVERRIDE_COUNT:
                        raise RuntimeError("perturbation did not override exactly two agents")
            current = {agent_id: decisions[agent_id] for agent_id in sorted(decisions)}
            memory.commit(decisions)
            series.append(dict(current))
            probability_trace.append({action["agent_id"]: dict(action["choice_probabilities"]) for action in actions})
            ordered_agents = sorted(step_scores)
            used_trace.append(tuple(tuple(step_scores[agent_id][place_id] for place_id in PLACE_IDS) for agent_id in ordered_agents))
            endo_trace.append(tuple(tuple(endogenous_scores[agent_id][place_id] for place_id in PLACE_IDS) for agent_id in ordered_agents))
            mid_trace.append(
                None if not midpoint else tuple(tuple(midpoint[agent_id][place_id] for place_id in PLACE_IDS) for agent_id in ordered_agents)
            )
            if logged >= ENVIRONMENT_STEP:
                endogenous_field[logged] = {agent_id: dict(endogenous_scores[agent_id]) for agent_id in ordered_agents}
            rng_draw_counts.append(step_rng)
            max_blend_error = max(max_blend_error, step_error)
            if logged >= ENVIRONMENT_STEP and require_donors:
                decision_hash = canonical_sha256(step_scores)
                midpoint_hash = canonical_sha256(midpoint)
                if lambda_value == 0.0 and decision_hash != midpoint_hash:
                    lambda0_field_mismatch = True
                if on_step is not None:
                    on_step(
                        logged,
                        current,
                        memory,
                        actions,
                        {
                            "midpoint": midpoint,
                            "endogenous": endogenous_scores,
                            "used": step_scores,
                            "error": step_error,
                            "decision_hash": decision_hash,
                            "memory_hash": canonical_sha256(memory_payload(memory)),
                        },
                    )
            elif on_step is not None:
                on_step(logged, current, memory, actions, None)
            experiment_012._extend_hash(dynamics, _step_payload(logged, current, actions, memory))
    if lambda0_field_mismatch:
        raise RuntimeError("lambda 0 decision field is not the shared midpoint")
    if require_donors and lambda_value == 1.0 and donor_mismatch > BLEND_TOLERANCE:
        raise RuntimeError("lambda 1 endogenous field does not match the held-out donor")
    if captured_plan is None and timesteps >= INTERVENTION_STEP:
        raise RuntimeError("intervention step was not reached")
    return {
        "series": series,
        "moves": _movement(series),
        "probabilities": probability_trace,
        "used_trace": used_trace,
        "endo_trace": endo_trace,
        "mid_trace": mid_trace,
        "rng_draw_counts": rng_draw_counts,
        "endogenous_field": endogenous_field,
        "captured_plan": captured_plan,
        "max_blend_error": max_blend_error,
        "override_steps": [INTERVENTION_STEP] if history == "PERTURBED" and timesteps >= INTERVENTION_STEP else [],
        "override_count": OVERRIDE_COUNT if history == "PERTURBED" and timesteps >= INTERVENTION_STEP else 0,
        "canonical_dynamics_sha256": dynamics.hexdigest(),
        "trajectory_sha256": trajectory_hash(series),
        "llm_calls": 0,
    }


def build_intervention_record(atlas, locations, spec, seed: int) -> dict[str, Any]:
    result = simulate_history(
        atlas,
        locations,
        spec,
        choice_seed_value=seed,
        history="CONTROL",
        lambda_value=1.0,
        plan=None,
        donors=None,
        timesteps=INTERVENTION_STEP,
        require_donors=False,
    )
    live = result["captured_plan"]
    if live is None or live["override_consumes_rng"] != 0:
        raise RuntimeError("intervention preview did not use the frozen generator")
    if plan_intervention is not intervention_experiment.plan_intervention:
        raise RuntimeError("intervention generator is not the Experiment 007 function")
    selected = list(live["selected_agents"])
    proposed = live["proposed_destinations"]
    ordered = sorted(
        [agent_id for agent_id in sorted(proposed) if proposed[agent_id] != live["target_place"]],
        key=lambda agent_id: (intervention_hash(spec["regime_id"], seed, agent_id), agent_id),
    )
    if ordered[:OVERRIDE_COUNT] != selected or len(selected) != OVERRIDE_COUNT:
        raise RuntimeError("held-out selection does not match Experiment 007 plan_intervention")
    record = {
        "regime": spec["regime_id"],
        "choice_seed": seed,
        "intervention_timestep": INTERVENTION_STEP,
        "target_place": live["target_place"],
        "place_ranking": list(live["place_ranking"]),
        "occupancy": live["occupancy"],
        "selected_agents": selected,
        "normal_proposed_destinations": {agent_id: proposed[agent_id] for agent_id in selected},
        "original_proposed_destinations": list(live["original_proposed_destinations"]),
        "eligible_count": live["eligible_count"],
        "agent_selection": [
            {
                "agent_id": agent_id,
                "intervention_hash": intervention_hash(spec["regime_id"], seed, agent_id),
                "rank": index,
                "selected": index < OVERRIDE_COUNT,
            }
            for index, agent_id in enumerate(ordered)
        ],
        "override_count": OVERRIDE_COUNT,
        "override_consumes_rng": 0,
        "generator": "world.experiment_007.plan_intervention",
        "llm_calls": 0,
    }
    record["registry_sha256"] = canonical_sha256(record)
    return record


def build_intervention_registry(atlas, locations, seeds: tuple[int, ...] = HELDOUT_SEEDS) -> dict[tuple[str, int], dict[str, Any]]:
    registry = {}
    for spec in REGIMES:
        for seed in seeds:
            registry[(spec["regime_id"], seed)] = build_intervention_record(atlas, locations, spec, seed)
    return registry


def run_experiment(
    atlas: FrozenUtilityAtlas | None = None,
    *,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = HELDOUT_SEEDS,
    timesteps: int = TIMESTEPS,
) -> Path:
    if timesteps != TIMESTEPS or seeds != HELDOUT_SEEDS:
        raise RuntimeError("the preregistered run is seeds 50-99 and 500 timesteps")
    destination = output_dir or OUTPUT_DIR
    _refuse_frozen_directories(destination)
    free = shutil.disk_usage(destination if destination.exists() else destination.parent).free
    if free < 40_000_000_000:
        raise RuntimeError("not enough free disk for experiment 013")
    destination.mkdir(parents=True, exist_ok=True)
    source = load_frozen_atlas() if atlas is None else atlas
    _require_atlas_hash()
    audit = audit_experiment_012_seeds()
    write_json(destination / "heldout_seed_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Experiment 012 substantive seeds are not exactly 0-49; held-out run stopped")
    initial = load_initial_locations()
    print("building held-out intervention registry", flush=True)
    plans = build_intervention_registry(source, initial, seeds)
    _write_registry(destination / "heldout_interventions.jsonl", plans, seeds)
    print("replaying held-out intervention registry", flush=True)
    replayed = build_intervention_registry(source, initial, seeds)
    if [_registry_hash(plans, spec, seed) for spec in REGIMES for seed in seeds] != [
        _registry_hash(replayed, spec, seed) for spec in REGIMES for seed in seeds
    ]:
        raise RuntimeError("held-out intervention registry is not deterministic")
    reference_012 = _exp012_reference()
    config = _config(seeds, timesteps, reference_012, audit)
    write_json(destination / "world_v0_13_config.json", config)
    names = (
        "world_states.jsonl",
        "social_memory_states.jsonl",
        "agent_actions.jsonl",
        "heldout_donor_social_fields.jsonl",
        "heldout_common_midpoint_field.jsonl",
        "feedback_blend_audit.jsonl",
        "heldout_autonomy_metrics.jsonl",
    )
    paths = {name: destination / name for name in names}
    for path in paths.values():
        path.write_text("", encoding="utf-8", newline="\n")
    metrics = []
    branch_worlds = []
    with paths["world_states.jsonl"].open("a", encoding="utf-8", newline="\n") as states, paths["social_memory_states.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as memories, paths["agent_actions.jsonl"].open("a", encoding="utf-8", newline="\n") as actions, paths[
        "heldout_donor_social_fields.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as donor_handle, paths["heldout_common_midpoint_field.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as midpoint_handle, paths["feedback_blend_audit.jsonl"].open("a", encoding="utf-8", newline="\n") as audits, paths[
        "heldout_autonomy_metrics.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as rows:
        for spec in REGIMES:
            for seed in seeds:
                record, worlds = _run_seed(
                    source,
                    initial,
                    spec,
                    seed,
                    plans[(spec["regime_id"], seed)],
                    states,
                    memories,
                    actions,
                    donor_handle,
                    midpoint_handle,
                    audits,
                )
                metrics.append(record)
                branch_worlds.extend(worlds)
                _dump(rows, record)
                print(
                    f"{spec['regime_id']} seed={seed} "
                    + " ".join(f"{item['lambda_label']}={item['D_lambda']:.4f}" for item in record["lambdas"]),
                    flush=True,
                )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    replays = _replay(source, initial, plans, metrics)
    summary = _summarize(metrics, branch_worlds, replays, reference_012, audit)
    reference_012["heldout_rows"] = metrics
    write_json(destination / "experiment_013_summary.json", summary)
    write_json(destination / "regime_comparison.json", _compare(summary))
    write_json(destination / "coarse_replication_analysis.json", _coarse(summary))
    write_json(destination / "refined_onset_analysis.json", _refined(summary))
    write_json(destination / "original_vs_heldout.json", _original_comparison(summary, reference_012))
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, metrics))
    return destination


def _run_seed(atlas, initial, spec, seed, plan, states, memories, actions, donor_handle, midpoint_handle, audits):
    donors, donor_hashes = _collect_donors(atlas, initial, spec, seed, plan)
    _write_donors(donor_handle, spec, seed, donors)
    _write_midpoint(midpoint_handle, spec, seed, donors)
    runs: dict[tuple[float, str], dict[str, Any]] = {}
    worlds = []
    for lambda_value in LAMBDAS:
        for history in HISTORIES:
            result = _run_logged(
                atlas, initial, spec, seed, history, lambda_value, plan, donors, states, memories, actions, audits
            )
            if lambda_value == 1.0 and result["trajectory_sha256"] != donor_hashes[history]:
                raise RuntimeError(f"{spec['regime_id']} seed {seed} {history} lambda 1 does not match its donor trajectory")
            runs[(lambda_value, history)] = result
            worlds.append(_branch_safety(spec, seed, history, lambda_value, result))
    _require_seed_gates(spec, seed, runs)
    record = _seed_metrics(spec, seed, runs, plan)
    for result in runs.values():
        for key in ("probabilities", "series", "used_trace", "endo_trace", "mid_trace", "endogenous_field"):
            result.pop(key, None)
    return record, worlds


def _collect_donors(atlas, initial, spec, seed, plan):
    donors = {}
    hashes = {}
    for history in HISTORIES:
        result = simulate_history(
            atlas,
            initial,
            spec,
            choice_seed_value=seed,
            history=history,
            lambda_value=1.0,
            plan=plan,
            donors=None,
            require_donors=False,
        )
        donors[history] = result["endogenous_field"]
        hashes[history] = result["trajectory_sha256"]
        if result["llm_calls"] != 0:
            raise RuntimeError("donor collection called a model")
    return donors, hashes


def _run_logged(atlas, initial, spec, seed, history, lambda_value, plan, donors, states, memories, actions, audits):
    _write_initial(states, memories, spec, seed, history, lambda_value, initial)

    def write(logged, current, memory, step_actions, audit):
        _dump_state(states, spec, seed, history, lambda_value, logged, current)
        payload = memory_payload(memory)
        _dump(
            memories,
            {
                "regime_id": spec["regime_id"],
                "history": history,
                "lambda": lambda_value,
                "choice_seed": seed,
                "timestep": logged,
                "architecture": payload["architecture"],
                "commit_count": len(payload["commits"]) if payload["architecture"] == "rolling" else None,
                "decision_relevant_fraction": 1.0 if logged < ENVIRONMENT_STEP else lambda_value,
                "llm_calls": 0,
            },
        )
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
        if audit is not None:
            _dump(audits, _audit_record(spec, seed, history, lambda_value, logged, audit))

    result = simulate_history(
        atlas,
        initial,
        spec,
        choice_seed_value=seed,
        history=history,
        lambda_value=lambda_value,
        plan=plan,
        donors=donors,
        on_step=write,
    )
    if result["llm_calls"] != 0:
        raise RuntimeError("experiment 013 called a model")
    return result


def _require_seed_gates(spec, seed, runs) -> None:
    base = runs[(0.0, "CONTROL")]
    if runs[(0.0, "CONTROL")]["used_trace"] != runs[(0.0, "PERTURBED")]["used_trace"]:
        raise RuntimeError("lambda 0 histories do not share the decision field")
    for (lambda_value, history), result in runs.items():
        if result["series"][:INTERVENTION_STEP] != base["series"][:INTERVENTION_STEP]:
            raise RuntimeError(f"{spec['regime_id']} seed {seed} diverged before t=21")
        if result["rng_draw_counts"] != base["rng_draw_counts"]:
            raise RuntimeError("lambda or history changed the RNG schedule")
        if result["max_blend_error"] > BLEND_TOLERANCE:
            raise RuntimeError("blend audit tolerance failed")
        if history == "CONTROL" and (result["override_count"] != 0 or result["override_steps"] != []):
            raise RuntimeError("control history has a t=21 override")
        if history == "PERTURBED" and (result["override_count"] != OVERRIDE_COUNT or result["override_steps"] != [INTERVENTION_STEP]):
            raise RuntimeError("perturbed history does not have exactly two t=21 overrides")
        if result["series"][:ENVIRONMENT_STEP] != runs[(1.0, history)]["series"][:ENVIRONMENT_STEP]:
            raise RuntimeError("environment blending changed the pre-t22 history")
        if lambda_value == 1.0:
            for used_step, endo_step in zip(result["used_trace"], result["endo_trace"], strict=True):
                if used_step != endo_step:
                    raise RuntimeError("lambda 1 used field is not the endogenous field")
    if spec["regime_id"] == "S0":
        for history in HISTORIES:
            expected = runs[(1.0, history)]["canonical_dynamics_sha256"]
            if any(runs[(lambda_value, history)]["canonical_dynamics_sha256"] != expected for lambda_value in LAMBDAS):
                raise RuntimeError(f"S0 {history} lambda levels are not dynamically equivalent")


def _seed_metrics(spec, seed, runs, plan) -> dict[str, Any]:
    levels = []
    for lambda_value in LAMBDAS:
        contrast = _series_contrast(runs[(lambda_value, "PERTURBED")], runs[(lambda_value, "CONTROL")])
        if len(contrast["occupancy_tv"]) != TIMESTEPS + 1:
            raise RuntimeError("metric window requires the 500-step run")
        late_tv = statistics.fmean(contrast["occupancy_tv"][401:501])
        levels.append(
            {
                "lambda": lambda_value,
                "lambda_label": _lambda_label(lambda_value),
                "shared_fraction": 1.0 - lambda_value,
                "autonomous_fraction": lambda_value,
                "D_lambda": late_tv,
                "persistence_class": classify_persistence(late_tv),
                "seed_persistent": late_tv >= 0.025,
                "late_js": statistics.fmean(contrast["js_distance"][401:501]),
                "late_agent_disagreement": statistics.fmean(contrast["agent_disagreement"][401:501]),
                "late_probability_tv": statistics.fmean(contrast["probability_tv"][401:501]),
                "late_field_mad": statistics.fmean(contrast["field_mad"][401:501]),
                "late_endogenous_field_mad": statistics.fmean(contrast["endogenous_field_mad"][401:501]),
                "late_midpoint_mad": statistics.fmean(contrast["midpoint_mad"][401:501]),
                "stable_attenuation_time": stable_attenuation_time(contrast["occupancy_tv"]),
                "ever_reconverged": reconvergence_time(contrast["occupancy_tv"]) is not None,
                "time_to_reconvergence": reconvergence_time(contrast["occupancy_tv"]),
                "post_t22_redivergence_onset": post_t22_redivergence_onset(contrast["occupancy_tv"]),
                "control_trajectory_sha256": runs[(lambda_value, "CONTROL")]["trajectory_sha256"],
                "perturbed_trajectory_sha256": runs[(lambda_value, "PERTURBED")]["trajectory_sha256"],
            }
        )
    return {
        "regime_id": spec["regime_id"],
        "choice_seed": seed,
        "target_place": plan["target_place"],
        "selected_agents": list(plan["selected_agents"]),
        "override_count": OVERRIDE_COUNT,
        "lambdas": levels,
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


def _branch_safety(spec, seed, history, lambda_value, result) -> dict[str, Any]:
    diagnostics = measure_world(result["series"], moves=result["moves"], timesteps=TIMESTEPS)
    late_share = experiment_012._share(experiment_012._presence(result["series"], 401, 500))
    return {
        "regime_id": spec["regime_id"],
        "history": history,
        "lambda": lambda_value,
        "choice_seed": seed,
        "movement_rate": diagnostics["movement_rate"],
        "mean_dwell": diagnostics["mean_dwell"],
        "occupancy_entropy": diagnostics["entropy_mean"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "consecutive_occupancy_tv": diagnostics["mean_total_variation"],
        "lag1": diagnostics["lag1"],
        "late_hhi": _hhi(late_share),
        "late_dominant_share": max(late_share.values()),
        "llm_calls": 0,
    }


def _summarize(metrics, worlds, replays, reference_012, audit) -> dict[str, Any]:
    regimes = {
        spec["regime_id"]: _regime_summary(spec, [row for row in metrics if row["regime_id"] == spec["regime_id"]], worlds)
        for spec in REGIMES
    }
    classified = {}
    for regime_id in ("RW50", "EXP99"):
        block = regimes[regime_id]
        levels = {value: block["levels"][_lambda_label(value)]["D_lambda"] for value in LAMBDAS}
        classified[regime_id] = classify_regime(levels)
        block.update(classified[regime_id])
    return {
        "experiment_id": "experiment_013_heldout_autonomy_onset",
        "note": HELDOUT_NOTE + " " + AUTONOMY_NOTE,
        "midpoint_note": MIDPOINT_NOTE,
        "coarse_rule": COARSE_RULE,
        "refinement_rule": REFINEMENT_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "heldout_seed_audit": audit,
        "experiment_012_reference": reference_012["means"],
        "coarse_replication_status": {name: classified[name]["coarse_replication_status"] for name in ("RW50", "EXP99")},
        "refinement_pattern": {name: classified[name]["refinement_pattern"] for name in ("RW50", "EXP99")},
        "refined_onset_band": {
            name: None if classified[name]["refined_onset_band"] is None else classified[name]["refined_onset_band"]["label"]
            for name in ("RW50", "EXP99")
        },
        "conclusion": overall_conclusion(classified),
        "gates": {
            "heldout_seed_audit": True,
            "intervention_replay": True,
            "lambda0_shared_field": True,
            "lambda1_endogenous_field": True,
            "blend_identity": True,
            "s0_lambda_equivalence": True,
            "donor_integrity": True,
            "algorithmic_equivalence": True,
            "matched_experiment_011_lineage": False,
            "pairs_checked": len(metrics),
        },
        "llm_calls": 0,
        "world_count": len(worlds),
        "seed_count": len(HELDOUT_SEEDS),
        "timesteps": TIMESTEPS,
        "lambdas": list(LAMBDAS),
        "js_definition": "Jensen-Shannon distance = sqrt(JS divergence), with natural log.",
        "regimes": regimes,
        "exact_replay": replays,
    }


def _regime_summary(spec, rows, worlds) -> dict[str, Any]:
    levels = {}
    for lambda_value in LAMBDAS:
        label = _lambda_label(lambda_value)
        selected = [_level(row, lambda_value) for row in rows]
        late = [item["D_lambda"] for item in selected]
        distribution = _dist(late)
        mean_late = distribution["mean"]
        heterogeneous = bool(mean_late is not None and mean_late >= 0.025 and distribution["median"] is not None and distribution["median"] < 0.025)
        levels[label] = {
            "lambda": lambda_value,
            "D_lambda": mean_late,
            "persistence_class": classify_persistence(mean_late),
            "classification_basis": "regime-mean late occupancy TV",
            "seed_persistent_rate": statistics.fmean(float(item["seed_persistent"]) for item in selected),
            "heterogeneous_seed_note": "mean-based persistence classification with heterogeneous seed-level outcomes" if heterogeneous else None,
            "late_tv": distribution,
            "late_js": _dist([item["late_js"] for item in selected]),
            "late_agent_disagreement": _dist([item["late_agent_disagreement"] for item in selected]),
            "late_probability_tv": _dist([item["late_probability_tv"] for item in selected]),
            "late_field_mad": _dist([item["late_field_mad"] for item in selected]),
            "late_endogenous_field_mad": _dist([item["late_endogenous_field_mad"] for item in selected]),
            "late_midpoint_mad": _dist([item["late_midpoint_mad"] for item in selected]),
            "stable_attenuation_time": _dist([item["stable_attenuation_time"] for item in selected]),
            "stable_attenuation_rate": statistics.fmean(float(item["stable_attenuation_time"] is not None) for item in selected),
            "ever_reconverged_rate": statistics.fmean(float(item["ever_reconverged"]) for item in selected),
            "time_to_reconvergence": _dist([item["time_to_reconvergence"] for item in selected]),
            "post_t22_redivergence_onset": _dist([item["post_t22_redivergence_onset"] for item in selected]),
            "post_t22_redivergence_rate": statistics.fmean(float(item["post_t22_redivergence_onset"] is not None) for item in selected),
            "safety": {
                history: _safety_summary(
                    [
                        world
                        for world in worlds
                        if world["regime_id"] == spec["regime_id"] and world["history"] == history and world["lambda"] == lambda_value
                    ]
                )
                for history in HISTORIES
            },
        }
    refinement_pairs = tuple(zip(REFINEMENT_LAMBDAS, REFINEMENT_LAMBDAS[1:], strict=False))
    return {
        "social_weight": spec["social_weight"],
        "n_seeds": len(rows),
        "llm_calls": 0,
        "levels": levels,
        "refinement_contrasts": {
            f"D_{_lambda_label(right)}_minus_D_{_lambda_label(left)}": experiment_012._effect(
                [_level(row, right)["D_lambda"] - _level(row, left)["D_lambda"] for row in rows]
            )
            for left, right in refinement_pairs
        },
        "spearman_refinement_lambda_late_tv": experiment_012._spearman(
            [lambda_value for row in rows for lambda_value in REFINEMENT_LAMBDAS],
            [_level(row, lambda_value)["D_lambda"] for row in rows for lambda_value in REFINEMENT_LAMBDAS],
        ),
        "spearman_full_lambda_late_tv": experiment_012._spearman(
            [lambda_value for row in rows for lambda_value in LAMBDAS],
            [_level(row, lambda_value)["D_lambda"] for row in rows for lambda_value in LAMBDAS],
        ),
    }


def _level(row, lambda_value: float) -> dict[str, Any]:
    return next(item for item in row["lambdas"] if item["lambda"] == lambda_value)


def _dist(values: list[float | int | None]) -> dict[str, float | int | None]:
    payload = dict(experiment_012._dist(values))
    finite = [float(value) for value in values if value is not None]
    payload["min"] = min(finite) if finite else None
    payload["max"] = max(finite) if finite else None
    return payload


def _compare(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_013_heldout_autonomy_onset",
        "note": "Lambda is categorical. Spearman and paired contrasts do not assign onset. No critical lambda is interpolated.",
        "conclusion": summary["conclusion"],
        "coarse_replication_status": summary["coarse_replication_status"],
        "refined_onset_band": summary["refined_onset_band"],
        "regimes": {
            regime_id: {
                "D_lambda": {label: block["levels"][label]["D_lambda"] for label in block["levels"]},
                "persistence_class": {label: block["levels"][label]["persistence_class"] for label in block["levels"]},
                "seed_persistent_rate": {label: block["levels"][label]["seed_persistent_rate"] for label in block["levels"]},
                "late_field_mad": {label: block["levels"][label]["late_field_mad"]["mean"] for label in block["levels"]},
                "coarse_replication_status": block.get("coarse_replication_status"),
                "refinement_pattern": block.get("refinement_pattern"),
                "refined_onset_band": None if block.get("refined_onset_band") is None else block["refined_onset_band"]["label"],
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


def _coarse(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_013_heldout_autonomy_onset",
        "rule": COARSE_RULE,
        "coarse_replication_status": summary["coarse_replication_status"],
        "conclusion": summary["conclusion"],
        "anchors": {
            regime_id: {
                label: {
                    "D_lambda": summary["regimes"][regime_id]["levels"][label]["D_lambda"],
                    "persistence_class": summary["regimes"][regime_id]["levels"][label]["persistence_class"],
                }
                for label in ("0.20", "0.40", "1.00")
            }
            for regime_id in ("RW50", "EXP99")
        },
    }


def _refined(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_013_heldout_autonomy_onset",
        "rule": REFINEMENT_RULE,
        "interpolation": False,
        "refinement_pattern": summary["refinement_pattern"],
        "refined_onset_band": summary["refined_onset_band"],
        "conclusion": summary["conclusion"],
        "regimes": {
            regime_id: {
                "coarse_replication_status": block.get("coarse_replication_status"),
                "refinement_pattern": block.get("refinement_pattern"),
                "refined_onset_lambda": block.get("refined_onset_lambda"),
                "refined_onset_band": block.get("refined_onset_band"),
                "refined_onset_claim": block.get("refined_onset_claim"),
                "D_lambda": {label: level["D_lambda"] for label, level in block["levels"].items()},
                "persistence_class": {label: level["persistence_class"] for label, level in block["levels"].items()},
                "seed_persistent_rate": {label: level["seed_persistent_rate"] for label, level in block["levels"].items()},
                "heterogeneous_seed_note": {label: level["heterogeneous_seed_note"] for label, level in block["levels"].items()},
                "refinement_contrasts": block["refinement_contrasts"],
            }
            for regime_id, block in summary["regimes"].items()
            if regime_id in ("RW50", "EXP99")
        },
    }


def _original_comparison(summary, reference_012) -> dict[str, Any]:
    payload = {
        "experiment_id": "experiment_013_heldout_autonomy_onset",
        "note": "Original seeds 0-49 and held-out seeds 50-99 are not paired. Replication is not decided by a nonsignificant difference.",
        "method": "independent bootstrap of the difference between resampled means",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "primary": "classification structure",
        "regimes": {},
    }
    for regime_id in ("RW50", "EXP99"):
        cells = {}
        for lambda_value in ANCHOR_LAMBDAS:
            label = _lambda_label(lambda_value)
            original = reference_012["seed_values"][regime_id][label]
            heldout = [
                _level(row, lambda_value)["D_lambda"]
                for row in reference_012["heldout_rows"]
                if row["regime_id"] == regime_id
            ]
            if len(original) != len(HELDOUT_SEEDS) or len(heldout) != len(HELDOUT_SEEDS):
                raise RuntimeError("original and held-out comparisons require 50 seeds each")
            effect = _independent_effect(original, heldout)
            effect["original_mean"] = statistics.fmean(original)
            effect["heldout_mean"] = statistics.fmean(heldout)
            cells[label] = effect
        payload["regimes"][regime_id] = cells
    return payload


def _independent_effect(original: list[float], heldout: list[float]) -> dict[str, float | int | None]:
    sampler = Random(ANALYSIS_BOOTSTRAP_SEED)
    differences = []
    original_size = len(original)
    heldout_size = len(heldout)
    for _draw in range(BOOTSTRAP_DRAWS):
        original_mean = statistics.fmean(original[sampler.randrange(original_size)] for _index in range(original_size))
        heldout_mean = statistics.fmean(heldout[sampler.randrange(heldout_size)] for _index in range(heldout_size))
        differences.append(heldout_mean - original_mean)
    differences.sort()
    original_mean = statistics.fmean(original)
    heldout_mean = statistics.fmean(heldout)
    original_sd = statistics.stdev(original) if len(original) > 1 else 0.0
    heldout_sd = statistics.stdev(heldout) if len(heldout) > 1 else 0.0
    pooled_denominator = len(original) + len(heldout) - 2
    pooled = (
        (((len(original) - 1) * original_sd**2) + ((len(heldout) - 1) * heldout_sd**2)) / pooled_denominator
    ) ** 0.5
    return {
        "mean_difference": heldout_mean - original_mean,
        "ci95_low": differences[int(0.025 * (BOOTSTRAP_DRAWS - 1))],
        "ci95_high": differences[int(0.975 * (BOOTSTRAP_DRAWS - 1))],
        "standardized_independent_group_effect": None if pooled == 0.0 else (heldout_mean - original_mean) / pooled,
        "n_original": len(original),
        "n_heldout": len(heldout),
        "paired": False,
    }


def _exp012_reference() -> dict[str, Any]:
    summary = json.loads((WORLD_V12_DIR / "experiment_012_summary.json").read_text(encoding="utf-8"))
    means = {}
    seed_values: dict[str, dict[str, list[float]]] = {}
    for regime_id in ("RW50", "EXP99"):
        means[regime_id] = {
            label: summary["regimes"][regime_id]["levels"][label]["D_lambda"] for label in ("0.20", "0.40", "1.00")
        }
        seed_values[regime_id] = {label: [] for label in ("0.20", "0.40", "1.00")}
    with (WORLD_V12_DIR / "autonomy_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["regime_id"] not in seed_values:
                continue
            for item in row["lambdas"]:
                label = item["lambda_label"]
                if label in seed_values[row["regime_id"]]:
                    seed_values[row["regime_id"]][label].append(item["D_lambda"])
    return {"means": means, "seed_values": seed_values, "heldout_rows": []}


def _replay(atlas, initial, plans, metrics) -> dict[str, Any]:
    found = {}
    for regime_id, history, lambda_value, seed in REPLAYS:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        plan = plans[(regime_id, seed)]
        donors, _hashes = _collect_donors(atlas, initial, spec, seed, plan)
        result = simulate_history(
            atlas,
            initial,
            spec,
            choice_seed_value=seed,
            history=history,
            lambda_value=lambda_value,
            plan=plan,
            donors=donors,
        )
        key = "control_trajectory_sha256" if history == "CONTROL" else "perturbed_trajectory_sha256"
        original = _level(next(row for row in metrics if row["regime_id"] == regime_id and row["choice_seed"] == seed), lambda_value)[key]
        if result["trajectory_sha256"] != original:
            raise RuntimeError(f"{regime_id} {history} lambda {lambda_value} seed {seed} replay mismatch")
        found[f"{regime_id}_{history}_lambda_{_lambda_label(lambda_value)}_seed{seed}"] = {
            "trajectory_sha256": result["trajectory_sha256"],
            "match": True,
        }
    return found


def _config(seeds, timesteps, reference_012, audit) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_13",
        "experiment_id": "experiment_013_heldout_autonomy_onset",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "normalization": CENTERED_NORMALIZATION,
        "regimes": [dict(spec) for spec in REGIMES],
        "histories": list(HISTORIES),
        "lambdas": list(LAMBDAS),
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "intervention_step": INTERVENTION_STEP,
        "environment_step": ENVIRONMENT_STEP,
        "intervention_generator": "world.experiment_007.plan_intervention",
        "override_count": OVERRIDE_COUNT,
        "blend_tolerance": BLEND_TOLERANCE,
        "midpoint_note": MIDPOINT_NOTE,
        "coarse_rule": COARSE_RULE,
        "refinement_rule": REFINEMENT_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "heldout_note": HELDOUT_NOTE,
        "experiment_012_reference": reference_012["means"],
        "heldout_seed_audit_passed": audit["passed"],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "windows": [{"name": name, "start": start, "stop": stop} for name, start, stop in WINDOWS],
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
        for level in row["lambdas"]:
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:CONTROL:{level['lambda_label']}"] = level["control_trajectory_sha256"]
            trajectories[f"{row['regime_id']}:{row['choice_seed']}:PERTURBED:{level['lambda_label']}"] = level["perturbed_trajectory_sha256"]
    return {
        "experiment_id": "experiment_013_heldout_autonomy_onset",
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "seed_count": summary["seed_count"],
        "complete": summary["world_count"] == len(REGIMES) * len(HELDOUT_SEEDS) * len(LAMBDAS) * len(HISTORIES),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "gates": summary["gates"],
        "coarse_replication_status": summary["coarse_replication_status"],
        "refined_onset_band": summary["refined_onset_band"],
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": trajectories,
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_13_config.json",
                "heldout_seed_audit.json",
                "heldout_interventions.jsonl",
                "heldout_donor_social_fields.jsonl",
                "heldout_common_midpoint_field.jsonl",
                "feedback_blend_audit.jsonl",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "heldout_autonomy_metrics.jsonl",
                "coarse_replication_analysis.json",
                "refined_onset_analysis.json",
                "original_vs_heldout.json",
                "regime_comparison.json",
                "experiment_013_summary.json",
            )
        },
    }


def _write_registry(path: Path, plans, seeds) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for spec in REGIMES:
            for seed in seeds:
                _dump(handle, plans[(spec["regime_id"], seed)])


def _registry_hash(plans, spec, seed: int) -> str:
    return plans[(spec["regime_id"], seed)]["registry_sha256"]


def _write_donors(handle, spec, seed, donors) -> None:
    for history, donor_name in (("CONTROL", "CONTROL_ENDOGENOUS"), ("PERTURBED", "PERTURBED_ENDOGENOUS")):
        for timestep in FIELD_STEPS:
            for agent_id in sorted(donors[history][timestep]):
                scores = donors[history][timestep][agent_id]
                if any(value < -2.0 or value > 2.0 for value in scores.values()):
                    raise RuntimeError("donor score is outside the clipped range")
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
    for timestep in FIELD_STEPS:
        control = donors["CONTROL"][timestep]
        perturbed = donors["PERTURBED"][timestep]
        control_hash = canonical_sha256(control)
        perturbed_hash = canonical_sha256(perturbed)
        for agent_id in sorted(control):
            scores = midpoint_scores(control[agent_id], perturbed[agent_id])
            _dump(
                handle,
                {
                    "regime": spec["regime_id"],
                    "choice_seed": seed,
                    "timestep": timestep,
                    "agent_id": agent_id,
                    "scores": scores,
                    "control_donor_hash": control_hash,
                    "perturbed_donor_hash": perturbed_hash,
                    "renormalized": False,
                    "recentered": False,
                    "reclipped": False,
                    "llm_calls": 0,
                },
            )


def _audit_record(spec, seed, history, lambda_value, logged, audit) -> dict[str, Any]:
    agents = sorted(audit["used"])
    errors = []
    midpoint_rows = []
    endogenous_rows = []
    expected_rows = []
    used_rows = []
    for agent_id in agents:
        midpoint_row = []
        endogenous_row = []
        expected_row = []
        used_row = []
        error_row = []
        for place_id in PLACE_IDS:
            midpoint_score = audit["midpoint"][agent_id][place_id]
            endogenous_score = audit["endogenous"][agent_id][place_id]
            expected = (1.0 - lambda_value) * midpoint_score + lambda_value * endogenous_score
            used = audit["used"][agent_id][place_id]
            midpoint_row.append(midpoint_score)
            endogenous_row.append(endogenous_score)
            expected_row.append(expected)
            used_row.append(used)
            error_row.append(abs(used - expected))
        midpoint_rows.append(midpoint_row)
        endogenous_rows.append(endogenous_row)
        expected_rows.append(expected_row)
        used_rows.append(used_row)
        errors.append(error_row)
    return {
        "regime": spec["regime_id"],
        "choice_seed": seed,
        "lambda": lambda_value,
        "history": history,
        "timestep": logged,
        "agent_ids": agents,
        "place_ids": list(PLACE_IDS),
        "midpoint_score": midpoint_rows,
        "endogenous_score": endogenous_rows,
        "expected_blended_score": expected_rows,
        "actual_used_score": used_rows,
        "absolute_error": errors,
        "max_absolute_error": max(max(row) for row in errors),
        "checked_agent_place_count": len(agents) * len(PLACE_IDS),
        "shadow_memory_hash": audit["memory_hash"],
        "decision_field_hash": audit["decision_hash"],
        "llm_calls": 0,
    }


def _write_initial(states, memories, spec, seed, history, lambda_value, locations) -> None:
    _dump_state(states, spec, seed, history, lambda_value, 0, locations)
    memory = _make_memory(spec, list(locations))
    payload = memory_payload(memory)
    _dump(
        memories,
        {
            "regime_id": spec["regime_id"],
            "history": history,
            "lambda": lambda_value,
            "choice_seed": seed,
            "timestep": 0,
            "architecture": payload["architecture"],
            "commit_count": 0 if payload["architecture"] == "rolling" else None,
            "decision_relevant_fraction": 1.0,
            "llm_calls": 0,
        },
    )


def _dump_state(handle, spec, seed, history, lambda_value, logged, locations) -> None:
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
            ATLAS_DIR,
            CALIBRATION_DIR,
        )
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 013 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_013")
    main()
