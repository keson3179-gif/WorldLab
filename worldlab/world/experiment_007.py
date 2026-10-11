"""Experiment 007: one controlled early perturbation, then free evolution.

S0 is the zero-social control. RW50 and EXP99 use the Experiment 006 reference weight.
This module does not call a model and does not retune the intervention after seeing results.
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
    _unique_leader,
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
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_7"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
EXPECTED_S0_SEED0 = "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
INTERVENTION_STEP = 21
OVERRIDE_COUNT = 2
INITIAL_TV = 0.10
RECONVERGENCE_RUN = 20
ANALYSIS_BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 10000
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
    "One two-agent override at the t=20 to t=21 transition, then unforced evolution. "
    "Classes describe perturbation persistence. They do not establish collective path dependence, "
    "lock-in, self-organization, emergence, a butterfly effect, human realism, or behavioral validity."
)
PERSISTENCE_RULE = (
    "Regime class uses the mean occupancy TV over t=401–500 across the 50 matched pairs. "
    "attenuated: late TV < 0.025. "
    "persistent residual: 0.025 <= late TV < 0.10. "
    "amplified divergence: late TV >= 0.10. "
    "persistence_ratio = late TV / 0.10. "
    "Pair counts are descriptive. The regime label is not chosen by a p-value."
)
CONCLUSION_RULE = (
    "If every regime is attenuated: small early perturbations are not persistently retained under the tested regimes. "
    "If any regime is amplified: early perturbation divergence can amplify under the tested simulation regime. "
    "Otherwise: early perturbations leave persistent collective residuals."
)
RECONVERGENCE_RULE = (
    "Stable occupancy reconvergence requires 20 consecutive timesteps with occupancy TV exactly 0, "
    "counted only from t=21 onward. time_to_reconvergence is the timestep that completes that run. "
    "Agent-level exact reconvergence uses the same rule on disagreement and is an audit only."
)
HASH_MATERIAL = "experiment_007{regime_id}{choice_seed}{agent_id}"


@contextmanager
def _weight(value: float) -> Iterator[None]:
    original = memory_experiment.SOCIAL_WEIGHT
    memory_experiment.SOCIAL_WEIGHT = float(value)
    try:
        yield
    finally:
        memory_experiment.SOCIAL_WEIGHT = original


def intervention_hash(regime_id: str, choice_seed_value: int, agent_id: str) -> str:
    material = f"experiment_007{regime_id}{choice_seed_value}{agent_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def ranked_places(occupancy: dict[str, int]) -> list[str]:
    return sorted(PLACE_IDS, key=lambda place_id: (-int(occupancy[place_id]), place_id))


def plan_intervention(
    regime_id: str,
    choice_seed_value: int,
    occupancy: dict[str, int],
    proposed: dict[str, str],
) -> dict[str, Any]:
    ranking = ranked_places(occupancy)
    target = ranking[1]
    candidates = [agent_id for agent_id in sorted(proposed) if proposed[agent_id] != target]
    ordered = sorted(candidates, key=lambda agent_id: (intervention_hash(regime_id, choice_seed_value, agent_id), agent_id))
    if len(ordered) < OVERRIDE_COUNT:
        raise RuntimeError(
            f"{regime_id} seed {choice_seed_value} has {len(ordered)} agents outside {target}"
        )
    chosen = ordered[:OVERRIDE_COUNT]
    return {
        "target_place": target,
        "place_ranking": ranking,
        "selected_agents": chosen,
        "original_proposed_destinations": [proposed[agent_id] for agent_id in chosen],
        "eligible_count": len(ordered),
    }


def classify_persistence(late_tv: float) -> str:
    if late_tv < 0.025:
        return "attenuated"
    if late_tv < INITIAL_TV:
        return "persistent residual"
    return "amplified divergence"


def overall_conclusion(regime_classes: dict[str, str]) -> str:
    labels = [regime_classes[item["regime_id"]] for item in REGIMES]
    if all(label == "attenuated" for label in labels):
        return "small early perturbations are not persistently retained under the tested regimes"
    if any(label == "amplified divergence" for label in labels):
        return "early perturbation divergence can amplify under the tested simulation regime"
    return "early perturbations leave persistent collective residuals"


def simulate_branch(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    spec: dict[str, Any],
    *,
    choice_seed_value: int,
    apply_intervention: bool,
    timesteps: int = TIMESTEPS,
    on_step: Any = None,
) -> dict[str, Any]:
    matrix = distance_matrix()
    places = place_index()
    memory = _make_memory(spec, list(locations))
    current = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    series = [dict(current)]
    moves = 0
    intervention: dict[str, Any] | None = None
    override_steps: list[int] = []
    with _weight(float(spec["social_weight"])):
        for timestep in range(timesteps):
            logged = timestep + 1
            occupancy = occupancy_of(current)
            decisions: dict[str, str] = {}
            actions = []
            for agent_id in sorted(current):
                origin = current[agent_id]
                _unclipped, scores = centered_scores(memory.loo_raw(agent_id))
                bases, totals, social_term = _utilities(
                    atlas, matrix, places, agent_id, origin, occupancy, scores
                )
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
                    destination, probabilities = _select(list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw)
                    _base_destination, _base_probabilities = _select(
                        list(PLACE_IDS), [bases[place_id] for place_id in PLACE_IDS], draw
                    )
                    destination_changed = destination != _base_destination
                decisions[agent_id] = destination
                actions.append(
                    {
                        "timestep": logged,
                        "agent_id": agent_id,
                        "origin": origin,
                        "destination": destination,
                        "proposed_destination": destination,
                        "stayed": destination == origin,
                        "reconsidered": does_reconsider,
                        "social_weight": float(spec["social_weight"]),
                        "familiarity_weight": FAMILIARITY_WEIGHT,
                        "normalization": CENTERED_NORMALIZATION,
                        "intervention_applied": False,
                        "base_utility": bases,
                        "popularity_score": scores,
                        "social_utility": social_term,
                        "total_utility": totals,
                        "choice_probabilities": probabilities,
                        "occupancy_before": occupancy,
                        "ranking_changed": ranking_changed,
                        "destination_changed": destination_changed,
                        "llm_calls": 0,
                    }
                )
            if logged == INTERVENTION_STEP:
                intervention = plan_intervention(spec["regime_id"], choice_seed_value, occupancy, decisions)
                intervention["applied"] = apply_intervention
                if apply_intervention:
                    for agent_id in intervention["selected_agents"]:
                        decisions[agent_id] = intervention["target_place"]
                    chosen = set(intervention["selected_agents"])
                    for action in actions:
                        if action["agent_id"] in chosen:
                            action["destination"] = intervention["target_place"]
                            action["stayed"] = action["destination"] == action["origin"]
                            action["intervention_applied"] = True
                    intervention["effective_overrides"] = sum(action["intervention_applied"] for action in actions)
                    override_steps.append(logged)
                    if intervention["effective_overrides"] != OVERRIDE_COUNT:
                        raise RuntimeError("intervention did not override exactly two agents")
                else:
                    intervention["effective_overrides"] = 0
            moves += sum(1 for agent_id, destination in decisions.items() if destination != current[agent_id])
            current = {agent_id: decisions[agent_id] for agent_id in sorted(decisions)}
            memory.commit(decisions)
            series.append(dict(current))
            if on_step is not None:
                on_step(logged, current, memory, actions, intervention if logged == INTERVENTION_STEP else None)
    if intervention is None:
        raise RuntimeError("intervention step was not reached")
    return {"series": series, "moves": moves, "intervention": intervention, "override_steps": override_steps, "llm_calls": 0}


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
    locations = load_initial_locations()
    references = _reference_hashes()
    config = _config(seeds, timesteps)
    write_json(destination / "world_v0_7_config.json", config)
    paths = {
        name: destination / name
        for name in (
            "world_states.jsonl",
            "social_memory_states.jsonl",
            "agent_actions.jsonl",
            "interventions.jsonl",
            "matched_fork_metrics.jsonl",
        )
    }
    for path in paths.values():
        path.write_text("", encoding="utf-8", newline="\n")
    pairs = []
    branch_worlds = []
    with paths["world_states.jsonl"].open("a", encoding="utf-8", newline="\n") as states, paths[
        "social_memory_states.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as memories, paths["agent_actions.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as actions, paths["interventions.jsonl"].open("a", encoding="utf-8", newline="\n") as interventions, paths[
        "matched_fork_metrics.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as forks:
        for spec in REGIMES:
            for seed in seeds:
                control = _run_logged(source, locations, spec, seed, False, states, memories, actions)
                perturbed = _run_logged(source, locations, spec, seed, True, states, memories, actions)
                _require_gates(spec, seed, control, perturbed, references)
                pair = _pair_metrics(spec, seed, control, perturbed)
                pairs.append(pair)
                branch_worlds.append(_branch_safety(spec, seed, "CONTROL", control))
                branch_worlds.append(_branch_safety(spec, seed, "PERTURBED", perturbed))
                _dump(interventions, _intervention_record(spec, seed, False, control, pair))
                _dump(interventions, _intervention_record(spec, seed, True, perturbed, pair))
                _dump(forks, pair)
                print(
                    f"{spec['regime_id']} seed={seed} late_tv={pair['late_tv']:.4f} class={pair['pair_class']}",
                    flush=True,
                )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    replays = _replay(source, locations, pairs)
    summary = _summarize(pairs, branch_worlds, replays)
    write_json(destination / "experiment_007_summary.json", summary)
    comparison = _compare(pairs, summary)
    write_json(destination / "regime_comparison.json", comparison)
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, pairs))
    return destination


def _run_logged(atlas, locations, spec, seed, apply, states, memories, actions) -> dict[str, Any]:
    branch = "PERTURBED" if apply else "CONTROL"
    _write_initial(states, memories, spec, seed, branch, locations)

    def write(logged, current, memory, step_actions, intervention):
        _dump_state(states, spec, seed, branch, logged, current)
        _dump(
            memories,
            {
                "regime_id": spec["regime_id"],
                "branch": branch,
                "choice_seed": seed,
                "timestep": logged,
                "social_weight": spec["social_weight"],
                "normalization": CENTERED_NORMALIZATION,
                "raw_total": memory.total,
                "raw_own": memory.own,
                "llm_calls": 0,
            },
        )
        for action in step_actions:
            _dump(actions, {"regime_id": spec["regime_id"], "branch": branch, "choice_seed": seed, **action})

    result = simulate_branch(
        atlas, locations, spec, choice_seed_value=seed, apply_intervention=apply, on_step=write
    )
    if result["llm_calls"] != 0:
        raise RuntimeError("experiment 007 called a model")
    result["trajectory_sha256"] = trajectory_hash(result["series"])
    return result


def _require_gates(spec, seed, control, perturbed, references) -> None:
    if control["series"][: INTERVENTION_STEP] != perturbed["series"][: INTERVENTION_STEP]:
        raise RuntimeError(f"{spec['regime_id']} seed {seed} diverged before the intervention")
    if control["intervention"]["selected_agents"] != perturbed["intervention"]["selected_agents"]:
        raise RuntimeError(f"{spec['regime_id']} seed {seed} intervention plan diverged")
    if control["intervention"]["target_place"] != perturbed["intervention"]["target_place"]:
        raise RuntimeError(f"{spec['regime_id']} seed {seed} target diverged")
    if control["override_steps"] != [] or perturbed["override_steps"] != [INTERVENTION_STEP]:
        raise RuntimeError("intervention continued after t=21")
    tv = _tv(_share_of(control["series"][INTERVENTION_STEP]), _share_of(perturbed["series"][INTERVENTION_STEP]))
    if abs(tv - INITIAL_TV) > 1e-12:
        raise RuntimeError(f"t=21 occupancy TV is {tv}, not 0.10")
    expected = references[spec["regime_id"]].get(seed)
    if expected is not None and control["trajectory_sha256"] != expected:
        raise RuntimeError(f"{spec['regime_id']} CONTROL seed {seed} does not match the frozen reference trajectory")
    if spec["regime_id"] == "S0" and seed == 0 and control["trajectory_sha256"] != EXPECTED_S0_SEED0:
        raise RuntimeError("S0 seed 0 hash gate failed")


def _pair_metrics(spec: dict[str, Any], seed: int, control: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, Any]:
    target = perturbed["intervention"]["target_place"]
    agents = sorted(control["series"][0])
    population = len(agents)
    tv_series = []
    js_series = []
    disagreement_series = []
    target_series = []
    for timestep in range(TIMESTEPS + 1):
        left = control["series"][timestep]
        right = perturbed["series"][timestep]
        tv_series.append(_tv(_share_of(left), _share_of(right)))
        js_series.append(_js_distance(_share_of(left), _share_of(right)))
        disagreement_series.append(sum(left[agent_id] != right[agent_id] for agent_id in agents) / population)
        target_series.append((occupancy_of(right)[target] - occupancy_of(left)[target]) / population)
    late_tv = statistics.fmean(tv_series[401:501])
    final_control = _unique_leader(_presence(control["series"], 401, 500))
    final_perturbed = _unique_leader(_presence(perturbed["series"], 401, 500))
    control_share = _share(_presence(control["series"], 401, 500))
    perturbed_share = _share(_presence(perturbed["series"], 401, 500))
    reconverged, reconverge_time = _first_run(tv_series, RECONVERGENCE_RUN)
    agent_reconverged, agent_time = _first_run(disagreement_series, RECONVERGENCE_RUN)
    return {
        "regime_id": spec["regime_id"],
        "social_weight": spec["social_weight"],
        "choice_seed": seed,
        "target_place": target,
        "selected_agents": perturbed["intervention"]["selected_agents"],
        "initial_tv": tv_series[INTERVENTION_STEP],
        "late_tv": late_tv,
        "persistence_ratio": late_tv / INITIAL_TV,
        "pair_class": classify_persistence(late_tv),
        "windows": {
            name: {
                "occupancy_tv": statistics.fmean(tv_series[start : stop + 1]),
                "js_distance": statistics.fmean(js_series[start : stop + 1]),
                "agent_disagreement": statistics.fmean(disagreement_series[start : stop + 1]),
                "target_share_delta": statistics.fmean(target_series[start : stop + 1]),
            }
            for name, start, stop in WINDOWS
        },
        "occupancy_tv": tv_series,
        "js_distance": js_series,
        "agent_disagreement": disagreement_series,
        "target_share_delta": target_series,
        "final_dominant_control": final_control,
        "final_dominant_perturbed": final_perturbed,
        "final_dominant_disagreement": final_control != final_perturbed,
        "late_hhi_difference": _hhi(perturbed_share) - _hhi(control_share),
        "late_entropy_difference": _entropy(perturbed_share) - _entropy(control_share),
        "late_dominant_share_difference": max(perturbed_share.values()) - max(control_share.values()),
        "target_late_share_difference": statistics.fmean(target_series[401:501]),
        "ever_reconverged": reconverged,
        "time_to_reconvergence": reconverge_time,
        "agent_exact_reconverged": agent_reconverged,
        "agent_time_to_reconvergence": agent_time,
        "control_trajectory_sha256": control["trajectory_sha256"],
        "perturbed_trajectory_sha256": perturbed["trajectory_sha256"],
        "llm_calls": 0,
    }


def _branch_safety(spec, seed, branch, result) -> dict[str, Any]:
    diagnostics = measure_world(result["series"], moves=result["moves"], timesteps=TIMESTEPS)
    late_counts = _presence(result["series"], 401, 500)
    late_share = _share(late_counts)
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
        "late_dominant_share": max(late_share.values()),
        "llm_calls": 0,
    }


def _summarize(pairs: list[dict[str, Any]], worlds: list[dict[str, Any]], replays: dict[str, Any]) -> dict[str, Any]:
    regimes = {}
    classes = {}
    for spec in REGIMES:
        rows = [pair for pair in pairs if pair["regime_id"] == spec["regime_id"]]
        late = statistics.fmean(pair["late_tv"] for pair in rows)
        label = classify_persistence(late)
        classes[spec["regime_id"]] = label
        regimes[spec["regime_id"]] = _regime_summary(spec, rows, worlds, late, label)
    return {
        "experiment_id": "experiment_007_early_perturbation",
        "note": ANALYSIS_NOTE,
        "persistence_rule": PERSISTENCE_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
        "conclusion": overall_conclusion(classes),
        "normalization": CENTERED_NORMALIZATION,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "llm_calls": 0,
        "world_count": len(worlds),
        "pair_count": len(pairs),
        "timesteps": TIMESTEPS,
        "choice_seeds": list(CHOICE_SEEDS),
        "s0_control_matches_zero_social_reference": True,
        "control_matches_experiment_006": True,
        "js_definition": "Jensen-Shannon distance = sqrt(JS divergence), with natural log.",
        "final_dominant_window": "t=401–500 committed presence, unique maximum, tie unmarked",
        "regimes": regimes,
        "exact_replay": replays,
    }


def _regime_summary(spec, rows, worlds, late, label) -> dict[str, Any]:
    return {
        "social_weight": spec["social_weight"],
        "n_pairs": len(rows),
        "llm_calls": 0,
        "initial_tv": _dist([pair["initial_tv"] for pair in rows]),
        "late_tv": _dist([pair["late_tv"] for pair in rows]),
        "persistence_ratio": _dist([pair["persistence_ratio"] for pair in rows]),
        "regime_late_tv": late,
        "regime_persistence_ratio": late / INITIAL_TV,
        "persistence_class": label,
        "pair_class_counts": {
            name: sum(pair["pair_class"] == name for pair in rows)
            for name in ("attenuated", "persistent residual", "amplified divergence")
        },
        "windows": {
            name: {
                metric: _dist([pair["windows"][name][metric] for pair in rows])
                for metric in ("occupancy_tv", "js_distance", "agent_disagreement", "target_share_delta")
            }
            for name, _start, _stop in WINDOWS
        },
        "final_dominant_disagreement_rate": statistics.fmean(
            float(pair["final_dominant_disagreement"]) for pair in rows
        ),
        "late_hhi_difference": _dist([pair["late_hhi_difference"] for pair in rows]),
        "late_entropy_difference": _dist([pair["late_entropy_difference"] for pair in rows]),
        "late_dominant_share_difference": _dist([pair["late_dominant_share_difference"] for pair in rows]),
        "target_late_share_difference": _dist([pair["target_late_share_difference"] for pair in rows]),
        "ever_reconverged_rate": statistics.fmean(float(pair["ever_reconverged"]) for pair in rows),
        "time_to_reconvergence": _dist([pair["time_to_reconvergence"] for pair in rows]),
        "agent_exact_reconverged_rate": statistics.fmean(float(pair["agent_exact_reconverged"]) for pair in rows),
        "branches": {
            branch: _safety_summary(
                [world for world in worlds if world["regime_id"] == spec["regime_id"] and world["branch"] == branch]
            )
            for branch in ("CONTROL", "PERTURBED")
        },
    }


def _safety_summary(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    lag1 = {place_id: statistics.fmean(world["lag1"][place_id] for world in worlds) for place_id in ("site_01", "site_02")}
    late_hhi = statistics.fmean(world["late_hhi"] for world in worlds)
    dominant = statistics.fmean(world["late_dominant_share"] for world in worlds)
    saturation = statistics.fmean(world["saturation_frequency"] for world in worlds)
    alternation = statistics.fmean(world["alternation_rate"] for world in worlds)
    consecutive = statistics.fmean(world["consecutive_occupancy_tv"] for world in worlds)
    unstable = is_unstable(
        lag1_site_01=lag1["site_01"],
        lag1_site_02=lag1["site_02"],
        alternation=alternation,
        alternation_cutoff=ALTERNATION_UPPER_QUARTILE,
        late_hhi=late_hhi,
        dominant_share=dominant,
        saturation=saturation,
        consecutive_tv=consecutive,
        consecutive_tv_cutoff=CONSECUTIVE_TV_UPPER_QUARTILE,
    )
    return {
        "n": len(worlds),
        "movement_rate": _dist([world["movement_rate"] for world in worlds]),
        "mean_dwell": _dist([world["mean_dwell"] for world in worlds]),
        "occupancy_entropy": _dist([world["occupancy_entropy"] for world in worlds]),
        "saturation_frequency": _dist([world["saturation_frequency"] for world in worlds]),
        "empty_frequency": _dist([world["empty_frequency"] for world in worlds]),
        "alternation_rate": _dist([world["alternation_rate"] for world in worlds]),
        "consecutive_occupancy_tv": _dist([world["consecutive_occupancy_tv"] for world in worlds]),
        "lag1": {place_id: _dist([world["lag1"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "lag2": {place_id: _dist([world["lag2"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "late_hhi": late_hhi,
        "late_dominant_share": dominant,
        "unstable": unstable,
    }


def _compare(pairs: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    by_regime = {spec["regime_id"]: [pair for pair in pairs if pair["regime_id"] == spec["regime_id"]] for spec in REGIMES}
    contrasts = {
        "RW50_minus_S0": _paired(by_regime["S0"], by_regime["RW50"]),
        "EXP99_minus_S0": _paired(by_regime["S0"], by_regime["EXP99"]),
        "EXP99_minus_RW50": _paired(by_regime["RW50"], by_regime["EXP99"]),
    }
    return {
        "experiment_id": "experiment_007_early_perturbation",
        "note": "Paired by choice seed. p-values are not used. Percentile intervals use 10000 resamples.",
        "conclusion": summary["conclusion"],
        "persistence_class": {
            regime_id: summary["regimes"][regime_id]["persistence_class"] for regime_id in summary["regimes"]
        },
        "paired_contrasts": contrasts,
        "mean_trajectory": {
            regime_id: {
                "occupancy_tv": _column_mean(rows, "occupancy_tv"),
                "js_distance": _column_mean(rows, "js_distance"),
                "agent_disagreement": _column_mean(rows, "agent_disagreement"),
                "target_share_delta": _column_mean(rows, "target_share_delta"),
            }
            for regime_id, rows in by_regime.items()
        },
    }


def _paired(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["choice_seed"]: row for row in left_rows}
    right = {row["choice_seed"]: row for row in right_rows}
    seeds = sorted(left)
    report = {}
    for metric in (
        "late_tv",
        "persistence_ratio",
        "target_late_share_difference",
        "final_dominant_disagreement",
        "late_hhi_difference",
    ):
        diffs = [float(right[seed][metric]) - float(left[seed][metric]) for seed in seeds]
        report[metric] = _effect(diffs)
    return report


def _effect(diffs: list[float]) -> dict[str, float | None]:
    mean = statistics.fmean(diffs)
    scale = statistics.stdev(diffs) if len(diffs) > 1 else None
    low, high = _bootstrap_ci(diffs)
    return {
        "mean_difference": mean,
        "sd": scale,
        "cohens_dz": None if scale in (None, 0.0) else mean / scale,
        "ci95_low": low,
        "ci95_high": high,
        "n_pairs": len(diffs),
    }


def _bootstrap_ci(diffs: list[float]) -> tuple[float, float]:
    sampler = random.Random(ANALYSIS_BOOTSTRAP_SEED)
    means = []
    size = len(diffs)
    for _draw in range(BOOTSTRAP_DRAWS):
        sample = [diffs[sampler.randrange(size)] for _index in range(size)]
        means.append(statistics.fmean(sample))
    means.sort()
    low = means[int(0.025 * (BOOTSTRAP_DRAWS - 1))]
    high = means[int(0.975 * (BOOTSTRAP_DRAWS - 1))]
    return low, high


def _column_mean(rows: list[dict[str, Any]], key: str) -> list[float]:
    width = len(rows[0][key])
    return [statistics.fmean(row[key][index] for row in rows) for index in range(width)]


def _replay(atlas, locations, pairs) -> dict[str, Any]:
    requested = (
        ("S0", False, 0),
        ("RW50", True, 0),
        ("EXP99", False, 0),
    )
    found = {}
    for regime_id, apply, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        result = simulate_branch(atlas, locations, spec, choice_seed_value=seed, apply_intervention=apply)
        digest = trajectory_hash(result["series"])
        field = "perturbed_trajectory_sha256" if apply else "control_trajectory_sha256"
        original = next(pair[field] for pair in pairs if pair["regime_id"] == regime_id and pair["choice_seed"] == seed)
        if digest != original:
            raise RuntimeError(f"{regime_id} {'PERTURBED' if apply else 'CONTROL'} seed {seed} replay mismatch")
        found[f"{regime_id}_{'PERTURBED' if apply else 'CONTROL'}_seed{seed}"] = {
            "trajectory_sha256": digest,
            "match": True,
        }
    return found


def _reference_hashes() -> dict[str, dict[int, str]]:
    s0 = json.loads((WORLD_V04_DIR / "run_manifest.json").read_text(encoding="utf-8"))["trajectory_sha256"]
    later = json.loads((WORLD_V06_DIR / "run_manifest.json").read_text(encoding="utf-8"))["trajectory_sha256"]
    if s0["S0:0"] != EXPECTED_S0_SEED0:
        raise RuntimeError("frozen S0 seed 0 hash is not the preregistered value")
    return {
        "S0": {seed: s0[f"S0:{seed}"] for seed in CHOICE_SEEDS},
        "RW50": {seed: later[f"RW50:100:{seed}"] for seed in CHOICE_SEEDS},
        "EXP99": {seed: later[f"EXP99:100:{seed}"] for seed in CHOICE_SEEDS},
    }


def _make_memory(spec: dict[str, Any], agent_ids: list[str]) -> RollingMemory | DecayMemory:
    if spec["memory"] == "rolling":
        return RollingMemory(agent_ids, int(spec["window"]))
    return DecayMemory(agent_ids, float(spec["decay"]))


def _intervention_record(spec, seed, applied, result, pair) -> dict[str, Any]:
    plan = result["intervention"]
    return {
        "experiment_id": "experiment_007_early_perturbation",
        "regime": spec["regime_id"],
        "choice_seed": seed,
        "branch": "PERTURBED" if applied else "CONTROL",
        "applied": applied,
        "intervention_t": INTERVENTION_STEP,
        "target_place": plan["target_place"],
        "target_rule": "t20_runner_up",
        "selected_agents": plan["selected_agents"],
        "original_proposed_destinations": plan["original_proposed_destinations"],
        "forced_destination": plan["target_place"],
        "effective_overrides": plan["effective_overrides"],
        "initial_occupancy_tv": pair["initial_tv"] if applied else 0.0,
        "llm_calls": 0,
    }


def _first_run(values: list[float], length: int) -> tuple[bool, int | None]:
    run = 0
    for timestep in range(INTERVENTION_STEP, len(values)):
        if values[timestep] == 0.0:
            run += 1
            if run >= length:
                return True, timestep
        else:
            run = 0
    return False, None


def _presence(series: list[dict[str, str]], start: int, stop: int) -> dict[str, int]:
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


def _config(seeds: tuple[int, ...], timesteps: int) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_7",
        "experiment_id": "experiment_007_early_perturbation",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "normalization": CENTERED_NORMALIZATION,
        "regimes": [dict(spec) for spec in REGIMES],
        "branches": ["CONTROL", "PERTURBED"],
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "intervention_step": INTERVENTION_STEP,
        "override_count": OVERRIDE_COUNT,
        "initial_occupancy_tv": INITIAL_TV,
        "hash_material": HASH_MATERIAL,
        "windows": [{"name": name, "start": start, "stop": stop} for name, start, stop in WINDOWS],
        "persistence_rule": PERSISTENCE_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
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


def _manifest(destination: Path, config: dict[str, Any], summary: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_007_early_perturbation",
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "pair_count": summary["pair_count"],
        "complete": summary["world_count"] == len(REGIMES) * len(CHOICE_SEEDS) * 2,
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "s0_seed0_sha256": EXPECTED_S0_SEED0,
        "persistence_class": {
            regime_id: summary["regimes"][regime_id]["persistence_class"] for regime_id in summary["regimes"]
        },
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": {
            f"{pair['regime_id']}:{pair['choice_seed']}:CONTROL": pair["control_trajectory_sha256"] for pair in pairs
        }
        | {
            f"{pair['regime_id']}:{pair['choice_seed']}:PERTURBED": pair["perturbed_trajectory_sha256"] for pair in pairs
        },
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_7_config.json",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "interventions.jsonl",
                "matched_fork_metrics.jsonl",
                "experiment_007_summary.json",
                "regime_comparison.json",
            )
        },
    }


def _write_initial(states, memories, spec, seed, branch, locations) -> None:
    _dump_state(states, spec, seed, branch, 0, locations)
    zeros = {place_id: 0 for place_id in PLACE_IDS}
    _dump(
        memories,
        {
            "regime_id": spec["regime_id"],
            "branch": branch,
            "choice_seed": seed,
            "timestep": 0,
            "social_weight": spec["social_weight"],
            "raw_total": zeros,
            "raw_own": {agent_id: dict(zeros) for agent_id in sorted(locations)},
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
            ATLAS_DIR,
            CALIBRATION_DIR,
        )
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 007 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_007")
    main()
