"""Experiment 009: causal state-carrier decomposition.

One-shot perturbations are copied from Experiment 007. After t=21 is committed,
selected branches swap non-social state and social memory. This module does not
call a model and does not retune partitions after seeing results.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import world.experiment_005 as memory_experiment
from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.choice import CHOICE_ALGORITHM, RECONSIDER_ALGORITHM, agent_reconsiders, choice_seed
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
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_9"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
EXPECTED_S0_SEED0 = "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
INTERVENTION_STEP = 21
CHECKPOINT = "POST_T21_PRE_T22"
OVERRIDE_COUNT = 2
INITIAL_TV = 0.10
MEMORY_TOLERANCE = 1e-12
ANALYSIS_BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 10000
BRANCHES = ("CONTROL", "PERTURBED", "NON_SOCIAL_ONLY", "SOCIAL_MEMORY_ONLY", "FULL_RESET")
HYBRID_BRANCHES = ("NON_SOCIAL_ONLY", "SOCIAL_MEMORY_ONLY", "FULL_RESET")
PERTURBED_LINEAGE = ("PERTURBED", "NON_SOCIAL_ONLY", "SOCIAL_MEMORY_ONLY", "FULL_RESET")
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
POST_SURGERY_WINDOWS = (
    ("t22_50", 22, 50),
    ("t51_100", 51, 100),
    ("t101_200", 101, 200),
    ("t201_300", 201, 300),
    ("t301_400", 301, 400),
    ("t401_500", 401, 500),
)
EXPECTED_PERTURBED_CLASS = {"S0": "attenuated", "RW50": "persistent residual", "EXP99": "persistent residual"}
PARTITIONS = (
    "non_social_decision_state",
    "social_memory_state",
    "RNG_state",
    "logging_only_state",
    "derived_recomputable_state",
)
ANALYSIS_NOTE = (
    "State surgery at POST_T21_PRE_T22 is a causal mechanism probe inside the frozen simulation. "
    "Hybrid branches are not natural histories. Results describe which internal state carries a "
    "collective residual. They do not establish human path dependence, tourist-behavior validity, "
    "real-world causality, collective path dependence, emergence, or self-organization."
)
SURGERY_RULE = (
    "Finish the t=20 to t=21 transition, including the Experiment 007 override, the t=21 commit, "
    "and the social-memory update that t=22 will read. Surgery then copies partitions and consumes no RNG. "
    "t=22 autonomous evolution resumes from the spliced state."
)
CARRIER_RULE = (
    "N means NON_SOCIAL_ONLY late TV is not attenuated. M means SOCIAL_MEMORY_ONLY late TV is not attenuated. "
    "Type A: perturbed persists, N is false, and M is false. "
    "Type B: N is false and M is true. Type C: N is true and M is false. Type D: N is true and M is true. "
    "Retention fractions are descriptive ratios of regime-mean late TV. They are not additive mediation shares."
)
CONCLUSION_RULE = (
    "If the full-reset gate, the S0 memory-only gate, or the perturbed lineage gate fails: "
    "Experiment 009 mechanism classification is invalid pending state-dependency audit. "
    "If RW50 and EXP99 carrier types differ: the causal carrier of historical residuals depends on social-memory architecture. "
    "If both are type A: persistent collective residuals require coupled perturbed social-memory and non-social state under the tested architecture. "
    "If both are type B: perturbed social-memory state can independently regenerate persistent collective divergence after spatial/non-social state is reset. "
    "If both are type C: perturbed non-social decision state can independently regenerate persistent collective divergence after social memory is reset. "
    "If both are type D: persistent collective divergence can be carried by multiple sufficient internal state channels under the tested architecture."
)
GATE_FAILURE_CONCLUSION = "Experiment 009 mechanism classification is invalid pending state-dependency audit"


@contextmanager
def _weight(value: float) -> Iterator[None]:
    original = memory_experiment.SOCIAL_WEIGHT
    memory_experiment.SOCIAL_WEIGHT = float(value)
    try:
        yield
    finally:
        memory_experiment.SOCIAL_WEIGHT = original


def build_decision_state_inventory() -> dict[str, Any]:
    fields = [
        _field(
            "current_location",
            "branch location map",
            "decision loop origin; occupancy_of feeds Atlas crowding",
            "replaced by the committed destination after each timestep",
            "decision-relevant",
            "non_social_decision_state",
            "copy the full agent-to-place map from the named source",
        ),
        _field(
            "occupancy",
            "derived from current_location",
            "Atlas lookup crowding input inside _utilities",
            "recomputed at the start of each timestep; not stored",
            "decision-relevant only as a function of current_location",
            "derived_recomputable_state",
            "do not copy; recompute from the installed locations",
        ),
        _field(
            "distance_matrix",
            "frozen geography",
            "travel_distance(origin, destination) inside _utilities",
            "not written during a run",
            "decision-relevant frozen input",
            "derived_recomputable_state",
            "untouched",
        ),
        _field(
            "place_capacity",
            "frozen place table",
            "Atlas crowding level inside _utilities",
            "not written during a run",
            "decision-relevant frozen input",
            "derived_recomputable_state",
            "untouched",
        ),
        _field(
            "utility_atlas",
            "frozen utility_atlas_v1",
            "atlas.lookup(agent, place, distance, occupancy, capacity)",
            "not written during a run",
            "decision-relevant frozen input",
            "derived_recomputable_state",
            "untouched",
        ),
        _field(
            "familiarity_weight",
            "experiment_005 constant 0",
            "logged on actions; _utilities does not read it",
            "not written during a run",
            "logged/history but not decision-relevant under Experiment 009",
            "logging_only_state",
            "untouched",
        ),
        _field(
            "visit_count",
            "absent from the decision loop",
            "not read",
            "not stored",
            "logged/history but not decision-relevant under Experiment 009",
            "logging_only_state",
            "no field to copy",
        ),
        _field(
            "previous_location",
            "absent from the decision loop",
            "not read; dwell is derived later by measure_world from the location series",
            "not stored",
            "logged/history but not decision-relevant under Experiment 009",
            "logging_only_state",
            "no field to copy",
        ),
        _field(
            "dwell_or_cooldown",
            "absent from the decision loop",
            "not read; reconsideration has no stored cooldown",
            "not stored",
            "logged/history but not decision-relevant under Experiment 009",
            "logging_only_state",
            "no field to copy",
        ),
        _field(
            "reconsideration_draw",
            "world.choice.agent_reconsiders",
            "decides whether the agent samples a new destination",
            "stateless SHA-256 seed of choice seed, timestep index, and agent id",
            "decision-relevant RNG schedule",
            "RNG_state",
            "untouched; surgery consumes no draw",
        ),
        _field(
            "softmax_draw",
            "world.choice.choice_seed plus _select",
            "samples one destination when the agent reconsiders",
            "stateless SHA-256 seed; one shared draw for total and base utilities",
            "decision-relevant RNG schedule",
            "RNG_state",
            "untouched; surgery consumes no draw",
        ),
        _field(
            "rolling_commits",
            "RollingMemory.commits",
            "source of the rolling presence counts",
            "append the committed destination map; drop the oldest map past the window",
            "decision-relevant social memory",
            "social_memory_state",
            "copy the full deque, oldest to newest, not the current z-score",
        ),
        _field(
            "rolling_total",
            "RollingMemory.total",
            "loo_raw subtracts the focal agent from this cache",
            "incremented and decremented inside commit",
            "decision-relevant social memory",
            "social_memory_state",
            "copy the place-count cache together with the commit buffer",
        ),
        _field(
            "rolling_own",
            "RollingMemory.own",
            "loo_raw subtracts this agent-place cache",
            "incremented and decremented inside commit",
            "decision-relevant social memory",
            "social_memory_state",
            "copy every agent-place count together with the commit buffer",
        ),
        _field(
            "rolling_window",
            "RollingMemory.window",
            "commit drops history beyond this length",
            "fixed at construction",
            "decision-relevant architecture constant",
            "social_memory_state",
            "untouched within a regime; RW50 and S0 use window 50",
        ),
        _field(
            "decay_total",
            "DecayMemory.total",
            "loo_raw reads the decayed occupancy accumulator",
            "decay * previous + current committed occupancy",
            "decision-relevant social memory",
            "social_memory_state",
            "copy the full pre-normalization accumulator",
        ),
        _field(
            "decay_own",
            "DecayMemory.own",
            "loo_raw subtracts the focal decayed presence",
            "decay * previous + the focal agent's current commitment",
            "decision-relevant social memory",
            "social_memory_state",
            "copy every agent accumulator, not the current z-score",
        ),
        _field(
            "decay_rate",
            "DecayMemory.decay",
            "scales the accumulator on commit",
            "fixed at construction",
            "decision-relevant architecture constant",
            "social_memory_state",
            "untouched within a regime; EXP99 uses 0.99",
        ),
        _field(
            "centered_social_score",
            "centered_scores(loo_raw)",
            "_utilities multiplies the clipped score by social weight",
            "recomputed every agent-timestep; not stored on the memory",
            "decision-relevant only as a function of social memory",
            "derived_recomputable_state",
            "do not copy the z-score; recompute after the memory partition is installed",
        ),
        _field(
            "social_weight",
            "experiment_005.SOCIAL_WEIGHT regime assignment",
            "_utilities adds weight * score to base utility",
            "set for the regime and restored afterward",
            "decision-relevant regime constant, identical across branches",
            "derived_recomputable_state",
            "untouched by surgery",
        ),
        _field(
            "location_series_and_move_counter",
            "simulator log",
            "not read by the next decision",
            "appended after commit; movement rate is derived from the series",
            "logging only",
            "logging_only_state",
            "do not feed the log back into the decision state",
        ),
    ]
    unknown = [field for field in fields if field["partition"] not in PARTITIONS]
    if unknown:
        raise RuntimeError("decision-state inventory has an unassigned partition")
    return {
        "experiment_id": "experiment_009_state_carriers",
        "checkpoint": CHECKPOINT,
        "method": "Trace of the Experiment 007 decision loop: reconsideration, distance, Atlas lookup, base utility, social utility, softmax, commit, and memory update.",
        "partitions": list(PARTITIONS),
        "decision_relevant_partitions": ["non_social_decision_state", "social_memory_state", "RNG_state"],
        "fields": fields,
        "copy_depth": {
            "rolling": ["commits", "total", "own", "window"],
            "exponential": ["total", "own", "decay"],
            "forbidden": "copying only the current centered score",
        },
    }


def classify_persistence(late_tv: float) -> str:
    if late_tv < 0.025:
        return "attenuated"
    if late_tv < INITIAL_TV:
        return "persistent residual"
    return "amplified divergence"


def classify_carrier(perturbed_class: str, non_social_class: str, memory_class: str) -> str:
    non_social_persists = non_social_class != "attenuated"
    memory_persists = memory_class != "attenuated"
    if perturbed_class != "attenuated" and not non_social_persists and not memory_persists:
        return "A"
    if not non_social_persists and memory_persists:
        return "B"
    if non_social_persists and not memory_persists:
        return "C"
    if non_social_persists and memory_persists:
        return "D"
    raise RuntimeError("carrier taxonomy does not cover this persistence pattern")


def retention_fraction(component: float, full: float) -> float | None:
    if full == 0.0:
        return None
    return component / full


def overall_conclusion(carriers: dict[str, str] | None, *, gates_ok: bool) -> str:
    if not gates_ok or not carriers:
        return GATE_FAILURE_CONCLUSION
    if carriers["RW50"] != carriers["EXP99"]:
        return "the causal carrier of historical residuals depends on social-memory architecture"
    sentences = {
        "A": "persistent collective residuals require coupled perturbed social-memory and non-social state under the tested architecture",
        "B": "perturbed social-memory state can independently regenerate persistent collective divergence after spatial/non-social state is reset",
        "C": "perturbed non-social decision state can independently regenerate persistent collective divergence after social memory is reset",
        "D": "persistent collective divergence can be carried by multiple sufficient internal state channels under the tested architecture",
    }
    return sentences[carriers["RW50"]]


def memory_payload(memory: RollingMemory | DecayMemory) -> dict[str, Any]:
    if isinstance(memory, RollingMemory):
        return {
            "architecture": "rolling",
            "window": memory.window,
            "commits": [{agent_id: step[agent_id] for agent_id in sorted(step)} for step in memory.commits],
            "total": {place_id: int(memory.total[place_id]) for place_id in PLACE_IDS},
            "own": {
                agent_id: {place_id: int(memory.own[agent_id][place_id]) for place_id in PLACE_IDS}
                for agent_id in sorted(memory.own)
            },
        }
    return {
        "architecture": "exponential",
        "decay": float(memory.decay),
        "total": {place_id: float(memory.total[place_id]) for place_id in PLACE_IDS},
        "own": {
            agent_id: {place_id: float(memory.own[agent_id][place_id]) for place_id in PLACE_IDS}
            for agent_id in sorted(memory.own)
        },
    }


def install_memory(memory: RollingMemory | DecayMemory, payload: dict[str, Any]) -> None:
    if isinstance(memory, RollingMemory):
        if payload["architecture"] != "rolling" or memory.window != payload["window"]:
            raise RuntimeError("rolling memory payload does not match the branch architecture")
        memory.commits = deque(dict(step) for step in payload["commits"])
        memory.total = {place_id: int(payload["total"][place_id]) for place_id in PLACE_IDS}
        memory.own = {
            agent_id: {place_id: int(payload["own"][agent_id][place_id]) for place_id in PLACE_IDS}
            for agent_id in payload["own"]
        }
        return
    if payload["architecture"] != "exponential" or float(memory.decay) != float(payload["decay"]):
        raise RuntimeError("exponential memory payload does not match the branch architecture")
    memory.total = {place_id: float(payload["total"][place_id]) for place_id in PLACE_IDS}
    memory.own = {
        agent_id: {place_id: float(payload["own"][agent_id][place_id]) for place_id in PLACE_IDS}
        for agent_id in payload["own"]
    }


def state_hashes(locations: dict[str, str], memory: RollingMemory | DecayMemory, choice_seed_value: int) -> dict[str, Any]:
    non_social = {"locations": {agent_id: locations[agent_id] for agent_id in sorted(locations)}}
    social = memory_payload(memory)
    rng_state = {
        "choice_algorithm": CHOICE_ALGORITHM,
        "reconsider_algorithm": RECONSIDER_ALGORITHM,
        "choice_seed": choice_seed_value,
        "next_timestep_index": INTERVENTION_STEP,
        "surgery_rng_draws": 0,
    }
    decision = {"non_social": non_social, "social_memory": social}
    return {
        "non_social": non_social,
        "social_memory": social,
        "rng_state": rng_state,
        "non_social_sha256": canonical_sha256(non_social),
        "social_memory_sha256": canonical_sha256(social),
        "rng_state_sha256": canonical_sha256(rng_state),
        "decision_state_sha256": canonical_sha256(decision),
    }


def raw_memory_vector(memory: RollingMemory | DecayMemory) -> tuple[float, ...]:
    payload = memory_payload(memory)
    totals = [float(payload["total"][place_id]) for place_id in PLACE_IDS]
    owns = [
        float(payload["own"][agent_id][place_id])
        for agent_id in sorted(payload["own"])
        for place_id in PLACE_IDS
    ]
    return tuple(totals + owns)


def memory_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def simulate_branch(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    spec: dict[str, Any],
    *,
    choice_seed_value: int,
    branch: str,
    plan: dict[str, Any],
    timesteps: int = TIMESTEPS,
    sources: dict[str, dict[str, Any]] | None = None,
    on_step: Any = None,
) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise RuntimeError("unknown carrier branch")
    if branch in HYBRID_BRANCHES and not sources:
        raise RuntimeError("hybrid surgery requires control and perturbed checkpoints")
    matrix = distance_matrix()
    places = place_index()
    memory = _make_memory(spec, list(locations))
    current = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    series = [dict(current)]
    memory_trace = [raw_memory_vector(memory)]
    dynamics = hashlib.sha256()
    decisions_only = hashlib.sha256()
    checkpoint = None
    surgery_record = None
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
                    destination, probabilities = _select(
                        list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw
                    )
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
                        "branch": branch,
                        "intervention_applied": False,
                        "base_utility": bases,
                        "popularity_score": scores,
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
            if logged == INTERVENTION_STEP and branch in PERTURBED_LINEAGE:
                chosen = set(plan["selected_agents"])
                for agent_id in plan["selected_agents"]:
                    decisions[agent_id] = plan["target_place"]
                for action in actions:
                    if action["agent_id"] in chosen:
                        action["destination"] = plan["target_place"]
                        action["stayed"] = action["destination"] == action["origin"]
                        action["intervention_applied"] = True
                if sum(action["intervention_applied"] for action in actions) != OVERRIDE_COUNT:
                    raise RuntimeError("perturbation did not override exactly two agents")
                override_steps.append(logged)
            current = {agent_id: decisions[agent_id] for agent_id in sorted(decisions)}
            memory.commit(decisions)
            if logged == INTERVENTION_STEP:
                pre = state_hashes(current, memory, choice_seed_value)
                _apply_surgery(branch, current, memory, sources)
                current = _surgery_locations(branch, current, sources)
                post = state_hashes(current, memory, choice_seed_value)
                checkpoint = post
                surgery_record = _surgery_record(spec, choice_seed_value, branch, plan, pre, post, current, sources)
                _require_surgery(branch, post, sources, surgery_record)
            series.append(dict(current))
            memory_trace.append(raw_memory_vector(memory))
            if logged >= INTERVENTION_STEP + 1:
                _extend_hash(dynamics, _step_payload(logged, current, actions, memory, include_memory=True))
                _extend_hash(decisions_only, _step_payload(logged, current, actions, memory, include_memory=False))
            if on_step is not None:
                on_step(logged, current, memory, actions, surgery_record if logged == INTERVENTION_STEP else None)
    if checkpoint is None or surgery_record is None:
        raise RuntimeError("surgery checkpoint was not reached")
    moves = sum(
        previous[agent_id] != nxt[agent_id]
        for previous, nxt in zip(series, series[1:])
        for agent_id in previous
    )
    return {
        "series": series,
        "moves": moves,
        "checkpoint": checkpoint,
        "surgery": surgery_record,
        "override_steps": override_steps,
        "memory_trace": memory_trace,
        "canonical_dynamics_sha256": dynamics.hexdigest(),
        "decision_dynamics_sha256": decisions_only.hexdigest(),
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
    inventory = build_decision_state_inventory()
    write_json(destination / "decision_state_inventory.json", inventory)
    config = _config(seeds, timesteps)
    write_json(destination / "world_v0_9_config.json", config)
    names = (
        "world_states.jsonl",
        "social_memory_states.jsonl",
        "agent_actions.jsonl",
        "state_surgery_manifest.jsonl",
        "carrier_metrics.jsonl",
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
    ) as actions, paths["state_surgery_manifest.jsonl"].open("a", encoding="utf-8", newline="\n") as surgeries, paths[
        "carrier_metrics.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as carriers:
        for spec in REGIMES:
            for seed in seeds:
                runs = _run_seed(
                    source, initial, spec, seed, plans[(spec["regime_id"], seed)], references, states, memories, actions, surgeries
                )
                record = _seed_metrics(spec, seed, runs)
                for result in runs.values():
                    result.pop("memory_trace", None)
                metrics.append(record)
                for branch, result in runs.items():
                    branch_worlds.append(_branch_safety(spec, seed, branch, result))
                _dump(carriers, record)
                print(
                    f"{spec['regime_id']} seed={seed} full={record['branches']['PERTURBED']['late_tv']:.4f} "
                    f"non_social={record['branches']['NON_SOCIAL_ONLY']['late_tv']:.4f} "
                    f"memory={record['branches']['SOCIAL_MEMORY_ONLY']['late_tv']:.4f}",
                    flush=True,
                )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    _require_perturbed_replication(metrics)
    replays = _replay(source, initial, plans, metrics)
    summary = _summarize(metrics, branch_worlds, replays)
    write_json(destination / "experiment_009_summary.json", summary)
    comparison = _compare(metrics, summary)
    write_json(destination / "regime_comparison.json", comparison)
    write_json(destination / "carrier_decomposition.json", _decomposition(summary))
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, metrics))
    return destination


def load_intervention_plans() -> dict[tuple[str, int], dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (WORLD_V07_DIR / "interventions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["regime"], int(row["choice_seed"])), {})[row["branch"]] = row
    expected = {(spec["regime_id"], seed) for spec in REGIMES for seed in CHOICE_SEEDS}
    if set(grouped) != expected:
        raise RuntimeError("Experiment 007 intervention metadata does not cover every regime and seed")
    plans = {}
    for key, pair in grouped.items():
        control, perturbed = pair["CONTROL"], pair["PERTURBED"]
        if control["target_place"] != perturbed["target_place"] or control["selected_agents"] != perturbed["selected_agents"]:
            raise RuntimeError(f"{key} control and perturbed intervention metadata disagree")
        if int(perturbed["effective_overrides"]) != OVERRIDE_COUNT or int(control["effective_overrides"]) != 0:
            raise RuntimeError(f"{key} intervention override count is not the Experiment 007 record")
        if int(perturbed["intervention_t"]) != INTERVENTION_STEP:
            raise RuntimeError(f"{key} intervention timing is not t=21")
        plans[key] = {
            "target_place": perturbed["target_place"],
            "selected_agents": list(perturbed["selected_agents"]),
        }
    return plans


def _run_seed(atlas, initial, spec, seed, plan, references, states, memories, actions, surgeries) -> dict[str, dict[str, Any]]:
    runs = {}
    sources: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        result = _run_logged(
            atlas,
            initial,
            spec,
            seed,
            branch,
            plan,
            None if branch == "CONTROL" else sources,
            states,
            memories,
            actions,
        )
        if branch in {"CONTROL", "PERTURBED"}:
            sources[branch] = result["checkpoint"]
            expected = references[f"{spec['regime_id']}:{seed}:{branch}"]
            if result["trajectory_sha256"] != expected:
                raise RuntimeError(f"{spec['regime_id']} {branch} seed {seed} does not match Experiment 007")
        runs[branch] = result
        _dump(surgeries, result["surgery"])
    _require_seed_gates(spec, seed, runs)
    return runs


def _run_logged(atlas, initial, spec, seed, branch, plan, sources, states, memories, actions) -> dict[str, Any]:
    _write_initial(states, memories, spec, seed, branch, initial)

    def write(logged, current, memory, step_actions, _surgery):
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
                "llm_calls": 0,
            },
        )
        for action in step_actions:
            _dump(actions, {"regime_id": spec["regime_id"], "choice_seed": seed, **action})

    result = simulate_branch(
        atlas,
        initial,
        spec,
        choice_seed_value=seed,
        branch=branch,
        plan=plan,
        sources=sources,
        on_step=write,
    )
    if result["llm_calls"] != 0:
        raise RuntimeError("experiment 009 called a model")
    return result


def _require_seed_gates(spec, seed, runs) -> None:
    control = runs["CONTROL"]
    prefix = control["series"][:INTERVENTION_STEP]
    for branch, result in runs.items():
        if result["series"][:INTERVENTION_STEP] != prefix:
            raise RuntimeError(f"{spec['regime_id']} seed {seed} {branch} diverged before t=21")
        if branch == "CONTROL" and result["override_steps"] != []:
            raise RuntimeError("control applied an override")
        if branch in PERTURBED_LINEAGE and result["override_steps"] != [INTERVENTION_STEP]:
            raise RuntimeError(f"{branch} intervention was not confined to t=21")
    if runs["FULL_RESET"]["canonical_dynamics_sha256"] != control["canonical_dynamics_sha256"]:
        raise RuntimeError(f"{spec['regime_id']} seed {seed} FULL_RESET dynamics diverge from CONTROL after surgery")
    if spec["regime_id"] == "S0":
        if runs["SOCIAL_MEMORY_ONLY"]["decision_dynamics_sha256"] != control["decision_dynamics_sha256"]:
            raise RuntimeError("S0 social memory changed decisions while social weight was 0")
        if runs["NON_SOCIAL_ONLY"]["decision_dynamics_sha256"] != runs["PERTURBED"]["decision_dynamics_sha256"]:
            raise RuntimeError("S0 social memory changed the perturbed non-social trajectory")
    if spec["regime_id"] == "S0" and seed == 0 and control["trajectory_sha256"] != EXPECTED_S0_SEED0:
        raise RuntimeError("S0 seed 0 hash gate failed")


def _require_perturbed_replication(metrics: list[dict[str, Any]]) -> None:
    for regime_id, expected in EXPECTED_PERTURBED_CLASS.items():
        rows = [row for row in metrics if row["regime_id"] == regime_id]
        late = statistics.fmean(row["branches"]["PERTURBED"]["late_tv"] for row in rows)
        if classify_persistence(late) != expected:
            raise RuntimeError(f"{regime_id} perturbed lineage class is {classify_persistence(late)}, expected {expected}")


def _seed_metrics(spec: dict[str, Any], seed: int, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    control_series = runs["CONTROL"]["series"]
    control_trace = runs["CONTROL"]["memory_trace"]
    branches = {}
    for branch, result in runs.items():
        tv, js, disagreement = [], [], []
        agents = sorted(control_series[0])
        population = len(agents)
        for timestep in range(len(result["series"])):
            share_branch = _share_of(result["series"][timestep])
            share_control = _share_of(control_series[timestep])
            tv.append(_tv(share_branch, share_control))
            js.append(_js_distance(share_branch, share_control))
            disagreement.append(
                sum(result["series"][timestep][agent_id] != control_series[timestep][agent_id] for agent_id in agents) / population
            )
        distances = [memory_distance(left, right) for left, right in zip(result["memory_trace"], control_trace, strict=True)]
        late_tv = statistics.fmean(tv[401:501])
        branches[branch] = {
            "late_tv": late_tv,
            "persistence_class": classify_persistence(late_tv),
            "windows": {
                name: {
                    "occupancy_tv": statistics.fmean(tv[start : stop + 1]),
                    "js_distance": statistics.fmean(js[start : stop + 1]),
                    "agent_disagreement": statistics.fmean(disagreement[start : stop + 1]),
                }
                for name, start, stop in WINDOWS
            },
            "post_surgery_mean_tv": {
                name: statistics.fmean(tv[start : stop + 1]) for name, start, stop in POST_SURGERY_WINDOWS
            },
            "time_to_divergence_gt_025": _first_true(tv, lambda value: value > 0.025),
            "time_to_divergence_gt_050": _first_true(tv, lambda value: value > 0.05),
            "maximum_tv": max(tv),
            "time_of_maximum_tv": tv.index(max(tv)),
            "regeneration_latency": _first_true(tv, lambda value: value >= 0.025),
            "memory_regeneration_latency": _first_true(distances, lambda value: value > MEMORY_TOLERANCE),
            "first_choice_divergence_t": _first_location_divergence(result["series"], control_series),
            "first_agent_location_divergence_t": _first_location_divergence(result["series"], control_series),
            "first_occupancy_tv_ge_025_t": _first_true(tv, lambda value: value >= 0.025),
            "checkpoint_occupancy_tv": tv[INTERVENTION_STEP],
            "checkpoint_agent_disagreement": disagreement[INTERVENTION_STEP],
            "trajectory_sha256": result["trajectory_sha256"],
            "canonical_dynamics_sha256": result["canonical_dynamics_sha256"],
            "occupancy_tv": tv,
            "js_distance": js,
            "agent_disagreement": disagreement,
        }
    full = branches["PERTURBED"]["late_tv"]
    return {
        "regime_id": spec["regime_id"],
        "social_weight": spec["social_weight"],
        "choice_seed": seed,
        "target_place": runs["PERTURBED"]["surgery"]["target_place"],
        "selected_agents": runs["PERTURBED"]["surgery"]["selected_agents"],
        "full_reset_matches_control": runs["FULL_RESET"]["canonical_dynamics_sha256"] == runs["CONTROL"]["canonical_dynamics_sha256"],
        "branches": branches,
        "non_social_retention_component": branches["NON_SOCIAL_ONLY"]["late_tv"],
        "memory_retention_component": branches["SOCIAL_MEMORY_ONLY"]["late_tv"],
        "memory_removal_effect": full - branches["NON_SOCIAL_ONLY"]["late_tv"],
        "non_social_removal_effect": full - branches["SOCIAL_MEMORY_ONLY"]["late_tv"],
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


def _summarize(metrics, worlds, replays) -> dict[str, Any]:
    regimes = {}
    carriers = {}
    for spec in REGIMES:
        rows = [row for row in metrics if row["regime_id"] == spec["regime_id"]]
        regimes[spec["regime_id"]] = _regime_summary(spec, rows, worlds)
    for regime_id in ("RW50", "EXP99"):
        block = regimes[regime_id]
        carriers[regime_id] = classify_carrier(
            block["branches"]["PERTURBED"]["persistence_class"],
            block["branches"]["NON_SOCIAL_ONLY"]["persistence_class"],
            block["branches"]["SOCIAL_MEMORY_ONLY"]["persistence_class"],
        )
        block["carrier_type"] = carriers[regime_id]
        block["carrier_induced_amplification"] = [
            branch
            for branch in ("NON_SOCIAL_ONLY", "SOCIAL_MEMORY_ONLY")
            if block["branches"][branch]["persistence_class"] == "amplified divergence"
        ]
    return {
        "experiment_id": "experiment_009_state_carriers",
        "note": ANALYSIS_NOTE,
        "surgery_rule": SURGERY_RULE,
        "carrier_rule": CARRIER_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "timeline": {
            "t21": "intervention",
            "t21_5": "conceptual surgery checkpoint POST_T21_PRE_T22",
            "t22": "autonomous evolution resumes",
        },
        "conclusion": overall_conclusion(carriers, gates_ok=True),
        "carrier_type": carriers,
        "gates": {
            "full_reset_exact": all(row["full_reset_matches_control"] for row in metrics),
            "s0_memory_only_exact": True,
            "perturbed_lineage": True,
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
    full = statistics.fmean(row["branches"]["PERTURBED"]["late_tv"] for row in rows)
    non_social = statistics.fmean(row["branches"]["NON_SOCIAL_ONLY"]["late_tv"] for row in rows)
    memory = statistics.fmean(row["branches"]["SOCIAL_MEMORY_ONLY"]["late_tv"] for row in rows)
    return {
        "social_weight": spec["social_weight"],
        "n_seeds": len(rows),
        "llm_calls": 0,
        "D_full": full,
        "D_non_social": non_social,
        "D_memory": memory,
        "non_social_retention_fraction": retention_fraction(non_social, full),
        "memory_retention_fraction": retention_fraction(memory, full),
        "removal_contrasts": {
            "memory_removal_effect": _effect([row["memory_removal_effect"] for row in rows]),
            "non_social_removal_effect": _effect([row["non_social_removal_effect"] for row in rows]),
        },
        "branches": {
            branch: _branch_summary(rows, worlds, spec["regime_id"], branch) for branch in BRANCHES
        },
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
                for metric in ("occupancy_tv", "js_distance", "agent_disagreement")
            }
            for name, _start, _stop in WINDOWS
        },
        "post_surgery_mean_tv": {
            name: _dist([row["branches"][branch]["post_surgery_mean_tv"][name] for row in rows])
            for name, _start, _stop in POST_SURGERY_WINDOWS
        },
        "time_to_divergence_gt_025": _dist([row["branches"][branch]["time_to_divergence_gt_025"] for row in rows]),
        "time_to_divergence_gt_050": _dist([row["branches"][branch]["time_to_divergence_gt_050"] for row in rows]),
        "maximum_tv": _dist([row["branches"][branch]["maximum_tv"] for row in rows]),
        "time_of_maximum_tv": _dist([row["branches"][branch]["time_of_maximum_tv"] for row in rows]),
        "regeneration_latency": _dist([row["branches"][branch]["regeneration_latency"] for row in rows]),
        "memory_regeneration_latency": _dist([row["branches"][branch]["memory_regeneration_latency"] for row in rows]),
        "first_choice_divergence_t": _dist([row["branches"][branch]["first_choice_divergence_t"] for row in rows]),
        "first_agent_location_divergence_t": _dist(
            [row["branches"][branch]["first_agent_location_divergence_t"] for row in rows]
        ),
        "first_occupancy_tv_ge_025_t": _dist([row["branches"][branch]["first_occupancy_tv_ge_025_t"] for row in rows]),
        "checkpoint_occupancy_tv": _dist([row["branches"][branch]["checkpoint_occupancy_tv"] for row in rows]),
        "checkpoint_agent_disagreement": _dist([row["branches"][branch]["checkpoint_agent_disagreement"] for row in rows]),
        "safety": _safety_summary(
            [world for world in worlds if world["regime_id"] == regime_id and world["branch"] == branch]
        ),
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
        "experiment_id": "experiment_009_state_carriers",
        "note": "Paired by choice seed. Carrier class uses regime-mean late TV, not a p-value. Intervals use 10000 paired resamples.",
        "conclusion": summary["conclusion"],
        "carrier_type": summary["carrier_type"],
        "paired_contrasts": {
            "RW50_minus_S0_perturbed": _paired(by_regime["S0"], by_regime["RW50"], "PERTURBED"),
            "EXP99_minus_S0_perturbed": _paired(by_regime["S0"], by_regime["EXP99"], "PERTURBED"),
            "EXP99_minus_RW50_perturbed": _paired(by_regime["RW50"], by_regime["EXP99"], "PERTURBED"),
            "RW50_memory_minus_non_social": _within(by_regime["RW50"]),
            "EXP99_memory_minus_non_social": _within(by_regime["EXP99"]),
        },
        "mean_trajectory": {
            regime_id: {
                branch: {
                    "occupancy_tv": _column_mean(rows, branch, "occupancy_tv"),
                    "js_distance": _column_mean(rows, branch, "js_distance"),
                    "agent_disagreement": _column_mean(rows, branch, "agent_disagreement"),
                }
                for branch in BRANCHES
                if branch != "CONTROL"
            }
            for regime_id, rows in by_regime.items()
        },
    }


def _decomposition(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_009_state_carriers",
        "carrier_rule": CARRIER_RULE,
        "conclusion": summary["conclusion"],
        "carrier_type": summary["carrier_type"],
        "gates": summary["gates"],
        "regimes": {
            regime_id: {
                "D_full": block["D_full"],
                "D_non_social": block["D_non_social"],
                "D_memory": block["D_memory"],
                "non_social_retention_fraction": block["non_social_retention_fraction"],
                "memory_retention_fraction": block["memory_retention_fraction"],
                "removal_contrasts": block["removal_contrasts"],
                "carrier_type": block.get("carrier_type"),
                "carrier_induced_amplification": block.get("carrier_induced_amplification", []),
                "persistence_class": {
                    branch: block["branches"][branch]["persistence_class"] for branch in BRANCHES if branch != "CONTROL"
                },
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


def _paired(left_rows, right_rows, branch: str) -> dict[str, Any]:
    left = {row["choice_seed"]: row for row in left_rows}
    right = {row["choice_seed"]: row for row in right_rows}
    diffs = [
        right[seed]["branches"][branch]["late_tv"] - left[seed]["branches"][branch]["late_tv"] for seed in sorted(left)
    ]
    return _effect(diffs)


def _within(rows) -> dict[str, Any]:
    diffs = [
        row["branches"]["SOCIAL_MEMORY_ONLY"]["late_tv"] - row["branches"]["NON_SOCIAL_ONLY"]["late_tv"] for row in rows
    ]
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


def _column_mean(rows: list[dict[str, Any]], branch: str, key: str) -> list[float]:
    return [statistics.fmean(row["branches"][branch][key][index] for row in rows) for index in range(len(rows[0]["branches"][branch][key]))]


def _replay(atlas, initial, plans, metrics) -> dict[str, Any]:
    requested = (("S0", "CONTROL", 0), ("RW50", "PERTURBED", 0), ("EXP99", "NON_SOCIAL_ONLY", 0))
    found = {}
    for regime_id, branch, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        sources = None
        if branch in HYBRID_BRANCHES:
            sources = {
                name: simulate_branch(
                    atlas, initial, spec, choice_seed_value=seed, branch=name, plan=plans[(regime_id, seed)]
                )["checkpoint"]
                for name in ("CONTROL", "PERTURBED")
            }
        result = simulate_branch(
            atlas, initial, spec, choice_seed_value=seed, branch=branch, plan=plans[(regime_id, seed)], sources=sources
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


def _apply_surgery(branch: str, _locations: dict[str, str], memory: RollingMemory | DecayMemory, sources) -> None:
    if branch == "NON_SOCIAL_ONLY":
        install_memory(memory, sources["CONTROL"]["social_memory"])
    elif branch == "FULL_RESET":
        install_memory(memory, sources["CONTROL"]["social_memory"])
    elif branch == "SOCIAL_MEMORY_ONLY":
        install_memory(memory, sources["PERTURBED"]["social_memory"])


def _surgery_locations(branch: str, current: dict[str, str], sources) -> dict[str, str]:
    if branch in {"SOCIAL_MEMORY_ONLY", "FULL_RESET"}:
        return dict(sources["CONTROL"]["non_social"]["locations"])
    return current


def _require_surgery(branch, post, sources, record) -> None:
    if branch == "CONTROL":
        return
    if branch == "PERTURBED":
        return
    control, perturbed = sources["CONTROL"], sources["PERTURBED"]
    if branch == "NON_SOCIAL_ONLY":
        if post["non_social_sha256"] != perturbed["non_social_sha256"] or post["social_memory_sha256"] != control["social_memory_sha256"]:
            raise RuntimeError("NON_SOCIAL_ONLY partition hashes do not match the surgery rule")
    elif branch == "SOCIAL_MEMORY_ONLY":
        if post["non_social_sha256"] != control["non_social_sha256"] or post["social_memory_sha256"] != perturbed["social_memory_sha256"]:
            raise RuntimeError("SOCIAL_MEMORY_ONLY partition hashes do not match the surgery rule")
        if post["social_memory_sha256"] == control["social_memory_sha256"]:
            raise RuntimeError("SOCIAL_MEMORY_ONLY social memory matches CONTROL")
        if record["occupancy_tv_vs_control"] != 0.0 or record["agent_disagreement_vs_control"] != 0.0:
            raise RuntimeError("SOCIAL_MEMORY_ONLY checkpoint still differs in locations")
    elif branch == "FULL_RESET":
        if post["decision_state_sha256"] != control["decision_state_sha256"]:
            raise RuntimeError("FULL_RESET decision state does not match CONTROL")
    if post["rng_state_sha256"] != control["rng_state_sha256"]:
        raise RuntimeError("surgery changed the RNG schedule")


def _surgery_record(spec, seed, branch, plan, pre, post, locations, sources) -> dict[str, Any]:
    control_locations = locations if sources is None else sources["CONTROL"]["non_social"]["locations"]
    if branch == "CONTROL":
        control_locations = locations
    occupancy = occupancy_of(locations)
    control_occupancy = occupancy_of(control_locations)
    copied, recomputed, untouched = _surgery_fields(branch)
    non_social_source, memory_source = _sources(branch)
    return {
        "experiment_id": "experiment_009_state_carriers",
        "regime": spec["regime_id"],
        "choice_seed": seed,
        "branch": branch,
        "checkpoint": CHECKPOINT,
        "target_place": plan["target_place"],
        "selected_agents": list(plan["selected_agents"]),
        "non_social_state_source": non_social_source,
        "social_memory_state_source": memory_source,
        "pre_surgery_non_social_sha256": pre["non_social_sha256"],
        "pre_surgery_social_memory_sha256": pre["social_memory_sha256"],
        "pre_surgery_decision_state_sha256": pre["decision_state_sha256"],
        "post_surgery_non_social_sha256": post["non_social_sha256"],
        "post_surgery_social_memory_sha256": post["social_memory_sha256"],
        "post_surgery_rng_sha256": post["rng_state_sha256"],
        "post_surgery_decision_state_sha256": post["decision_state_sha256"],
        "current_occupancy": occupancy,
        "occupancy_tv_vs_control": _tv(_share_of(locations), _share_of(control_locations)),
        "agent_disagreement_vs_control": _disagreement(locations, control_locations),
        "social_memory_sha256": post["social_memory_sha256"],
        "social_memory_differs_from_control": post["social_memory_sha256"] != (
            post["social_memory_sha256"] if sources is None else sources["CONTROL"]["social_memory_sha256"]
        ),
        "rng_state_sha256": post["rng_state_sha256"],
        "fields_copied": copied,
        "fields_recomputed": recomputed,
        "fields_untouched": untouched,
        "surgery_rng_draws": 0,
        "llm_calls": 0,
    }


def _sources(branch: str) -> tuple[str, str]:
    return {
        "CONTROL": ("CONTROL", "CONTROL"),
        "PERTURBED": ("PERTURBED", "PERTURBED"),
        "NON_SOCIAL_ONLY": ("PERTURBED", "CONTROL"),
        "SOCIAL_MEMORY_ONLY": ("CONTROL", "PERTURBED"),
        "FULL_RESET": ("CONTROL", "CONTROL"),
    }[branch]


def _surgery_fields(branch: str) -> tuple[list[str], list[str], list[str]]:
    recomputed = ["occupancy", "centered_social_score"]
    untouched = ["rng_schedule", "distance_matrix", "utility_atlas", "social_weight"]
    if branch == "NON_SOCIAL_ONLY":
        return ["social_memory"], recomputed, ["current_location", *untouched]
    if branch == "SOCIAL_MEMORY_ONLY":
        return ["current_location", "social_memory"], recomputed, untouched
    if branch == "FULL_RESET":
        return ["current_location", "social_memory"], recomputed, untouched
    return [], recomputed, ["current_location", "social_memory", *untouched]


def _step_payload(logged, locations, actions, memory, *, include_memory: bool) -> dict[str, Any]:
    payload = {
        "timestep": logged,
        "locations": {agent_id: locations[agent_id] for agent_id in sorted(locations)},
        "occupancy": occupancy_of(locations),
        "agents": [
            {
                "agent_id": action["agent_id"],
                "reconsidered": action["reconsidered"],
                "destination": action["destination"],
                "probabilities": action["choice_probabilities"],
                "rng_draws": action["rng_draws"],
            }
            for action in actions
        ],
        "rng_draw_count": sum(action["rng_draws"] for action in actions),
    }
    if include_memory:
        payload["social_memory"] = memory_payload(memory)
    return payload


def _extend_hash(hasher: hashlib._Hash, payload: dict[str, Any]) -> None:
    hasher.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    hasher.update(b"\n")


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


def _first_true(values: list[float], predicate) -> int | None:
    for timestep, value in enumerate(values):
        if predicate(value):
            return timestep
    return None


def _first_location_divergence(series: list[dict[str, str]], control: list[dict[str, str]]) -> int | None:
    for timestep, (left, right) in enumerate(zip(series, control, strict=True)):
        if any(left[agent_id] != right[agent_id] for agent_id in left):
            return timestep
    return None


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


def _disagreement(left: dict[str, str], right: dict[str, str]) -> float:
    return sum(left[agent_id] != right[agent_id] for agent_id in left) / len(left)


def _field(name, owner, read, written, relevance, partition, reset) -> dict[str, str]:
    return {
        "field_name": name,
        "owner": owner,
        "where_read": read,
        "where_written": written,
        "decision_relevance": relevance,
        "partition": partition,
        "reset_rule": reset,
    }


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
        "world_id": "world_v0_9",
        "experiment_id": "experiment_009_state_carriers",
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
        "checkpoint": CHECKPOINT,
        "intervention_source": "data/worlds/world_v0_7/interventions.jsonl",
        "override_count": OVERRIDE_COUNT,
        "memory_distance_tolerance": MEMORY_TOLERANCE,
        "windows": [{"name": name, "start": start, "stop": stop} for name, start, stop in WINDOWS],
        "post_surgery_windows": [{"name": name, "start": start, "stop": stop} for name, start, stop in POST_SURGERY_WINDOWS],
        "surgery_rule": SURGERY_RULE,
        "carrier_rule": CARRIER_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "expected_perturbed_class": EXPECTED_PERTURBED_CLASS,
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
        "experiment_id": "experiment_009_state_carriers",
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "seed_count": summary["seed_count"],
        "complete": summary["world_count"] == len(REGIMES) * len(CHOICE_SEEDS) * len(BRANCHES),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "s0_seed0_sha256": EXPECTED_S0_SEED0,
        "gates": summary["gates"],
        "carrier_type": summary["carrier_type"],
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": trajectories,
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_9_config.json",
                "decision_state_inventory.json",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "state_surgery_manifest.jsonl",
                "carrier_metrics.jsonl",
                "carrier_decomposition.json",
                "experiment_009_summary.json",
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
            ATLAS_DIR,
            CALIBRATION_DIR,
        )
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 009 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_009")
    main()
