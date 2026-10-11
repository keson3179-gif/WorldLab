"""Experiment 008: symmetric alternative-history forks.

Two equal-sized early overrides point the same agents at two different places.
This module does not call a model and does not retune rules after seeing results.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from contextlib import contextmanager
from itertools import combinations
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
from world.experiment_007 import OUTPUT_DIR as WORLD_V07_DIR
from world.experiment_007 import ranked_places
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_8"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
EXPECTED_S0_SEED0 = "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
INTERVENTION_STEP = 21
OVERRIDE_COUNT = 2
INITIAL_TV = 0.10
RECONVERGENCE_RUN = 20
DIRECTIONAL_THRESHOLD = 0.01
ANALYSIS_BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 10000
BRANCHES = ("CONTROL", "HISTORY_A", "HISTORY_B")
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
    "Symmetric two-agent alternative histories at the t=20 to t=21 transition. "
    "Classes describe persistence and directional retention of assigned histories. "
    "They do not establish collective path dependence, lock-in, a butterfly effect, "
    "self-organization, emergence, human realism, or behavioral validity."
)
PAIR_RULE = (
    "Exclude the t=20 dominant place, using lexical place_id order when occupancy ties. "
    "Among the remaining places, choose the unordered pair with the smallest absolute expected-choice-mass gap. "
    "Equal gaps break by lexical (min place_id, max place_id). "
    "HISTORY_A receives the lexical smaller place and HISTORY_B the larger. "
    "No seed is dropped for balance."
)
PERSISTENCE_RULE = (
    "Regime class uses the mean A-B occupancy TV over t=401-500. "
    "attenuated: late TV < 0.025. persistent residual: 0.025 <= late TV < 0.10. "
    "amplified divergence: late TV >= 0.10. persistence_ratio = late TV / 0.10."
)
DIRECTION_RULE = (
    "late DHI is the mean over seeds of the mean directional history index during t=401-500. "
    "no directional retention: late DHI <= 0. weak directional retention: 0 < late DHI < 0.01. "
    "directional retention: late DHI >= 0.01. "
    "directionally consistent also requires the paired bootstrap 95 percent interval lower bound to exceed 0. "
    "Otherwise a mean at or above 0.01 is directional signal with uncertainty. The 0.01 cutoff is fixed."
)
TAXONOMY_RULE = (
    "Type 0 attenuation: late A-B TV < 0.025. "
    "Type 1 nondirectional persistent divergence: late TV >= 0.025 and the directional-retention gate fails. "
    "Type 2 directional persistent history dependence: late TV >= 0.025, late DHI >= 0.01, and the interval lower bound exceeds 0. "
    "Type 3 amplified directional divergence: late TV >= 0.10, late DHI >= 0.01, and the interval lower bound exceeds 0. "
    "Type 2 or 3 is described as persistent directional dependence on experimentally assigned early histories."
)
CONCLUSION_RULE = (
    "If every regime is type 0: symmetric alternative early histories do not produce persistent collective divergence under the tested regimes. "
    "If a social regime is type 1 and none is type 2 or 3: symmetric alternative early histories produce persistent but nondirectional collective divergence under social feedback. "
    "If a social regime is type 2 and none is type 3: experimentally assigned alternative early histories produce persistent directional differences in later collective states. "
    "If a social regime is type 3: experimentally assigned alternative early histories produce amplified and persistent directional differences in later collective states. "
    "Social regimes are RW50 and EXP99. These sentences do not claim that collective path dependence is proven."
)
RECONVERGENCE_RULE = (
    "Stable A-B reconvergence requires 20 consecutive timesteps with occupancy TV exactly 0, counted only from t=21. "
    "time_to_reconvergence is the timestep that completes that run. Agent-level reconvergence is an audit only."
)
HASH_MATERIAL = "experiment_008{regime_id}{choice_seed}{agent_id}"


@contextmanager
def _weight(value: float) -> Iterator[None]:
    original = memory_experiment.SOCIAL_WEIGHT
    memory_experiment.SOCIAL_WEIGHT = float(value)
    try:
        yield
    finally:
        memory_experiment.SOCIAL_WEIGHT = original


def intervention_hash(regime_id: str, choice_seed_value: int, agent_id: str) -> str:
    material = f"experiment_008{regime_id}{choice_seed_value}{agent_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def expected_choice_mass(actions: list[dict[str, Any]]) -> dict[str, float]:
    mass = {place_id: 0.0 for place_id in PLACE_IDS}
    for action in actions:
        for place_id, probability in action["choice_probabilities"].items():
            mass[place_id] += float(probability)
    return mass


def select_history_pair(occupancy: dict[str, int], mass: dict[str, float]) -> dict[str, Any]:
    dominant = ranked_places(occupancy)[0]
    candidates = [place_id for place_id in PLACE_IDS if place_id != dominant]
    ranked = sorted(
        combinations(candidates, 2),
        key=lambda pair: (abs(mass[pair[0]] - mass[pair[1]]), *sorted(pair)),
    )
    left, right = sorted(ranked[0])
    return {
        "current_dominant": dominant,
        "target_A": left,
        "target_B": right,
        "expected_mass_A": mass[left],
        "expected_mass_B": mass[right],
        "absolute_gap": abs(mass[left] - mass[right]),
        "relative_gap": _relative_gap(mass[left], mass[right]),
    }


def select_agents(
    regime_id: str,
    choice_seed_value: int,
    proposed: dict[str, str],
    target_a: str,
    target_b: str,
) -> list[str]:
    eligible = [
        agent_id
        for agent_id in sorted(proposed)
        if proposed[agent_id] not in {target_a, target_b}
    ]
    ordered = sorted(
        eligible,
        key=lambda agent_id: (intervention_hash(regime_id, choice_seed_value, agent_id), agent_id),
    )
    if len(ordered) < OVERRIDE_COUNT:
        raise RuntimeError(
            f"{regime_id} seed {choice_seed_value} has {len(ordered)} agents outside {target_a} and {target_b}"
        )
    return ordered[:OVERRIDE_COUNT]


def directional_history_index(share_a_in_a: float, share_a_in_b: float, share_b_in_b: float, share_b_in_a: float) -> float:
    return ((share_a_in_a - share_a_in_b) + (share_b_in_b - share_b_in_a)) / 2.0


def classify_persistence(late_tv: float) -> str:
    if late_tv < 0.025:
        return "attenuated"
    if late_tv < INITIAL_TV:
        return "persistent residual"
    return "amplified divergence"


def classify_direction(late_dhi: float, ci_low: float) -> str:
    if late_dhi <= 0.0:
        return "no directional retention"
    if late_dhi < DIRECTIONAL_THRESHOLD:
        return "weak directional retention"
    if ci_low > 0.0:
        return "directionally consistent"
    return "directional signal with uncertainty"


def classify_outcome(late_tv: float, late_dhi: float, ci_low: float) -> str:
    directional = late_dhi >= DIRECTIONAL_THRESHOLD and ci_low > 0.0
    if late_tv >= INITIAL_TV and directional:
        return "type_3"
    if late_tv >= 0.025 and directional:
        return "type_2"
    if late_tv >= 0.025:
        return "type_1"
    return "type_0"


def overall_conclusion(outcomes: dict[str, str]) -> str:
    social = [outcomes["RW50"], outcomes["EXP99"]]
    if any(item == "type_3" for item in social):
        return (
            "experimentally assigned alternative early histories produce amplified "
            "and persistent directional differences in later collective states"
        )
    if any(item == "type_2" for item in social):
        return (
            "experimentally assigned alternative early histories produce persistent "
            "directional differences in later collective states"
        )
    if any(item == "type_1" for item in social):
        return (
            "symmetric alternative early histories produce persistent but nondirectional "
            "collective divergence under social feedback"
        )
    if all(item == "type_0" for item in outcomes.values()):
        return (
            "symmetric alternative early histories do not produce persistent collective "
            "divergence under the tested regimes"
        )
    return (
        "social-feedback regimes show no persistent A-B divergence; "
        "regime classes are reported separately and are not reclassified"
    )


def simulate_branch(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    spec: dict[str, Any],
    *,
    choice_seed_value: int,
    branch: str,
    timesteps: int = TIMESTEPS,
    on_step: Any = None,
) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise RuntimeError("unknown history branch")
    matrix = distance_matrix()
    places = place_index()
    memory = _make_memory(spec, list(locations))
    current = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    series = [dict(current)]
    moves = 0
    plan: dict[str, Any] | None = None
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
                        "llm_calls": 0,
                    }
                )
            if logged == INTERVENTION_STEP:
                mass = expected_choice_mass(actions)
                chosen_pair = select_history_pair(occupancy, mass)
                agents = select_agents(
                    spec["regime_id"],
                    choice_seed_value,
                    decisions,
                    chosen_pair["target_A"],
                    chosen_pair["target_B"],
                )
                plan = {
                    **chosen_pair,
                    "expected_choice_mass": mass,
                    "selected_agents": agents,
                    "control_proposed_destinations": [decisions[agent_id] for agent_id in agents],
                    "branch": branch,
                }
                target = None
                if branch == "HISTORY_A":
                    target = chosen_pair["target_A"]
                elif branch == "HISTORY_B":
                    target = chosen_pair["target_B"]
                if target is not None:
                    chosen = set(agents)
                    for agent_id in agents:
                        decisions[agent_id] = target
                    for action in actions:
                        if action["agent_id"] in chosen:
                            action["destination"] = target
                            action["stayed"] = action["destination"] == action["origin"]
                            action["intervention_applied"] = True
                    plan["effective_overrides"] = sum(action["intervention_applied"] for action in actions)
                    override_steps.append(logged)
                    if plan["effective_overrides"] != OVERRIDE_COUNT:
                        raise RuntimeError("alternative history did not override exactly two agents")
                else:
                    plan["effective_overrides"] = 0
            moves += sum(destination != current[agent_id] for agent_id, destination in decisions.items())
            current = {agent_id: decisions[agent_id] for agent_id in sorted(decisions)}
            memory.commit(decisions)
            series.append(dict(current))
            if on_step is not None:
                on_step(logged, current, memory, actions, plan if logged == INTERVENTION_STEP else None)
    if plan is None:
        raise RuntimeError("intervention step was not reached")
    return {
        "series": series,
        "moves": moves,
        "plan": plan,
        "override_steps": override_steps,
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
    locations = load_initial_locations()
    references = _reference_hashes()
    config = _config(seeds, timesteps)
    write_json(destination / "world_v0_8_config.json", config)
    names = (
        "world_states.jsonl",
        "social_memory_states.jsonl",
        "agent_actions.jsonl",
        "alternative_history_interventions.jsonl",
        "matched_history_metrics.jsonl",
    )
    paths = {name: destination / name for name in names}
    for path in paths.values():
        path.write_text("", encoding="utf-8", newline="\n")
    pairs = []
    branch_worlds = []
    with paths["world_states.jsonl"].open("a", encoding="utf-8", newline="\n") as states, paths[
        "social_memory_states.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as memories, paths["agent_actions.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as actions, paths["alternative_history_interventions.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as interventions, paths["matched_history_metrics.jsonl"].open("a", encoding="utf-8", newline="\n") as forks:
        for spec in REGIMES:
            for seed in seeds:
                runs = {
                    branch: _run_logged(source, locations, spec, seed, branch, states, memories, actions)
                    for branch in BRANCHES
                }
                _require_gates(spec, seed, runs, references)
                pair = _pair_metrics(spec, seed, runs)
                pairs.append(pair)
                for branch, result in runs.items():
                    branch_worlds.append(_branch_safety(spec, seed, branch, result))
                _dump(interventions, _intervention_record(spec, seed, runs, pair))
                _dump(forks, pair)
                print(
                    f"{spec['regime_id']} seed={seed} late_tv={pair['late_tv']:.4f} late_dhi={pair['late_dhi']:.4f}",
                    flush=True,
                )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    replays = _replay(source, locations, pairs)
    summary = _summarize(pairs, branch_worlds, replays)
    write_json(destination / "experiment_008_summary.json", summary)
    comparison = _compare(pairs, summary)
    write_json(destination / "regime_comparison.json", comparison)
    write_json(destination / "directional_history_analysis.json", _directional(pairs, summary))
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, pairs))
    return destination


def _run_logged(atlas, locations, spec, seed, branch, states, memories, actions) -> dict[str, Any]:
    _write_initial(states, memories, spec, seed, branch, locations)

    def write(logged, current, memory, step_actions, _plan):
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
            _dump(actions, {"regime_id": spec["regime_id"], "choice_seed": seed, **action})

    result = simulate_branch(
        atlas, locations, spec, choice_seed_value=seed, branch=branch, on_step=write
    )
    if result["llm_calls"] != 0:
        raise RuntimeError("experiment 008 called a model")
    return result


def _require_gates(spec, seed, runs, references) -> None:
    control, history_a, history_b = runs["CONTROL"], runs["HISTORY_A"], runs["HISTORY_B"]
    if control["series"][:INTERVENTION_STEP] != history_a["series"][:INTERVENTION_STEP]:
        raise RuntimeError(f"{spec['regime_id']} seed {seed} HISTORY_A diverged before t=21")
    if control["series"][:INTERVENTION_STEP] != history_b["series"][:INTERVENTION_STEP]:
        raise RuntimeError(f"{spec['regime_id']} seed {seed} HISTORY_B diverged before t=21")
    plans = [runs[branch]["plan"]["selected_agents"] for branch in BRANCHES]
    targets = [(runs[branch]["plan"]["target_A"], runs[branch]["plan"]["target_B"]) for branch in BRANCHES]
    if len(set(map(tuple, plans))) != 1 or len(set(targets)) != 1:
        raise RuntimeError(f"{spec['regime_id']} seed {seed} history plan diverged across branches")
    if control["override_steps"] != [] or history_a["override_steps"] != [INTERVENTION_STEP]:
        raise RuntimeError("HISTORY_A intervention was not confined to t=21")
    if history_b["override_steps"] != [INTERVENTION_STEP]:
        raise RuntimeError("HISTORY_B intervention was not confined to t=21")
    if history_a["plan"]["target_A"] == history_a["plan"]["target_B"]:
        raise RuntimeError("history targets are not distinct")
    if history_a["plan"]["current_dominant"] in {history_a["plan"]["target_A"], history_a["plan"]["target_B"]}:
        raise RuntimeError("current dominant was used as a history target")
    for left, right, label in (
        (control, history_a, "CONTROL-A"),
        (control, history_b, "CONTROL-B"),
        (history_a, history_b, "A-B"),
    ):
        tv = _tv(_share_of(left["series"][INTERVENTION_STEP]), _share_of(right["series"][INTERVENTION_STEP]))
        if abs(tv - INITIAL_TV) > 1e-12:
            raise RuntimeError(f"{label} t=21 occupancy TV is {tv}, not 0.10")
    expected = references[spec["regime_id"]].get(seed)
    if expected is not None and control["trajectory_sha256"] != expected:
        raise RuntimeError(f"{spec['regime_id']} CONTROL seed {seed} does not match the frozen reference trajectory")
    if spec["regime_id"] == "S0" and seed == 0 and control["trajectory_sha256"] != EXPECTED_S0_SEED0:
        raise RuntimeError("S0 seed 0 hash gate failed")


def _pair_metrics(spec: dict[str, Any], seed: int, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    control, history_a, history_b = runs["CONTROL"]["series"], runs["HISTORY_A"]["series"], runs["HISTORY_B"]["series"]
    plan = runs["CONTROL"]["plan"]
    target_a, target_b = plan["target_A"], plan["target_B"]
    agents = sorted(control[0])
    population = len(agents)
    tv_series, js_series, disagreement_series = [], [], []
    dhi_series, advantage_a, advantage_b = [], [], []
    for timestep in range(TIMESTEPS + 1):
        share_a = _share_of(history_a[timestep])
        share_b = _share_of(history_b[timestep])
        own_a = share_a[target_a] - share_b[target_a]
        own_b = share_b[target_b] - share_a[target_b]
        tv_series.append(_tv(share_a, share_b))
        js_series.append(_js_distance(share_a, share_b))
        disagreement_series.append(
            sum(history_a[timestep][agent_id] != history_b[timestep][agent_id] for agent_id in agents) / population
        )
        advantage_a.append(own_a)
        advantage_b.append(own_b)
        dhi_series.append(directional_history_index(share_a[target_a], share_b[target_a], share_b[target_b], share_a[target_b]))
    late_tv = statistics.fmean(tv_series[401:501])
    late_dhi = statistics.fmean(dhi_series[401:501])
    final_a = _unique_leader(_presence(history_a, 401, 500))
    final_b = _unique_leader(_presence(history_b, 401, 500))
    final_control = _unique_leader(_presence(control, 401, 500))
    control_a_tv = [_tv(_share_of(control[timestep]), _share_of(history_a[timestep])) for timestep in range(TIMESTEPS + 1)]
    control_b_tv = [_tv(_share_of(control[timestep]), _share_of(history_b[timestep])) for timestep in range(TIMESTEPS + 1)]
    control_a_js = [_js_distance(_share_of(control[timestep]), _share_of(history_a[timestep])) for timestep in range(401, 501)]
    control_b_js = [_js_distance(_share_of(control[timestep]), _share_of(history_b[timestep])) for timestep in range(401, 501)]
    reconverged, reconverge_time = _first_run(tv_series)
    agent_reconverged, agent_time = _first_run(disagreement_series)
    control_a_reconverged, control_a_time = _first_run(control_a_tv)
    control_b_reconverged, control_b_time = _first_run(control_b_tv)
    return {
        "regime_id": spec["regime_id"],
        "social_weight": spec["social_weight"],
        "choice_seed": seed,
        "current_dominant": plan["current_dominant"],
        "target_A": target_a,
        "target_B": target_b,
        "absolute_gap": plan["absolute_gap"],
        "relative_gap": plan["relative_gap"],
        "selected_agents": plan["selected_agents"],
        "initial_tv": tv_series[INTERVENTION_STEP],
        "late_tv": late_tv,
        "persistence_ratio": late_tv / INITIAL_TV,
        "pair_class": classify_persistence(late_tv),
        "late_dhi": late_dhi,
        "windows": {
            name: {
                "occupancy_tv": statistics.fmean(tv_series[start : stop + 1]),
                "js_distance": statistics.fmean(js_series[start : stop + 1]),
                "agent_disagreement": statistics.fmean(disagreement_series[start : stop + 1]),
                "directional_history_index": statistics.fmean(dhi_series[start : stop + 1]),
                "own_history_advantage_A": statistics.fmean(advantage_a[start : stop + 1]),
                "own_history_advantage_B": statistics.fmean(advantage_b[start : stop + 1]),
            }
            for name, start, stop in WINDOWS
        },
        "occupancy_tv": tv_series,
        "js_distance": js_series,
        "agent_disagreement": disagreement_series,
        "directional_history_index": dhi_series,
        "final_dominant_control": final_control,
        "final_dominant_A": final_a,
        "final_dominant_B": final_b,
        "final_dominant_disagreement": final_a != final_b,
        "control_a_disagreement": final_control != final_a,
        "control_b_disagreement": final_control != final_b,
        "own_target_A": final_a == target_a,
        "own_target_B": final_b == target_b,
        "control_a_final_late_tv": statistics.fmean(control_a_tv[401:501]),
        "control_b_final_late_tv": statistics.fmean(control_b_tv[401:501]),
        "control_a_final_late_js": statistics.fmean(control_a_js),
        "control_b_final_late_js": statistics.fmean(control_b_js),
        "control_a_ever_reconverged": control_a_reconverged,
        "control_a_time_to_reconvergence": control_a_time,
        "control_b_ever_reconverged": control_b_reconverged,
        "control_b_time_to_reconvergence": control_b_time,
        "ever_reconverged": reconverged,
        "time_to_reconvergence": reconverge_time,
        "agent_exact_reconverged": agent_reconverged,
        "agent_time_to_reconvergence": agent_time,
        "control_trajectory_sha256": runs["CONTROL"]["trajectory_sha256"],
        "history_a_trajectory_sha256": runs["HISTORY_A"]["trajectory_sha256"],
        "history_b_trajectory_sha256": runs["HISTORY_B"]["trajectory_sha256"],
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


def _summarize(pairs, worlds, replays) -> dict[str, Any]:
    outcomes = {}
    regimes = {}
    for spec in REGIMES:
        rows = [pair for pair in pairs if pair["regime_id"] == spec["regime_id"]]
        late_tv = statistics.fmean(pair["late_tv"] for pair in rows)
        late_dhi = statistics.fmean(pair["late_dhi"] for pair in rows)
        ci_low, ci_high = _bootstrap_ci([pair["late_dhi"] for pair in rows])
        outcomes[spec["regime_id"]] = classify_outcome(late_tv, late_dhi, ci_low)
        regimes[spec["regime_id"]] = _regime_summary(spec, rows, worlds, late_tv, late_dhi, ci_low, ci_high, outcomes[spec["regime_id"]])
    return {
        "experiment_id": "experiment_008_alternative_histories",
        "note": ANALYSIS_NOTE,
        "pair_rule": PAIR_RULE,
        "persistence_rule": PERSISTENCE_RULE,
        "direction_rule": DIRECTION_RULE,
        "taxonomy_rule": TAXONOMY_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
        "conclusion": overall_conclusion(outcomes),
        "outcome_type": outcomes,
        "normalization": CENTERED_NORMALIZATION,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "llm_calls": 0,
        "world_count": len(worlds),
        "pair_count": len(pairs),
        "timesteps": TIMESTEPS,
        "choice_seeds": list(CHOICE_SEEDS),
        "control_lineage_matches_reference": True,
        "js_definition": "Jensen-Shannon distance = sqrt(JS divergence), with natural log.",
        "final_dominant_window": "t=401-500 committed presence, unique maximum, tie unmarked",
        "regimes": regimes,
        "exact_replay": replays,
    }


def _regime_summary(spec, rows, worlds, late_tv, late_dhi, ci_low, ci_high, outcome) -> dict[str, Any]:
    return {
        "social_weight": spec["social_weight"],
        "n_pairs": len(rows),
        "llm_calls": 0,
        "absolute_gap": _dist([pair["absolute_gap"] for pair in rows]),
        "relative_gap": _dist([pair["relative_gap"] for pair in rows]),
        "initial_tv": _dist([pair["initial_tv"] for pair in rows]),
        "late_tv": _dist([pair["late_tv"] for pair in rows]),
        "regime_late_tv": late_tv,
        "persistence_ratio": late_tv / INITIAL_TV,
        "persistence_class": classify_persistence(late_tv),
        "pair_class_counts": {
            name: sum(pair["pair_class"] == name for pair in rows)
            for name in ("attenuated", "persistent residual", "amplified divergence")
        },
        "late_dhi": _dist([pair["late_dhi"] for pair in rows]),
        "regime_late_dhi": late_dhi,
        "late_dhi_ci95": [ci_low, ci_high],
        "directional_class": classify_direction(late_dhi, ci_low),
        "outcome_type": outcome,
        "windows": {
            name: {
                metric: _dist([pair["windows"][name][metric] for pair in rows])
                for metric in (
                    "occupancy_tv",
                    "js_distance",
                    "agent_disagreement",
                    "directional_history_index",
                    "own_history_advantage_A",
                    "own_history_advantage_B",
                )
            }
            for name, _start, _stop in WINDOWS
        },
        "final_dominant_disagreement_rate": statistics.fmean(float(pair["final_dominant_disagreement"]) for pair in rows),
        "own_target_winner_rate_A": statistics.fmean(float(pair["own_target_A"]) for pair in rows),
        "own_target_winner_rate_B": statistics.fmean(float(pair["own_target_B"]) for pair in rows),
        "own_target_winner_rate": statistics.fmean(
            (float(pair["own_target_A"]) + float(pair["own_target_B"])) / 2.0 for pair in rows
        ),
        "ever_reconverged_rate": statistics.fmean(float(pair["ever_reconverged"]) for pair in rows),
        "time_to_reconvergence": _dist([pair["time_to_reconvergence"] for pair in rows]),
        "agent_exact_reconverged_rate": statistics.fmean(float(pair["agent_exact_reconverged"]) for pair in rows),
        "control_comparisons": {
            "A": {
                "final_late_tv": _dist([pair["control_a_final_late_tv"] for pair in rows]),
                "final_late_js": _dist([pair["control_a_final_late_js"] for pair in rows]),
                "winner_disagreement_rate": statistics.fmean(float(pair["control_a_disagreement"]) for pair in rows),
                "ever_reconverged_rate": statistics.fmean(float(pair["control_a_ever_reconverged"]) for pair in rows),
            },
            "B": {
                "final_late_tv": _dist([pair["control_b_final_late_tv"] for pair in rows]),
                "final_late_js": _dist([pair["control_b_final_late_js"] for pair in rows]),
                "winner_disagreement_rate": statistics.fmean(float(pair["control_b_disagreement"]) for pair in rows),
                "ever_reconverged_rate": statistics.fmean(float(pair["control_b_ever_reconverged"]) for pair in rows),
            },
        },
        "branches": {
            branch: _safety_summary(
                [world for world in worlds if world["regime_id"] == spec["regime_id"] and world["branch"] == branch]
            )
            for branch in BRANCHES
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
        "lag1": {place_id: _dist([world["lag1"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "lag2": {place_id: _dist([world["lag2"][place_id] for world in worlds]) for place_id in PLACE_IDS},
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


def _compare(pairs, summary) -> dict[str, Any]:
    by_regime = {spec["regime_id"]: [pair for pair in pairs if pair["regime_id"] == spec["regime_id"]] for spec in REGIMES}
    return {
        "experiment_id": "experiment_008_alternative_histories",
        "note": "Paired by choice seed. Classification does not use p-values. Intervals use 10000 paired resamples.",
        "conclusion": summary["conclusion"],
        "outcome_type": summary["outcome_type"],
        "paired_contrasts": {
            "RW50_minus_S0": _paired(by_regime["S0"], by_regime["RW50"]),
            "EXP99_minus_S0": _paired(by_regime["S0"], by_regime["EXP99"]),
            "EXP99_minus_RW50": _paired(by_regime["RW50"], by_regime["EXP99"]),
        },
        "mean_trajectory": {
            regime_id: {
                "occupancy_tv": _column_mean(rows, "occupancy_tv"),
                "js_distance": _column_mean(rows, "js_distance"),
                "agent_disagreement": _column_mean(rows, "agent_disagreement"),
                "directional_history_index": _column_mean(rows, "directional_history_index"),
            }
            for regime_id, rows in by_regime.items()
        },
    }


def _directional(pairs, summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_008_alternative_histories",
        "direction_rule": DIRECTION_RULE,
        "taxonomy_rule": TAXONOMY_RULE,
        "conclusion": summary["conclusion"],
        "regimes": {
            regime_id: {
                "regime_late_dhi": summary["regimes"][regime_id]["regime_late_dhi"],
                "late_dhi_ci95": summary["regimes"][regime_id]["late_dhi_ci95"],
                "directional_class": summary["regimes"][regime_id]["directional_class"],
                "outcome_type": summary["regimes"][regime_id]["outcome_type"],
                "persistence_class": summary["regimes"][regime_id]["persistence_class"],
                "final_late_window": summary["regimes"][regime_id]["windows"]["final_late"],
                "target_balance": {
                    "absolute_gap": summary["regimes"][regime_id]["absolute_gap"],
                    "relative_gap": summary["regimes"][regime_id]["relative_gap"],
                },
            }
            for regime_id in summary["regimes"]
        },
    }


def _paired(left_rows, right_rows) -> dict[str, Any]:
    left = {row["choice_seed"]: row for row in left_rows}
    right = {row["choice_seed"]: row for row in right_rows}
    report = {}
    for metric in ("late_tv", "persistence_ratio", "late_dhi", "final_dominant_disagreement"):
        diffs = [float(right[seed][metric]) - float(left[seed][metric]) for seed in sorted(left)]
        report[metric] = _effect(diffs)
    return report


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


def _column_mean(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [statistics.fmean(row[key][index] for row in rows) for index in range(len(rows[0][key]))]


def _replay(atlas, locations, pairs) -> dict[str, Any]:
    requested = (("S0", "CONTROL", 0), ("RW50", "HISTORY_A", 0), ("EXP99", "HISTORY_B", 0))
    fields = {"CONTROL": "control_trajectory_sha256", "HISTORY_A": "history_a_trajectory_sha256", "HISTORY_B": "history_b_trajectory_sha256"}
    found = {}
    for regime_id, branch, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        result = simulate_branch(atlas, locations, spec, choice_seed_value=seed, branch=branch)
        original = next(pair[fields[branch]] for pair in pairs if pair["regime_id"] == regime_id and pair["choice_seed"] == seed)
        if result["trajectory_sha256"] != original:
            raise RuntimeError(f"{regime_id} {branch} seed {seed} replay mismatch")
        found[f"{regime_id}_{branch}_seed{seed}"] = {"trajectory_sha256": result["trajectory_sha256"], "match": True}
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


def _intervention_record(spec, seed, runs, pair) -> dict[str, Any]:
    plan = runs["CONTROL"]["plan"]
    control_a = _tv(_share_of(runs["CONTROL"]["series"][INTERVENTION_STEP]), _share_of(runs["HISTORY_A"]["series"][INTERVENTION_STEP]))
    control_b = _tv(_share_of(runs["CONTROL"]["series"][INTERVENTION_STEP]), _share_of(runs["HISTORY_B"]["series"][INTERVENTION_STEP]))
    return {
        "experiment_id": "experiment_008_alternative_histories",
        "regime": spec["regime_id"],
        "choice_seed": seed,
        "intervention_t": INTERVENTION_STEP,
        "current_dominant": plan["current_dominant"],
        "expected_choice_mass": plan["expected_choice_mass"],
        "target_A": plan["target_A"],
        "target_B": plan["target_B"],
        "expected_mass_A": plan["expected_mass_A"],
        "expected_mass_B": plan["expected_mass_B"],
        "absolute_gap": plan["absolute_gap"],
        "relative_gap": plan["relative_gap"],
        "selected_agents": plan["selected_agents"],
        "control_proposed_destinations": plan["control_proposed_destinations"],
        "override_count_A": runs["HISTORY_A"]["plan"]["effective_overrides"],
        "override_count_B": runs["HISTORY_B"]["plan"]["effective_overrides"],
        "initial_tv_control_A": control_a,
        "initial_tv_control_B": control_b,
        "initial_tv_A_B": pair["initial_tv"],
        "llm_calls": 0,
    }


def _first_run(values: list[float]) -> tuple[bool, int | None]:
    run = 0
    for timestep in range(INTERVENTION_STEP, len(values)):
        if values[timestep] == 0.0:
            run += 1
            if run >= RECONVERGENCE_RUN:
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


def _relative_gap(left: float, right: float) -> float | None:
    midpoint = (left + right) / 2.0
    if midpoint == 0.0:
        return None
    return abs(left - right) / midpoint


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
        "world_id": "world_v0_8",
        "experiment_id": "experiment_008_alternative_histories",
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
        "override_count": OVERRIDE_COUNT,
        "initial_occupancy_tv": INITIAL_TV,
        "directional_threshold": DIRECTIONAL_THRESHOLD,
        "hash_material": HASH_MATERIAL,
        "windows": [{"name": name, "start": start, "stop": stop} for name, start, stop in WINDOWS],
        "pair_rule": PAIR_RULE,
        "persistence_rule": PERSISTENCE_RULE,
        "direction_rule": DIRECTION_RULE,
        "taxonomy_rule": TAXONOMY_RULE,
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


def _manifest(destination, config, summary, pairs) -> dict[str, Any]:
    trajectories = {}
    for pair in pairs:
        trajectories[f"{pair['regime_id']}:{pair['choice_seed']}:CONTROL"] = pair["control_trajectory_sha256"]
        trajectories[f"{pair['regime_id']}:{pair['choice_seed']}:HISTORY_A"] = pair["history_a_trajectory_sha256"]
        trajectories[f"{pair['regime_id']}:{pair['choice_seed']}:HISTORY_B"] = pair["history_b_trajectory_sha256"]
    return {
        "experiment_id": "experiment_008_alternative_histories",
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "pair_count": summary["pair_count"],
        "complete": summary["world_count"] == len(REGIMES) * len(CHOICE_SEEDS) * len(BRANCHES),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "s0_seed0_sha256": EXPECTED_S0_SEED0,
        "outcome_type": summary["outcome_type"],
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": trajectories,
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_8_config.json",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "alternative_history_interventions.jsonl",
                "matched_history_metrics.jsonl",
                "experiment_008_summary.json",
                "regime_comparison.json",
                "directional_history_analysis.json",
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
            WORLD_V07_DIR,
            ATLAS_DIR,
            CALIBRATION_DIR,
        )
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 008 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_008")
    main()
