"""Experiment 012: feedback-autonomy dose response.

From t=22 the decision social score is a convex blend of a shared midpoint
field and the recipient branch's own endogenous score. Lambda is a simulation
control parameter, not a claim about human autonomy. This module does not call
a model and does not retune rules after seeing results.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from array import array
from contextlib import contextmanager
from pathlib import Path
from random import Random
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

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_12"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
WORLD_V10_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_10"
WORLD_V11_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_11"
DONOR_PATH = WORLD_V11_DIR / "donor_social_fields.jsonl"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
EXPECTED_S0_SEED0 = "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
EXPECTED_RW50_PERTURBED_SEED0 = "57a31820b38472a41a15a79dd49af2bbd8012cab13e672bf6190c8953e2a5939"
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
INTERVENTION_STEP = 21
ENVIRONMENT_STEP = 22
FIELD_STEPS = tuple(range(ENVIRONMENT_STEP, TIMESTEPS + 1))
OVERRIDE_COUNT = 2
ATTENUATION_RUN = 20
RECONVERGENCE_RUN = 20
ANALYSIS_BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 10000
BLEND_TOLERANCE = 1e-12
LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
HISTORIES = ("CONTROL", "PERTURBED")
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
MIDPOINT_NOTE = (
    "common_midpoint_field is an experimental simulation construct, not a naturally observed social field."
)
AUTONOMY_NOTE = (
    "Lambda is a simulation control parameter mixing a shared midpoint field with branch-specific "
    "endogenous social scores. It is not human autonomy, psychological independence, or individualism. "
    "These runs do not establish a phase transition, a critical point, collective path dependence, "
    "a human tipping point, or a real-world threshold."
)
ANALYSIS_NOTE = (
    "Experiment 011 showed that early-history persistence requires the two histories to receive different "
    "branch-endogenous social feedback trajectories under the tested frozen architecture. "
    "Experiment 012 varies that autonomy on a fixed grid. " + MIDPOINT_NOTE + " " + AUTONOMY_NOTE
)
ONSET_RULE = (
    "Persistence onset is the first tested lambda whose regime-mean late TV is at least 0.025, "
    "provided every higher tested lambda also stays at least 0.025. "
    "The band is (previous lambda, onset lambda]. It is an interval on the tested grid, not an interpolated threshold. "
    "If a higher lambda attenuates after a lower one persists, the pattern is nonmonotonic and no onset is assigned. "
    "If only lambda 1.00 persists, the pattern is FULL_AUTONOMY_BOUNDARY. "
    "If the first persistence lambda is below 1 and all higher lambdas persist, the pattern is GRADED_ONSET. "
    "If lambda 0 persists, the pattern is COMMON_MIDPOINT_PERSISTENCE and no threshold is claimed."
)
CONCLUSION_RULE = (
    "If both social regimes are FULL_AUTONOMY_BOUNDARY: persistent history divergence was observed only under complete "
    "branch-specific social feedback autonomy among the tested levels. "
    "If both are GRADED_ONSET: report each tested onset band and do not force the bands to match. "
    "If the regimes differ: the autonomy level at which history persistence appears depends on social-memory architecture. "
    "If either is nonmonotonic: history persistence depends non-monotonically on feedback autonomy, so no single onset threshold is assigned. "
    "If lambda 0 persists: the synthetic shared midpoint field itself supports persistent history divergence, "
    "unlike the donor-specific shared fields tested in Experiment 011."
)
ATTENUATION_RULE = (
    "Stable attenuation is the first completion of 20 consecutive timesteps with history-pair occupancy TV below 0.025, "
    "counted only from t=22. The recorded time is the timestep that completes that run. Otherwise null."
)
RECONVERGENCE_RULE = (
    "Reconvergence is 20 consecutive timesteps with history-pair occupancy TV exactly 0, counted only from t=21. "
    "It does not replace the persistence classification."
)
BOUNDARY_SENTENCE = (
    "persistent history divergence was observed only under complete branch-specific social feedback autonomy among the tested levels"
)
NONMONOTONIC_SENTENCE = (
    "history persistence depends non-monotonically on feedback autonomy, so no single onset threshold is assigned"
)
MIDPOINT_PERSISTENCE_SENTENCE = (
    "the synthetic shared midpoint field itself supports persistent history divergence, unlike the donor-specific shared fields tested in Experiment 011"
)
ARCHITECTURE_SENTENCE = (
    "the autonomy level at which history persistence appears depends on social-memory architecture"
)
ADJACENT = tuple(zip(LAMBDAS, LAMBDAS[1:], strict=False))


@contextmanager
def _weight(value: float) -> Iterator[None]:
    original = memory_experiment.SOCIAL_WEIGHT
    memory_experiment.SOCIAL_WEIGHT = float(value)
    try:
        yield
    finally:
        memory_experiment.SOCIAL_WEIGHT = original


def classify_autonomy(levels: list[float]) -> dict[str, Any]:
    if len(levels) != len(LAMBDAS):
        raise RuntimeError("autonomy classification requires the six fixed lambda levels")
    persists = [value >= 0.025 for value in levels]
    if persists[0]:
        return {"autonomy_pattern": "COMMON_MIDPOINT_PERSISTENCE", "persistence_onset_lambda": None, "persistence_onset_band": None}
    seen = False
    for flag in persists:
        if seen and not flag:
            return {"autonomy_pattern": "nonmonotonic", "persistence_onset_lambda": None, "persistence_onset_band": None}
        if flag:
            seen = True
    if not seen:
        return {"autonomy_pattern": "NO_PERSISTENCE_ON_TESTED_GRID", "persistence_onset_lambda": None, "persistence_onset_band": None}
    index = persists.index(True)
    onset = LAMBDAS[index]
    band = None if index == 0 else {"lower_exclusive": LAMBDAS[index - 1], "upper_inclusive": onset, "label": f"({_lambda_label(LAMBDAS[index - 1])}, {_lambda_label(onset)}]"}
    pattern = "FULL_AUTONOMY_BOUNDARY" if index == len(LAMBDAS) - 1 else "GRADED_ONSET"
    return {"autonomy_pattern": pattern, "persistence_onset_lambda": onset, "persistence_onset_band": band}


def overall_conclusion(patterns: dict[str, str], bands: dict[str, str | None]) -> str:
    names = [patterns["RW50"], patterns["EXP99"]]
    if "nonmonotonic" in names:
        return NONMONOTONIC_SENTENCE
    if "COMMON_MIDPOINT_PERSISTENCE" in names:
        return MIDPOINT_PERSISTENCE_SENTENCE
    if names[0] == names[1] == "FULL_AUTONOMY_BOUNDARY":
        return BOUNDARY_SENTENCE
    if names[0] == names[1] == "GRADED_ONSET":
        if bands["RW50"] == bands["EXP99"]:
            return f"history persistence shows a graded onset within the tested autonomy interval {bands['RW50']}"
        return (
            f"RW50 persistence onset band is {bands['RW50']}. EXP99 persistence onset band is {bands['EXP99']}. "
            + ARCHITECTURE_SENTENCE
        )
    return ARCHITECTURE_SENTENCE


def trajectory_shape(levels: list[float]) -> str:
    if levels[0] >= 0.025:
        return "the cell trajectory is reported without a threshold claim because the shared midpoint itself persists"
    if all(level < 0.025 for level in levels[:-1]) and levels[-1] >= 0.025:
        return "threshold-like / nonlinear pattern"
    diffs = [right - left for left, right in zip(levels, levels[1:], strict=False)]
    if all(diff >= -1e-12 for diff in diffs) and levels[-1] > levels[0] and any(level >= 0.025 for level in levels[:-1]):
        return "approximately graded over the tested levels"
    return "the cell trajectory is reported without a linear dose-response claim"


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


def _lambda_label(value: float) -> str:
    return f"{value:.2f}"


def midpoint_scores(control: dict[str, float], perturbed: dict[str, float]) -> dict[str, float]:
    return {place_id: 0.5 * (float(control[place_id]) + float(perturbed[place_id])) for place_id in PLACE_IDS}


def blend_scores(midpoint: dict[str, float], endogenous: dict[str, float], lambda_value: float) -> dict[str, float]:
    if lambda_value == 0.0:
        return {place_id: float(midpoint[place_id]) for place_id in PLACE_IDS}
    if lambda_value == 1.0:
        return {place_id: float(endogenous[place_id]) for place_id in PLACE_IDS}
    return {
        place_id: (1.0 - lambda_value) * float(midpoint[place_id]) + lambda_value * float(endogenous[place_id])
        for place_id in PLACE_IDS
    }


def simulate_history(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    spec: dict[str, Any],
    *,
    choice_seed_value: int,
    history: str,
    lambda_value: float,
    plan: dict[str, Any],
    donors: dict[str, dict[int, dict[str, dict[str, float]]]],
    timesteps: int = TIMESTEPS,
    on_step: Any = None,
) -> dict[str, Any]:
    if history not in HISTORIES:
        raise RuntimeError("unknown history")
    if lambda_value not in LAMBDAS:
        raise RuntimeError("lambda is outside the fixed grid")
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
    max_blend_error = 0.0
    lambda0_field_mismatch = False
    shadow_scores_differed = False
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
                bases, _totals, social_term = _utilities(
                    atlas, matrix, places, agent_id, origin, occupancy, endogenous
                )
                endogenous_agent = {place_id: float(endogenous[place_id]) for place_id in PLACE_IDS}
                endogenous_scores[agent_id] = endogenous_agent
                used_scores = dict(endogenous_agent)
                score_source = "endogenous"
                if logged >= ENVIRONMENT_STEP:
                    control_donor = donors["CONTROL"][logged][agent_id]
                    perturbed_donor = donors["PERTURBED"][logged][agent_id]
                    midpoint_agent = midpoint_scores(control_donor, perturbed_donor)
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
                    draw = Random(choice_seed(choice_seed_value, timestep, agent_id)).random()
                    destination, probabilities = _select(
                        list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw
                    )
                    _base_destination, _unused = _select(
                        list(PLACE_IDS), [bases[place_id] for place_id in PLACE_IDS], draw
                    )
                    destination_changed = destination != _base_destination
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
                        "social_weight": weight,
                        "lambda": lambda_value,
                        "history": history,
                        "score_source": score_source,
                        "intervention_applied": False,
                        "popularity_score": used_scores,
                        "social_utility": social_term,
                        "ranking_changed": ranking_changed,
                        "destination_changed": destination_changed,
                        "choice_probabilities": probabilities,
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
            ordered_agents = sorted(step_scores)
            used_trace.append(tuple(tuple(step_scores[agent_id][place_id] for place_id in PLACE_IDS) for agent_id in ordered_agents))
            endo_trace.append(tuple(tuple(endogenous_scores[agent_id][place_id] for place_id in PLACE_IDS) for agent_id in ordered_agents))
            mid_trace.append(
                None
                if not midpoint
                else tuple(tuple(midpoint[agent_id][place_id] for place_id in PLACE_IDS) for agent_id in ordered_agents)
            )
            rng_draw_counts.append(step_rng)
            max_blend_error = max(max_blend_error, step_error)
            if logged >= ENVIRONMENT_STEP:
                decision_hash = canonical_sha256(step_scores)
                endogenous_hash = canonical_sha256(endogenous_scores)
                midpoint_hash = canonical_sha256(midpoint)
                if endogenous_hash != decision_hash:
                    shadow_scores_differed = True
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
            _extend_hash(dynamics, _step_payload(logged, current, actions, memory))
    if lambda0_field_mismatch:
        raise RuntimeError("lambda 0 decision field is not the shared midpoint")
    if lambda_value == 1.0 and donor_mismatch > BLEND_TOLERANCE:
        raise RuntimeError("lambda 1 endogenous field does not match the Experiment 011 donor")
    return {
        "series": series,
        "moves": _movement(series),
        "probabilities": probability_trace,
        "used_trace": used_trace,
        "endo_trace": endo_trace,
        "mid_trace": mid_trace,
        "rng_draw_counts": rng_draw_counts,
        "max_blend_error": max_blend_error,
        "shadow_scores_differed": shadow_scores_differed,
        "override_steps": [INTERVENTION_STEP] if history == "PERTURBED" else [],
        "override_count": OVERRIDE_COUNT if history == "PERTURBED" else 0,
        "canonical_dynamics_sha256": dynamics.hexdigest(),
        "trajectory_sha256": trajectory_hash(series),
        "llm_calls": 0,
    }


class DonorStore:
    def __init__(self, agent_ids: list[str]) -> None:
        self.agent_ids = list(agent_ids)
        self.agent_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}
        self.regime_index = {spec["regime_id"]: index for index, spec in enumerate(REGIMES)}
        self.history_index = {"CONTROL_ENDOGENOUS": 0, "PERTURBED_ENDOGENOUS": 1}
        self.place_index = {place_id: index for index, place_id in enumerate(PLACE_IDS)}
        self.steps = len(FIELD_STEPS)
        self._block = 50 * 2 * self.steps * len(self.agent_ids) * len(PLACE_IDS)
        self.data = array("d", [float("nan")] * (len(REGIMES) * self._block))
        self.present = bytearray(len(REGIMES) * self._block)
        self.rows = 0

    def add(self, regime: str, seed: int, donor: str, timestep: int, agent_id: str, scores: dict[str, float]) -> None:
        base = self._offset(regime, seed, donor, timestep, agent_id, PLACE_IDS[0])
        for place_id in PLACE_IDS:
            value = float(scores[place_id])
            if value < -2.0 or value > 2.0:
                raise RuntimeError("donor score is outside the clipped range")
            slot = base + self.place_index[place_id]
            if self.present[slot]:
                raise RuntimeError("duplicate donor score")
            self.data[slot] = value
            self.present[slot] = 1
        self.rows += 1

    def field(self, regime: str, seed: int, history: str) -> dict[int, dict[str, dict[str, float]]]:
        donor = "CONTROL_ENDOGENOUS" if history == "CONTROL" else "PERTURBED_ENDOGENOUS"
        result: dict[int, dict[str, dict[str, float]]] = {}
        for timestep in FIELD_STEPS:
            agents = {}
            for agent_id in self.agent_ids:
                base = self._offset(regime, seed, donor, timestep, agent_id, PLACE_IDS[0])
                agents[agent_id] = {place_id: self.data[base + self.place_index[place_id]] for place_id in PLACE_IDS}
            result[timestep] = agents
        return result

    def require_complete(self, seeds: tuple[int, ...] = CHOICE_SEEDS) -> None:
        expected = len(REGIMES) * len(seeds) * 2 * self.steps * len(self.agent_ids)
        if self.rows != expected or any(self.present[index] == 0 for index in range(len(self.present))):
            raise RuntimeError("Experiment 011 donor field is incomplete")

    def _offset(self, regime: str, seed: int, donor: str, timestep: int, agent_id: str, place_id: str) -> int:
        if timestep < ENVIRONMENT_STEP or timestep > TIMESTEPS:
            raise RuntimeError("donor timestep is outside t=22-500")
        return (
            self.regime_index[regime] * self._block
            + (
                ((seed * 2 + self.history_index[donor]) * self.steps + (timestep - ENVIRONMENT_STEP)) * len(self.agent_ids)
                + self.agent_index[agent_id]
            )
            * len(PLACE_IDS)
            + self.place_index[place_id]
        )


def load_donor_store(agent_ids: list[str], path: Path = DONOR_PATH, wanted: set[tuple[str, int]] | None = None) -> DonorStore:
    store = DonorStore(agent_ids)
    pending = None if wanted is None else set(wanted)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["regime"], int(row["choice_seed"]))
            if pending is not None and key not in pending and key[0] not in {item[0] for item in pending}:
                continue
            if pending is not None and key not in pending:
                continue
            store.add(key[0], key[1], row["donor"], int(row["timestep"]), row["agent_id"], row["scores"])
            if pending is not None and store.rows == len(pending) * 2 * store.steps * len(agent_ids):
                break
    if pending is None:
        store.require_complete()
    elif store.rows != len(pending) * 2 * store.steps * len(agent_ids):
        raise RuntimeError("requested donor rows are missing")
    return store


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
    free = shutil.disk_usage(destination if destination.exists() else destination.parent).free
    if free < 8_000_000_000:
        raise RuntimeError("not enough free disk for experiment 012")
    destination.mkdir(parents=True, exist_ok=True)
    source = load_frozen_atlas() if atlas is None else atlas
    _require_atlas_hash()
    initial = load_initial_locations()
    plans = load_intervention_plans()
    references = _reference_hashes()
    reference_011 = _exp011_reference()
    print("loading Experiment 011 donor fields", flush=True)
    donors = load_donor_store(sorted(initial))
    config = _config(seeds, timesteps, reference_011)
    write_json(destination / "world_v0_12_config.json", config)
    names = (
        "world_states.jsonl",
        "social_memory_states.jsonl",
        "agent_actions.jsonl",
        "common_midpoint_field.jsonl",
        "feedback_blend_audit.jsonl",
        "autonomy_metrics.jsonl",
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
    ) as actions, paths["common_midpoint_field.jsonl"].open("a", encoding="utf-8", newline="\n") as midpoint_handle, paths[
        "feedback_blend_audit.jsonl"
    ].open("a", encoding="utf-8", newline="\n") as audits, paths["autonomy_metrics.jsonl"].open(
        "a", encoding="utf-8", newline="\n"
    ) as rows:
        for spec in REGIMES:
            for seed in seeds:
                record, worlds = _run_seed(
                    source,
                    initial,
                    spec,
                    seed,
                    plans[(spec["regime_id"], seed)],
                    donors,
                    references,
                    states,
                    memories,
                    actions,
                    midpoint_handle,
                    audits,
                )
                metrics.append(record)
                branch_worlds.extend(worlds)
                _dump(rows, record)
                print(
                    f"{spec['regime_id']} seed={seed} "
                    + " ".join(f"{_lambda_label(item['lambda'])}={item['D_lambda']:.4f}" for item in record["lambdas"]),
                    flush=True,
                )
    if memory_experiment.SOCIAL_WEIGHT != 150.0:
        raise RuntimeError("social weight override was not restored")
    replays = _replay(source, initial, plans, donors, metrics)
    summary = _summarize(metrics, branch_worlds, replays, reference_011)
    write_json(destination / "experiment_012_summary.json", summary)
    write_json(destination / "regime_comparison.json", _compare(summary))
    write_json(destination / "autonomy_dose_response.json", _dose(summary))
    write_json(destination / "persistence_onset_analysis.json", _onset(summary))
    write_json(destination / "run_manifest.json", _manifest(destination, config, summary, metrics))
    return destination


def _run_seed(atlas, initial, spec, seed, plan, donors, references, states, memories, actions, midpoint_handle, audits):
    pair = {
        "CONTROL": donors.field(spec["regime_id"], seed, "CONTROL"),
        "PERTURBED": donors.field(spec["regime_id"], seed, "PERTURBED"),
    }
    _write_midpoint(midpoint_handle, spec, seed, pair)
    runs: dict[tuple[float, str], dict[str, Any]] = {}
    worlds = []
    for lambda_value in LAMBDAS:
        for history in HISTORIES:
            result = _run_logged(
                atlas, initial, spec, seed, history, lambda_value, plan, pair, states, memories, actions, audits
            )
            if lambda_value == 1.0:
                label = "CONTROL" if history == "CONTROL" else "PERTURBED"
                expected = references[f"{spec['regime_id']}:{seed}:{label}_ENDOGENOUS"]
                if result["trajectory_sha256"] != expected:
                    raise RuntimeError(f"{spec['regime_id']} seed {seed} {history} lambda 1 does not match Experiment 011")
            runs[(lambda_value, history)] = result
            worlds.append(_branch_safety(spec, seed, history, lambda_value, result))
    _require_seed_gates(spec, seed, runs)
    record = _seed_metrics(spec, seed, runs, plan)
    for result in runs.values():
        result.pop("probabilities", None)
        result.pop("series", None)
        result.pop("used_trace", None)
        result.pop("endo_trace", None)
        result.pop("mid_trace", None)
    return record, worlds


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
        raise RuntimeError("experiment 012 called a model")
    return result


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
        "shadow_memory_hash": audit["memory_hash"],
        "decision_field_hash": audit["decision_hash"],
        "llm_calls": 0,
    }


def _write_midpoint(handle, spec, seed, donors) -> None:
    for timestep in FIELD_STEPS:
        control = donors["CONTROL"][timestep]
        perturbed = donors["PERTURBED"][timestep]
        control_hash = canonical_sha256(control)
        perturbed_hash = canonical_sha256(perturbed)
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
                    "control_donor_hash": control_hash,
                    "perturbed_donor_hash": perturbed_hash,
                    "renormalized": False,
                    "reclipped": False,
                    "llm_calls": 0,
                },
            )


def _require_seed_gates(spec, seed, runs) -> None:
    base = runs[(LAMBDAS[0], "CONTROL")]
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
        same_history = runs[(1.0, history)]["series"][:ENVIRONMENT_STEP]
        if result["series"][:ENVIRONMENT_STEP] != same_history:
            raise RuntimeError("environment blending changed the pre-t22 history")
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
        field_onset = _first_true(contrast["field_mad"], lambda value: value > BLEND_TOLERANCE)
        occupancy_onset = _first_true(contrast["occupancy_tv"], lambda value: value >= 0.025)
        levels.append(
            {
                "lambda": lambda_value,
                "lambda_label": _lambda_label(lambda_value),
                "shared_fraction": 1.0 - lambda_value,
                "autonomous_fraction": lambda_value,
                "D_lambda": late_tv,
                "persistence_class": classify_persistence(late_tv),
                "late_js": statistics.fmean(contrast["js_distance"][401:501]),
                "late_agent_disagreement": statistics.fmean(contrast["agent_disagreement"][401:501]),
                "late_probability_tv": statistics.fmean(contrast["probability_tv"][401:501]),
                "late_field_mad": statistics.fmean(contrast["field_mad"][401:501]),
                "late_endogenous_field_mad": statistics.fmean(contrast["endogenous_field_mad"][401:501]),
                "late_midpoint_mad": statistics.fmean(contrast["midpoint_mad"][401:501]),
                "windows": _windows(contrast),
                "stable_attenuation_time": stable_attenuation_time(contrast["occupancy_tv"]),
                "ever_reconverged": reconvergence_time(contrast["occupancy_tv"]) is not None,
                "time_to_reconvergence": reconvergence_time(contrast["occupancy_tv"]),
                "field_divergence_onset_t": field_onset,
                "occupancy_divergence_onset_t": occupancy_onset,
                "feedback_divergence_latency": None if field_onset is None or occupancy_onset is None else occupancy_onset - field_onset,
                "control_trajectory_sha256": runs[(lambda_value, "CONTROL")]["trajectory_sha256"],
                "perturbed_trajectory_sha256": runs[(lambda_value, "PERTURBED")]["trajectory_sha256"],
                "occupancy_tv": contrast["occupancy_tv"],
                "field_mad": contrast["field_mad"],
            }
        )
    return {
        "regime_id": spec["regime_id"],
        "choice_seed": seed,
        "target_place": plan["target_place"],
        "selected_agents": list(plan["selected_agents"]),
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
        "midpoint_mad": [_pair_anchor(perturbed, control, timestep) for timestep in range(len(perturbed["series"]))],
    }


def _pair_anchor(perturbed, control, timestep: int) -> float:
    return statistics.fmean(
        (
            _anchor_mad(perturbed["used_trace"][timestep], perturbed["mid_trace"][timestep]),
            _anchor_mad(control["used_trace"][timestep], control["mid_trace"][timestep]),
        )
    )


def _trace_mad(left, right) -> float:
    if left is None or right is None:
        return 0.0
    total = 0.0
    count = 0
    for left_agent, right_agent in zip(left, right, strict=True):
        for left_value, right_value in zip(left_agent, right_agent, strict=True):
            total += abs(left_value - right_value)
            count += 1
    return total / count


def _anchor_mad(used, midpoint) -> float:
    if used is None or midpoint is None:
        return 0.0
    total = 0.0
    count = 0
    for used_agent, midpoint_agent in zip(used, midpoint, strict=True):
        for used_value, midpoint_value in zip(used_agent, midpoint_agent, strict=True):
            total += abs(used_value - midpoint_value)
            count += 1
    return total / count


def _windows(contrast) -> dict[str, dict[str, float]]:
    return {
        name: {
            "occupancy_tv": statistics.fmean(contrast["occupancy_tv"][start : stop + 1]),
            "js_distance": statistics.fmean(contrast["js_distance"][start : stop + 1]),
            "agent_disagreement": statistics.fmean(contrast["agent_disagreement"][start : stop + 1]),
            "probability_tv": statistics.fmean(contrast["probability_tv"][start : stop + 1]),
            "field_mad": statistics.fmean(contrast["field_mad"][start : stop + 1]),
        }
        for name, start, stop in WINDOWS
    }


def _branch_safety(spec, seed, history, lambda_value, result) -> dict[str, Any]:
    diagnostics = measure_world(result["series"], moves=result["moves"], timesteps=TIMESTEPS)
    late_share = _share(_presence(result["series"], 401, 500))
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


def _summarize(metrics, worlds, replays, reference_011) -> dict[str, Any]:
    regimes = {
        spec["regime_id"]: _regime_summary(spec, [row for row in metrics if row["regime_id"] == spec["regime_id"]], worlds)
        for spec in REGIMES
    }
    patterns = {}
    bands = {}
    for regime_id in ("RW50", "EXP99"):
        block = regimes[regime_id]
        classified = classify_autonomy([block["levels"][_lambda_label(value)]["D_lambda"] for value in LAMBDAS])
        block.update(classified)
        patterns[regime_id] = classified["autonomy_pattern"]
        bands[regime_id] = None if classified["persistence_onset_band"] is None else classified["persistence_onset_band"]["label"]
        block["trajectory_shape"] = trajectory_shape([block["levels"][_lambda_label(value)]["D_lambda"] for value in LAMBDAS])
    return {
        "experiment_id": "experiment_012_feedback_autonomy",
        "note": ANALYSIS_NOTE,
        "midpoint_note": MIDPOINT_NOTE,
        "onset_rule": ONSET_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "attenuation_rule": ATTENUATION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
        "experiment_011_reference": reference_011,
        "autonomy_pattern": patterns,
        "persistence_onset_band": bands,
        "conclusion": overall_conclusion(patterns, bands),
        "gates": {
            "lambda1_lineage": True,
            "lambda0_shared_field": True,
            "blend_identity": True,
            "s0_lambda_equivalence": True,
            "donor_integrity": True,
            "pairs_checked": len(metrics),
        },
        "llm_calls": 0,
        "world_count": len(worlds),
        "seed_count": len(metrics),
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
        mean_late = statistics.fmean(late)
        levels[label] = {
            "lambda": lambda_value,
            "D_lambda": mean_late,
            "persistence_class": classify_persistence(mean_late),
            "late_tv": _dist(late),
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
            "feedback_divergence_latency": _dist([item["feedback_divergence_latency"] for item in selected]),
            "field_divergence_onset_t": _dist([item["field_divergence_onset_t"] for item in selected]),
            "occupancy_divergence_onset_t": _dist([item["occupancy_divergence_onset_t"] for item in selected]),
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
    return {
        "social_weight": spec["social_weight"],
        "n_seeds": len(rows),
        "llm_calls": 0,
        "levels": levels,
        "adjacent_contrasts": {
            f"D_{_lambda_label(right)}_minus_D_{_lambda_label(left)}": _effect(
                [_level(row, right)["D_lambda"] - _level(row, left)["D_lambda"] for row in rows]
            )
            for left, right in ADJACENT
        },
        "spearman_lambda_late_tv": _spearman(
            [lambda_value for row in rows for lambda_value in LAMBDAS],
            [_level(row, lambda_value)["D_lambda"] for row in rows for lambda_value in LAMBDAS],
        ),
    }


def _level(row, lambda_value: float) -> dict[str, Any]:
    return next(item for item in row["lambdas"] if item["lambda"] == lambda_value)


def _safety_summary(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    lag1 = {place_id: statistics.fmean(world["lag1"][place_id] for world in worlds) for place_id in ("site_01", "site_02")}
    late_hhi = statistics.fmean(world["late_hhi"] for world in worlds)
    dominant = statistics.fmean(world["late_dominant_share"] for world in worlds)
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


def _compare(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_012_feedback_autonomy",
        "note": "Lambda is categorical. Spearman association and adjacent paired contrasts are secondary and do not assign onset.",
        "conclusion": summary["conclusion"],
        "autonomy_pattern": summary["autonomy_pattern"],
        "persistence_onset_band": summary["persistence_onset_band"],
        "regimes": {
            regime_id: {
                "D_lambda": {label: block["levels"][label]["D_lambda"] for label in block["levels"]},
                "persistence_class": {label: block["levels"][label]["persistence_class"] for label in block["levels"]},
                "late_field_mad": {label: block["levels"][label]["late_field_mad"]["mean"] for label in block["levels"]},
                "adjacent_contrasts": block["adjacent_contrasts"],
                "spearman_lambda_late_tv": block["spearman_lambda_late_tv"],
                "trajectory_shape": block.get("trajectory_shape"),
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


def _dose(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_012_feedback_autonomy",
        "primary": "lambda is treated as six categorical cells",
        "secondary": "Spearman association and adjacent paired bootstrap contrasts do not decide onset",
        "conclusion": summary["conclusion"],
        "regimes": _compare(summary)["regimes"],
    }


def _onset(summary) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_012_feedback_autonomy",
        "onset_rule": ONSET_RULE,
        "autonomy_pattern": summary["autonomy_pattern"],
        "persistence_onset_band": summary["persistence_onset_band"],
        "conclusion": summary["conclusion"],
        "regimes": {
            regime_id: {
                "autonomy_pattern": block.get("autonomy_pattern"),
                "persistence_onset_lambda": block.get("persistence_onset_lambda"),
                "persistence_onset_band": block.get("persistence_onset_band"),
                "trajectory_shape": block.get("trajectory_shape"),
                "D_lambda": {label: level["D_lambda"] for label, level in block["levels"].items()},
                "persistence_class": {label: level["persistence_class"] for label, level in block["levels"].items()},
            }
            for regime_id, block in summary["regimes"].items()
        },
    }


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
    sampler = Random(ANALYSIS_BOOTSTRAP_SEED)
    means = []
    size = len(values)
    for _draw in range(BOOTSTRAP_DRAWS):
        means.append(statistics.fmean(values[sampler.randrange(size)] for _index in range(size)))
    means.sort()
    return means[int(0.025 * (BOOTSTRAP_DRAWS - 1))], means[int(0.975 * (BOOTSTRAP_DRAWS - 1))]


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(values):
        end = cursor
        while end + 1 < len(values) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end) / 2 + 1
        for index in range(cursor, end + 1):
            ranks[order[index]] = rank
        cursor = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    scale_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    scale_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if scale_x == 0.0 or scale_y == 0.0:
        return None
    return numerator / (scale_x * scale_y)


def _replay(atlas, initial, plans, donors, metrics) -> dict[str, Any]:
    requested = (("S0", "CONTROL", 1.0), ("RW50", "PERTURBED", 1.0), ("EXP99", "CONTROL", 0.4))
    found = {}
    for regime_id, history, lambda_value in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        plan = plans[(regime_id, 0)]
        pair = {
            "CONTROL": donors.field(regime_id, 0, "CONTROL"),
            "PERTURBED": donors.field(regime_id, 0, "PERTURBED"),
        }
        result = simulate_history(
            atlas,
            initial,
            spec,
            choice_seed_value=0,
            history=history,
            lambda_value=lambda_value,
            plan=plan,
            donors=pair,
        )
        key = "control_trajectory_sha256" if history == "CONTROL" else "perturbed_trajectory_sha256"
        original = _level(next(row for row in metrics if row["regime_id"] == regime_id and row["choice_seed"] == 0), lambda_value)[key]
        if result["trajectory_sha256"] != original:
            raise RuntimeError(f"{regime_id} {history} lambda {lambda_value} replay mismatch")
        found[f"{regime_id}_{history}_lambda_{_lambda_label(lambda_value)}_seed0"] = {
            "trajectory_sha256": result["trajectory_sha256"],
            "match": True,
        }
    if found["S0_CONTROL_lambda_1.00_seed0"]["trajectory_sha256"] != EXPECTED_S0_SEED0:
        raise RuntimeError("S0 lambda 1 replay is not the frozen hash")
    if found["RW50_PERTURBED_lambda_1.00_seed0"]["trajectory_sha256"] != EXPECTED_RW50_PERTURBED_SEED0:
        raise RuntimeError("RW50 lambda 1 replay is not the frozen hash")
    return found


def _probability_tv(branch_probs, control_probs) -> list[float]:
    series = [0.0]
    for timestep in range(1, len(branch_probs)):
        agents = sorted(branch_probs[timestep])
        series.append(
            statistics.fmean(
                0.5 * sum(abs(branch_probs[timestep][agent_id][place_id] - control_probs[timestep][agent_id][place_id]) for place_id in PLACE_IDS)
                for agent_id in agents
            )
        )
    return series


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


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
    manifest = json.loads((WORLD_V11_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = manifest["trajectory_sha256"]
    if hashes["S0:0:CONTROL_ENDOGENOUS"] != EXPECTED_S0_SEED0:
        raise RuntimeError("frozen S0 seed 0 hash is not the preregistered value")
    if hashes["RW50:0:PERTURBED_ENDOGENOUS"] != EXPECTED_RW50_PERTURBED_SEED0:
        raise RuntimeError("frozen RW50 perturbed seed 0 hash is not the preregistered value")
    return hashes


def _exp011_reference() -> dict[str, Any]:
    summary = json.loads((WORLD_V11_DIR / "experiment_011_summary.json").read_text(encoding="utf-8"))
    return {
        "off_class_exp011": summary["off_class_exp011"],
        "synchronization_type_exp011": summary["synchronization_type_exp011"],
        "D_endogenous": {regime_id: summary["regimes"][regime_id]["D_endogenous"] for regime_id in ("S0", "RW50", "EXP99")},
    }


def _first_true(values: list[float], predicate) -> int | None:
    for timestep, value in enumerate(values):
        if predicate(value):
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
        "sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "median": statistics.median(finite),
        "q1": quarters[0],
        "q3": quarters[2],
        "n": len(finite),
    }


def _config(seeds, timesteps, reference_011) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_12",
        "experiment_id": "experiment_012_feedback_autonomy",
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
        "intervention_source": "data/worlds/world_v0_7/interventions.jsonl",
        "donor_source": "data/worlds/world_v0_11/donor_social_fields.jsonl",
        "override_count": OVERRIDE_COUNT,
        "blend_tolerance": BLEND_TOLERANCE,
        "midpoint_note": MIDPOINT_NOTE,
        "onset_rule": ONSET_RULE,
        "conclusion_rule": CONCLUSION_RULE,
        "attenuation_rule": ATTENUATION_RULE,
        "reconvergence_rule": RECONVERGENCE_RULE,
        "experiment_011_reference": reference_011,
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
        "experiment_id": "experiment_012_feedback_autonomy",
        "llm_calls": 0,
        "world_count": summary["world_count"],
        "seed_count": summary["seed_count"],
        "complete": summary["world_count"] == len(REGIMES) * len(CHOICE_SEEDS) * len(LAMBDAS) * len(HISTORIES),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "s0_seed0_sha256": EXPECTED_S0_SEED0,
        "gates": summary["gates"],
        "autonomy_pattern": summary["autonomy_pattern"],
        "persistence_onset_band": summary["persistence_onset_band"],
        "conclusion": summary["conclusion"],
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": trajectories,
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_12_config.json",
                "world_states.jsonl",
                "social_memory_states.jsonl",
                "agent_actions.jsonl",
                "common_midpoint_field.jsonl",
                "feedback_blend_audit.jsonl",
                "autonomy_metrics.jsonl",
                "autonomy_dose_response.json",
                "persistence_onset_analysis.json",
                "experiment_012_summary.json",
                "regime_comparison.json",
            )
        },
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
            ATLAS_DIR,
            CALIBRATION_DIR,
        )
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 012 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_012")
    main()
