"""Experiment 004: endogenous social popularity feedback.

Familiarity weight is fixed at 0. Popularity is a lagged, leave-one-agent-out
function of committed presence. Utilities come from the frozen atlas plus that
Python term. This module does not call a model and does not write into earlier
world outputs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.experiment_001 import load_base_traits
from experiments.reproducibility_check import canonical_sha256
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index, travel_distance
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_4"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
WORLD_V03_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_3"
CALIBRATION_DIR = PROJECT_ROOT / "data" / "worlds" / "dynamics_calibration_001"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
FAMILIARITY_WEIGHT = 0.0
MARGIN = 0.01
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
CONDITIONS = (
    {"condition_id": "S0", "social_weight": 0.0, "signal_mode": "endogenous"},
    {"condition_id": "S50", "social_weight": 50.0, "signal_mode": "endogenous"},
    {"condition_id": "S150", "social_weight": 150.0, "signal_mode": "endogenous"},
    {"condition_id": "S150_X", "social_weight": 150.0, "signal_mode": "exogenous"},
)
ANALYSIS_NOTE = (
    "Descriptive simulation comparison of a lagged leave-one-out popularity term. "
    "This summary does not claim human realism, destination emergence, lock-in, or self-organization."
)
CLASSIFICATION_RULE = (
    "Apply the first matching class. MARGIN is 0.01 on the native scale of each metric. "
    "1. endogenous social feedback amplified collective path dependence: from S0 to S50 to S150, "
    "late-window visit HHI rises, late-window visit entropy falls, early-leader retention rises, "
    "and mean pairwise late visit-share TV rises; and each S150-minus-S0 movement in that direction "
    "exceeds the S150_X-minus-S0 movement by at least MARGIN. "
    "2. structural centralization: late HHI rises, late entropy falls, mean pairwise late TV falls, "
    "and the modal final dominant place is the same place with a rising seed proportion; "
    "also this class when site_02 visit share, site_02 final-dominance frequency, and site_02 "
    "dominance duration all rise while mean pairwise late TV does not rise. "
    "3. weak social persistence: early-leader retention rises, while absolute S150-minus-S0 changes "
    "in late HHI, late entropy, and mean pairwise late TV are all below MARGIN. "
    "4. social information effect without strong endogenous feedback amplification: "
    "absolute S150-minus-S150_X gaps on late HHI, late entropy, retention, and mean pairwise late TV "
    "are all below MARGIN, and at least one of those metrics still differs from S0 by at least MARGIN. "
    "5. no durable collective social effect: S50 and S150 both stay within MARGIN of S0 on those four metrics. "
    "6. mixed pattern outside the pre-specified classes."
)
POPULARITY_RULE = (
    "popularity_raw = log1p(cumulative committed presence of other agents through t-1). "
    "popularity_score = 0 when every raw value is 0; otherwise raw divided by the focal agent's "
    "maximum raw value across five places. Scores are in [0, 1]. "
    "The focal agent's own presence is excluded. The current timestep's decisions are excluded."
)


class SocialState:
    def __init__(self, own: dict[str, dict[str, int]], total: dict[str, int]) -> None:
        self.own = own
        self.total = total


def empty_social(agent_ids: list[str]) -> SocialState:
    own = {agent_id: {place_id: 0 for place_id in PLACE_IDS} for agent_id in sorted(agent_ids)}
    total = {place_id: 0 for place_id in PLACE_IDS}
    return SocialState(own, total)


def popularity_scores(state: SocialState, agent_id: str) -> dict[str, float]:
    raw = {}
    for place_id in PLACE_IDS:
        count = state.total[place_id] - state.own[agent_id][place_id]
        if count < 0:
            raise RuntimeError("leave-one-out social presence went negative")
        raw[place_id] = math.log1p(count)
    peak = max(raw.values())
    if peak == 0.0:
        return {place_id: 0.0 for place_id in PLACE_IDS}
    return {place_id: raw[place_id] / peak for place_id in PLACE_IDS}


def commit_social(state: SocialState, destinations: dict[str, str]) -> SocialState:
    own = {agent_id: dict(state.own[agent_id]) for agent_id in state.own}
    total = dict(state.total)
    for agent_id in sorted(destinations):
        place_id = destinations[agent_id]
        own[agent_id][place_id] += 1
        total[place_id] += 1
    return SocialState(own, total)


def simulate_world(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    social: SocialState,
    *,
    social_weight: float,
    choice_seed_value: int,
    timesteps: int = TIMESTEPS,
    score_source: dict[int, dict[str, dict[str, float]]] | None = None,
    on_step: Any = None,
) -> dict[str, Any]:
    matrix = distance_matrix()
    places = place_index()
    current = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    history = _clone_social(social)
    series = [dict(current)]
    snapshots = {0: dict(history.total)}
    entries = {agent_id: 0 for agent_id in current}
    repeats = {agent_id: 0 for agent_id in current}
    leader_steps = {place_id: 0 for place_id in PLACE_IDS}
    tie_steps = 0
    reconsidered = 0
    ranking_changes = 0
    destination_changes = 0
    social_abs = 0.0
    popularity_max = 0.0
    decision_count = 0
    used_scores: dict[int, dict[str, dict[str, float]]] = {}
    moves = 0
    for timestep in range(timesteps):
        logged = timestep + 1
        occupancy = occupancy_of(current)
        decisions: dict[str, str] = {}
        actions = []
        step_scores: dict[str, dict[str, float]] = {}
        for agent_id in sorted(current):
            origin = current[agent_id]
            if score_source is None:
                scores = popularity_scores(history, agent_id)
            else:
                scores = {place_id: float(score_source[logged][agent_id][place_id]) for place_id in PLACE_IDS}
            step_scores[agent_id] = scores
            bases, totals, social_term = _utilities(
                atlas,
                matrix,
                places,
                agent_id,
                origin,
                occupancy,
                scores,
                social_weight,
            )
            does_reconsider = agent_reconsiders(
                choice_seed_value,
                timestep,
                agent_id,
                REFERENCE_RECONSIDERATION_RATE,
            )
            ranking_changed = _argmax(totals) != _argmax(bases)
            if not does_reconsider:
                destination = origin
                base_destination = origin
                probabilities = {place_id: 1.0 if place_id == origin else 0.0 for place_id in PLACE_IDS}
            else:
                draw = random.Random(choice_seed(choice_seed_value, timestep, agent_id)).random()
                destination, probabilities = _select(list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw)
                base_destination, _base_probabilities = _select(
                    list(PLACE_IDS),
                    [bases[place_id] for place_id in PLACE_IDS],
                    draw,
                )
                reconsidered += 1
            destination_changed = destination != base_destination
            if ranking_changed:
                ranking_changes += 1
            if destination_changed:
                destination_changes += 1
            social_abs += sum(abs(value) for value in social_term.values()) / len(PLACE_IDS)
            popularity_max += max(scores.values())
            decision_count += 1
            if destination != origin:
                entries[agent_id] += 1
                if history.own[agent_id][destination] > 0:
                    repeats[agent_id] += 1
            decisions[agent_id] = destination
            actions.append(
                {
                    "timestep": logged,
                    "agent_id": agent_id,
                    "origin": origin,
                    "destination": destination,
                    "stayed": destination == origin,
                    "reconsidered": does_reconsider,
                    "social_weight": social_weight,
                    "familiarity_weight": FAMILIARITY_WEIGHT,
                    "popularity_source": "s0_replay" if score_source is not None else "endogenous",
                    "base_utility": bases,
                    "popularity_score": scores,
                    "social_utility": social_term,
                    "total_utility": totals,
                    "choice_probabilities": probabilities,
                    "occupancy_before": occupancy,
                    "social_total_before": dict(history.total),
                    "ranking_changed": ranking_changed,
                    "destination_changed": destination_changed,
                    "llm_calls": 0,
                }
            )
        used_scores[logged] = step_scores
        moves += sum(1 for agent_id in decisions if decisions[agent_id] != current[agent_id])
        current = {agent_id: decisions[agent_id] for agent_id in sorted(current)}
        history = commit_social(history, decisions)
        series.append(dict(current))
        if logged in {20, 100, 500} or logged == timesteps:
            snapshots[logged] = dict(history.total)
        winner = _unique_leader(history.total)
        if winner is None:
            tie_steps += 1
        else:
            leader_steps[winner] += 1
        if on_step is not None:
            on_step(logged, current, history, actions)
    return {
        "series": series,
        "snapshots": snapshots,
        "final_social": history,
        "entries": entries,
        "repeats": repeats,
        "dominance_duration": {place_id: leader_steps[place_id] / timesteps for place_id in PLACE_IDS},
        "dominance_tie_fraction": tie_steps / timesteps,
        "moves": moves,
        "used_scores": used_scores,
        "reconsidered": reconsidered,
        "ranking_change_rate": ranking_changes / decision_count,
        "destination_change_rate": None if reconsidered == 0 else destination_changes / reconsidered,
        "mean_abs_social_utility": social_abs / decision_count,
        "mean_max_popularity": popularity_max / decision_count,
        "llm_calls": 0,
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
    locations = load_initial_locations()
    bases = load_base_traits(PROJECT_ROOT / "data" / "personas" / "persona_set_v1", len(locations))
    _traits = {base["base_agent_id"]: base["traits"] for base in bases}
    if set(_traits) != set(locations):
        raise RuntimeError("persona ids do not match the reference locations")
    social = empty_social(list(locations))
    history_document = _social_document(social, locations)
    write_json(destination / "neutral_social_initialization.json", history_document)
    reference: dict[int, dict[int, dict[str, dict[str, float]]]] = {}
    config = _config(history_document["sha256"], seeds, timesteps)
    write_json(destination / "world_v0_4_config.json", config)
    state_path = destination / "world_states.jsonl"
    social_path = destination / "social_states.jsonl"
    action_path = destination / "agent_actions.jsonl"
    reference_path = destination / "social_signal_reference_s0.jsonl"
    for path in (state_path, social_path, action_path, reference_path):
        path.write_text("", encoding="utf-8", newline="\n")
    worlds = []
    with state_path.open("a", encoding="utf-8", newline="\n") as state_file, social_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as social_file, action_path.open("a", encoding="utf-8", newline="\n") as action_file, reference_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as reference_file:
        for spec in CONDITIONS:
            for seed in seeds:
                _write_initial(state_file, social_file, spec, seed, locations, social)
                score_source = reference[seed] if spec["signal_mode"] == "exogenous" else None
                result = simulate_world(
                    source,
                    locations,
                    social,
                    social_weight=float(spec["social_weight"]),
                    choice_seed_value=seed,
                    timesteps=timesteps,
                    score_source=score_source,
                    on_step=_log_step(state_file, social_file, action_file, spec, seed),
                )
                if result["llm_calls"] != 0:
                    raise RuntimeError("experiment 004 called a model")
                if spec["condition_id"] == "S0":
                    reference[seed] = result["used_scores"]
                    for logged, scores in result["used_scores"].items():
                        _dump(
                            reference_file,
                            {"choice_seed": seed, "timestep": logged, "popularity_score": scores, "llm_calls": 0},
                        )
                worlds.append(_world_metrics(spec, seed, result, timesteps))
                print(
                    f"{spec['condition_id']} seed={seed} late_hhi={worlds[-1]['late_hhi']:.3f}",
                    flush=True,
                )
            if spec["condition_id"] == "S0":
                _require_s0_matches_r0(worlds, timesteps)
    replays = {}
    for condition_id, weight, mode in (("S150", 150.0, None), ("S150_X", 150.0, reference[seeds[0]])):
        replay = simulate_world(
            source,
            locations,
            social,
            social_weight=weight,
            choice_seed_value=seeds[0],
            timesteps=timesteps,
            score_source=mode,
        )
        original = next(world for world in worlds if world["condition_id"] == condition_id and world["choice_seed"] == seeds[0])
        replay_hash = trajectory_hash(replay["series"])
        if replay_hash != original["trajectory_sha256"]:
            raise RuntimeError(f"{condition_id} exact replay hash did not match")
        replays[condition_id] = {"choice_seed": seeds[0], "trajectory_sha256": replay_hash, "match": True}
    summary = _summarize(worlds, seeds, timesteps, replays)
    write_json(destination / "experiment_004_summary.json", summary)
    manifest = _manifest(destination, config, history_document, summary, worlds)
    write_json(destination / "run_manifest.json", manifest)
    return destination


def trajectory_hash(series: list[dict[str, str]]) -> str:
    ordered = [{agent_id: step[agent_id] for agent_id in sorted(step)} for step in series]
    return canonical_sha256(ordered)


def classify_result(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s0, s50, s150, exogenous = (metrics[name] for name in ("S0", "S50", "S150", "S150_X"))
    hhi_up = _strictly_increasing([s0["late_hhi"], s50["late_hhi"], s150["late_hhi"]])
    entropy_down = _strictly_decreasing([s0["late_entropy"], s50["late_entropy"], s150["late_entropy"]])
    retention_up = _strictly_increasing([s0["retention"], s50["retention"], s150["retention"]])
    tv_up = _strictly_increasing([s0["between_tv"], s50["between_tv"], s150["between_tv"]])
    tv_down = _strictly_decreasing([s0["between_tv"], s50["between_tv"], s150["between_tv"]])
    gaps = {
        "late_hhi": (s150["late_hhi"] - s0["late_hhi"]) - (exogenous["late_hhi"] - s0["late_hhi"]),
        "late_entropy": (s0["late_entropy"] - s150["late_entropy"]) - (s0["late_entropy"] - exogenous["late_entropy"]),
        "retention": (s150["retention"] - s0["retention"]) - (exogenous["retention"] - s0["retention"]),
        "between_tv": (s150["between_tv"] - s0["between_tv"]) - (exogenous["between_tv"] - s0["between_tv"]),
    }
    clearly_endogenous = all(value is not None and value >= MARGIN for value in gaps.values())
    same_modal = s0["modal_place"] == s50["modal_place"] == s150["modal_place"] and s0["modal_place"] not in {None, "tie"}
    modal_up = _strictly_increasing([s0["modal_proportion"], s50["modal_proportion"], s150["modal_proportion"]])
    site_stronger = _strictly_increasing(
        [s0["site_02_share"], s50["site_02_share"], s150["site_02_share"]]
    ) and _strictly_increasing(
        [s0["site_02_dominance_frequency"], s50["site_02_dominance_frequency"], s150["site_02_dominance_frequency"]]
    ) and _strictly_increasing(
        [s0["site_02_dominance_duration"], s50["site_02_dominance_duration"], s150["site_02_dominance_duration"]]
    )
    small_long_run = all(abs(s150[key] - s0[key]) < MARGIN for key in ("late_hhi", "late_entropy", "between_tv"))
    small_versus_exogenous = all(
        abs(s150[key] - exogenous[key]) < MARGIN for key in ("late_hhi", "late_entropy", "retention", "between_tv")
    )
    durable_versus_s0 = any(
        abs(s150[key] - s0[key]) >= MARGIN or abs(s50[key] - s0[key]) >= MARGIN
        for key in ("late_hhi", "late_entropy", "retention", "between_tv")
    )
    no_durable = all(
        abs(s150[key] - s0[key]) < MARGIN and abs(s50[key] - s0[key]) < MARGIN
        for key in ("late_hhi", "late_entropy", "retention", "between_tv")
    )
    checks = {
        "late_hhi_rises": hhi_up,
        "late_entropy_falls": entropy_down,
        "early_leader_retention_rises": retention_up,
        "between_seed_late_tv_rises": tv_up,
        "s150_clearly_above_s150_x": clearly_endogenous,
        "structural_concentration": hhi_up and entropy_down and tv_down and same_modal and modal_up,
        "site_02_strengthens_without_divergence_rise": site_stronger and not tv_up,
        "small_long_run_change": small_long_run,
        "small_s150_versus_s150_x": small_versus_exogenous,
        "durable_versus_s0": durable_versus_s0,
    }
    if hhi_up and entropy_down and retention_up and tv_up and clearly_endogenous:
        label = "endogenous social feedback amplified collective path dependence"
    elif (hhi_up and entropy_down and tv_down and same_modal and modal_up) or (site_stronger and not tv_up):
        label = "structural centralization"
    elif retention_up and small_long_run:
        label = "weak social persistence"
    elif small_versus_exogenous and durable_versus_s0:
        label = "social information effect without strong endogenous feedback amplification"
    elif no_durable:
        label = "no durable collective social effect"
    else:
        label = "mixed pattern outside the pre-specified classes"
    return {"label": label, "checks": checks, "endogenous_minus_exogenous_gaps": gaps, "margin": MARGIN}


def _utilities(
    atlas: FrozenUtilityAtlas,
    matrix: dict[str, dict[str, float]],
    places: dict[str, dict[str, Any]],
    agent_id: str,
    origin: str,
    occupancy: dict[str, int],
    scores: dict[str, float],
    social_weight: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    bases: dict[str, float] = {}
    totals: dict[str, float] = {}
    social_term: dict[str, float] = {}
    for place_id in PLACE_IDS:
        capacity = int(places[place_id]["capacity"])
        distance = travel_distance(matrix, origin, place_id)
        base = float(atlas.lookup(agent_id, place_id, distance, occupancy[place_id], capacity))
        term = 0.0 if social_weight == 0.0 else social_weight * scores[place_id]
        bases[place_id] = base
        social_term[place_id] = term
        totals[place_id] = base + term
    return bases, totals, social_term


def _select(place_ids: list[str], utilities: list[float], draw: float) -> tuple[str, dict[str, float]]:
    peak = max(utilities)
    weights = [math.exp((utility - peak) / REFERENCE_CHOICE_TEMPERATURE) for utility in utilities]
    total = math.fsum(weights)
    probabilities = [weight / total for weight in weights]
    cursor = 0.0
    selected = place_ids[-1]
    for place_id, probability in zip(place_ids, probabilities, strict=True):
        cursor += probability
        if draw <= cursor:
            selected = place_id
            break
    return selected, {place_id: probability for place_id, probability in zip(place_ids, probabilities, strict=True)}


def _argmax(utilities: dict[str, float]) -> str:
    best = max(utilities[place_id] for place_id in PLACE_IDS)
    for place_id in PLACE_IDS:
        if utilities[place_id] == best:
            return place_id
    raise RuntimeError("utility argmax was empty")


def _world_metrics(spec: dict[str, Any], seed: int, result: dict[str, Any], timesteps: int) -> dict[str, Any]:
    snapshots = result["snapshots"]
    full_counts = snapshots[timesteps]
    early_counts = snapshots[20]
    late_counts = {place_id: snapshots[timesteps][place_id] - snapshots[100][place_id] for place_id in PLACE_IDS}
    full_share = _share(full_counts)
    early_share = _share(early_counts)
    late_share = _share(late_counts)
    early_leader = _unique_leader(early_counts)
    late_leader = _unique_leader(late_counts)
    final_leader = _unique_leader(full_counts)
    retained: bool | None
    if early_leader is None:
        retained = None
    else:
        retained = late_leader == early_leader
    diagnostics = measure_world(result["series"], moves=result["moves"], timesteps=timesteps)
    own = result["final_social"].own
    personal_entropy = [ _entropy(_share(own[agent_id])) for agent_id in sorted(own) ]
    personal_max = [max(_share(own[agent_id]).values()) for agent_id in sorted(own)]
    repeat_rates = [
        None if result["entries"][agent_id] == 0 else result["repeats"][agent_id] / result["entries"][agent_id]
        for agent_id in sorted(result["entries"])
    ]
    occupancy = [tuple(occupancy_of(step)[place_id] for place_id in PLACE_IDS) for step in result["series"]]
    record = {
        "condition_id": spec["condition_id"],
        "social_weight": spec["social_weight"],
        "signal_mode": spec["signal_mode"],
        "choice_seed": seed,
        "matched_reference_seed": seed if spec["signal_mode"] == "exogenous" else None,
        "trajectory_sha256": trajectory_hash(result["series"]),
        "series": result["series"] if spec["condition_id"] == "S0" else None,
        "late_share": late_share,
        "early_share": early_share,
        "full_share": full_share,
        "late_hhi": _hhi(late_share),
        "late_entropy": _entropy(late_share),
        "late_dominant_share": max(late_share.values()),
        "late_dominant": late_leader,
        "full_hhi": _hhi(full_share),
        "full_entropy": _entropy(full_share),
        "full_dominant_share": max(full_share.values()),
        "full_dominant": final_leader,
        "early_leader": early_leader,
        "early_leader_retained": retained,
        "late_share_of_early_leader": None if early_leader is None else late_share[early_leader],
        "early_late_correlation": _pearson(
            [early_share[place_id] for place_id in PLACE_IDS],
            [late_share[place_id] for place_id in PLACE_IDS],
        ),
        "early_late_tv": _tv(early_share, late_share),
        "early_late_js_distance": _js_distance(early_share, late_share),
        "dominance_duration": result["dominance_duration"],
        "dominance_tie_fraction": result["dominance_tie_fraction"],
        "personal_entropy_mean": statistics.fmean(personal_entropy),
        "max_personal_share_mean": statistics.fmean(personal_max),
        "repeat_visit_probability": _finite_mean(repeat_rates),
        "movement_rate": diagnostics["movement_rate"],
        "mean_dwell": diagnostics["mean_dwell"],
        "occupancy_entropy": diagnostics["entropy_mean"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "lag1": diagnostics["lag1"],
        "lag2": diagnostics["lag2"],
        "mean_occupancy": diagnostics["mean_occupancy"],
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "ranking_change_rate": result["ranking_change_rate"],
        "destination_change_rate": result["destination_change_rate"],
        "mean_abs_social_utility": result["mean_abs_social_utility"],
        "mean_max_popularity": result["mean_max_popularity"],
        "llm_calls": 0,
    }
    if spec["condition_id"] in {"S150", "S150_X"}:
        record["occupancy"] = occupancy
    return record


def _summarize(
    worlds: list[dict[str, Any]],
    seeds: tuple[int, ...],
    timesteps: int,
    replays: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_condition = {
        spec["condition_id"]: [world for world in worlds if world["condition_id"] == spec["condition_id"]]
        for spec in CONDITIONS
    }
    between = {condition_id: _between_seed(rows) for condition_id, rows in by_condition.items()}
    conditions = {condition_id: _condition_summary(rows, between[condition_id]) for condition_id, rows in by_condition.items()}
    classification_input = {
        condition_id: {
            "late_hhi": conditions[condition_id]["late_hhi"]["mean"],
            "late_entropy": conditions[condition_id]["late_entropy"]["mean"],
            "retention": conditions[condition_id]["early_leader_retention"]["mean"],
            "between_tv": between[condition_id]["tv_distance"]["mean"],
            "modal_place": conditions[condition_id]["modal_final_dominant_place"],
            "modal_proportion": conditions[condition_id]["modal_final_dominant_proportion"],
            "site_02_share": conditions[condition_id]["full_visit_share_by_place"]["site_02"]["mean"],
            "site_02_dominance_frequency": conditions[condition_id]["site_02_final_dominance_frequency"],
            "site_02_dominance_duration": conditions[condition_id]["dominance_duration"]["site_02"]["mean"],
        }
        for condition_id in by_condition
    }
    return {
        "experiment_id": "experiment_004_social_popularity",
        "note": ANALYSIS_NOTE,
        "classification_rule": CLASSIFICATION_RULE,
        "popularity_rule": POPULARITY_RULE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "margin": MARGIN,
        "llm_calls": 0,
        "world_count": len(worlds),
        "timesteps": timesteps,
        "choice_seeds": list(seeds),
        "s0_matches_r0_seeds_0_19": True,
        "js_definition": "Jensen-Shannon distance = sqrt(JS divergence), with natural log.",
        "rank_definition": (
            "Normalized Kendall distance: discordant place pairs divided by C(5, 2). "
            "A pair with a tie in either share vector is not discordant."
        ),
        "conditions": conditions,
        "between_seed": between,
        "s150_versus_s150_x": _matched_decomposition(by_condition["S150"], by_condition["S150_X"]),
        "classification": classify_result(classification_input),
        "exact_replay": replays,
    }


def _condition_summary(worlds: list[dict[str, Any]], between: dict[str, Any]) -> dict[str, Any]:
    counts = _dominant_counts([world["full_dominant"] for world in worlds])
    place_order = {place_id: index for index, place_id in enumerate(PLACE_IDS)}
    place_order["tie"] = len(PLACE_IDS)
    modal_place, modal_count = max(
        counts.items(),
        key=lambda item: (item[1], item[0] != "tie", -place_order[item[0]]),
    )
    return {
        "n_seeds": len(worlds),
        "llm_calls": 0,
        "signal_mode": worlds[0]["signal_mode"],
        "social_weight": worlds[0]["social_weight"],
        "late_hhi": _mean_sd([world["late_hhi"] for world in worlds]),
        "late_entropy": _mean_sd([world["late_entropy"] for world in worlds]),
        "late_dominant_share": _mean_sd([world["late_dominant_share"] for world in worlds]),
        "full_hhi": _mean_sd([world["full_hhi"] for world in worlds]),
        "full_entropy": _mean_sd([world["full_entropy"] for world in worlds]),
        "full_dominant_share": _mean_sd([world["full_dominant_share"] for world in worlds]),
        "early_leader_retention": _mean_sd(
            [None if world["early_leader_retained"] is None else float(world["early_leader_retained"]) for world in worlds]
        ),
        "late_share_of_early_leader": _mean_sd([world["late_share_of_early_leader"] for world in worlds]),
        "early_late_correlation": _mean_sd([world["early_late_correlation"] for world in worlds]),
        "early_late_tv": _mean_sd([world["early_late_tv"] for world in worlds]),
        "early_late_js_distance": _mean_sd([world["early_late_js_distance"] for world in worlds]),
        "late_visit_share_by_place": {
            place_id: _mean_sd([world["late_share"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "full_visit_share_by_place": {
            place_id: _mean_sd([world["full_share"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "final_dominant_counts": counts,
        "final_dominant_proportions": {key: value / len(worlds) for key, value in counts.items()},
        "final_dominant_entropy": _category_entropy(
            ["tie" if world["full_dominant"] is None else world["full_dominant"] for world in worlds]
        ),
        "modal_final_dominant_place": modal_place,
        "modal_final_dominant_proportion": modal_count / len(worlds),
        "site_02_final_dominance_frequency": counts.get("site_02", 0) / len(worlds),
        "dominance_duration": {
            place_id: _mean_sd([world["dominance_duration"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "dominance_tie_fraction": _mean_sd([world["dominance_tie_fraction"] for world in worlds]),
        "personal_entropy": _mean_sd([world["personal_entropy_mean"] for world in worlds]),
        "max_personal_share": _mean_sd([world["max_personal_share_mean"] for world in worlds]),
        "repeat_visit_probability": _mean_sd([world["repeat_visit_probability"] for world in worlds]),
        "movement_rate": _mean_sd([world["movement_rate"] for world in worlds]),
        "mean_dwell": _mean_sd([world["mean_dwell"] for world in worlds]),
        "occupancy_entropy": _mean_sd([world["occupancy_entropy"] for world in worlds]),
        "saturation_frequency": _mean_sd([world["saturation_frequency"] for world in worlds]),
        "empty_frequency": _mean_sd([world["empty_frequency"] for world in worlds]),
        "alternation_rate": _mean_sd([world["alternation_rate"] for world in worlds]),
        "mean_occupancy": {
            place_id: _mean_sd([world["mean_occupancy"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "lag1": {place_id: _mean_sd([world["lag1"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "lag2": {place_id: _mean_sd([world["lag2"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "ranking_change_rate": _mean_sd([world["ranking_change_rate"] for world in worlds]),
        "destination_change_rate": _mean_sd([world["destination_change_rate"] for world in worlds]),
        "mean_abs_social_utility": _mean_sd([world["mean_abs_social_utility"] for world in worlds]),
        "mean_max_popularity": _mean_sd([world["mean_max_popularity"] for world in worlds]),
        "between_seed_sd": {
            "late_hhi": between["tv_distance"]["sd"],
            "late_entropy": _stdev([world["late_entropy"] for world in worlds]),
            "full_dominant_share": _stdev([world["full_dominant_share"] for world in worlds]),
        },
    }


def _between_seed(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    vectors = [[world["late_share"][place_id] for place_id in PLACE_IDS] for world in worlds]
    labels = ["tie" if world["late_dominant"] is None else world["late_dominant"] for world in worlds]
    tv_values = []
    js_values = []
    dominant_values = []
    rank_values = []
    for left, right in itertools.combinations(range(len(worlds)), 2):
        tv_values.append(_tv_lists(vectors[left], vectors[right]))
        js_values.append(_js_distance_lists(vectors[left], vectors[right]))
        dominant_values.append(0.0 if labels[left] == labels[right] else 1.0)
        rank_values.append(_kendall(worlds[left]["late_share"], worlds[right]["late_share"]))
    return {
        "n_pairs": len(tv_values),
        "tv_distance": _distribution(tv_values),
        "js_distance": _distribution(js_values),
        "dominant_place_disagreement": _distribution(dominant_values),
        "kendall_distance": _distribution(rank_values),
    }


def _matched_decomposition(endogenous: list[dict[str, Any]], exogenous: list[dict[str, Any]]) -> dict[str, Any]:
    tv_values = []
    js_values = []
    dominant_different = []
    occupancy_tv = []
    hhi_difference = []
    retention_difference = []
    for left in endogenous:
        right = next(world for world in exogenous if world["choice_seed"] == left["choice_seed"])
        tv_values.append(_tv(left["late_share"], right["late_share"]))
        js_values.append(_js_distance(left["late_share"], right["late_share"]))
        dominant_different.append(0.0 if left["late_dominant"] == right["late_dominant"] else 1.0)
        occupancy_tv.append(_occupancy_curve_tv(left["occupancy"], right["occupancy"], 101, len(left["occupancy"]) - 1))
        hhi_difference.append(left["late_hhi"] - right["late_hhi"])
        if left["early_leader_retained"] is None or right["early_leader_retained"] is None:
            retention_difference.append(None)
        else:
            retention_difference.append(float(left["early_leader_retained"]) - float(right["early_leader_retained"]))
    return {
        "n_seeds": len(tv_values),
        "late_visit_share_tv": _distribution(tv_values),
        "late_visit_share_js_distance": _distribution(js_values),
        "late_dominant_place_different_rate": _mean_sd(dominant_different),
        "late_occupancy_tv": _distribution(occupancy_tv),
        "late_hhi_difference_s150_minus_s150_x": _mean_sd(hhi_difference),
        "early_leader_retention_difference_s150_minus_s150_x": _mean_sd(retention_difference),
    }


def _require_s0_matches_r0(worlds: list[dict[str, Any]], timesteps: int) -> None:
    reference = _load_r0_locations()
    compared = 0
    for world in worlds:
        if world["condition_id"] != "S0" or world["choice_seed"] not in reference:
            continue
        if timesteps < 500:
            raise RuntimeError("S0/R0 comparison needs 500 timesteps")
        expected = reference[world["choice_seed"]]
        actual = world["series"]
        for timestep, (left, right) in enumerate(zip(actual, expected, strict=True)):
            if left != right:
                raise RuntimeError(f"S0 diverges from R0 at choice seed {world['choice_seed']} timestep {timestep}")
        compared += 1
    if compared != 20:
        raise RuntimeError("S0/R0 comparison did not cover choice seeds 0–19")
    for world in worlds:
        if world["condition_id"] == "S0":
            world["series"] = None


def _load_r0_locations() -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[tuple[int, dict[str, str]]]] = {seed: [] for seed in range(20)}
    path = WORLD_V03_DIR / "world_states.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"condition_id":"R0"' not in line:
                continue
            row = json.loads(line)
            seed = int(row["choice_seed"])
            if row.get("condition_id") != "R0" or seed not in grouped:
                continue
            grouped[seed].append(
                (int(row["timestep"]), {str(key): str(value) for key, value in row["locations"].items()})
            )
    loaded = {}
    for seed, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        loaded[seed] = [locations for _timestep, locations in rows[:501]]
    return loaded


def _social_document(social: SocialState, locations: dict[str, str]) -> dict[str, Any]:
    payload = {
        "algorithm": "worldlab.neutral_social.v1",
        "note": "social_presence_count starts at 0. The initial location is not social history.",
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "social_presence_count": social.total,
        "own_presence": social.own,
        "initial_locations": locations,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _config(social_sha256: str, seeds: tuple[int, ...], timesteps: int) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_4",
        "experiment_id": "experiment_004_social_popularity",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "prompt_version": "independent_utility_v3",
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "conditions": [dict(spec) for spec in CONDITIONS],
        "popularity_rule": POPULARITY_RULE,
        "classification_rule": CLASSIFICATION_RULE,
        "margin": MARGIN,
        "neutral_social_sha256": social_sha256,
        "initial_state_source": str(WORLD_V01_DIR / "initial_state.json"),
        "llm_calls": 0,
        "decision_signal": "overall_utility_plus_lagged_leave_one_out_popularity",
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _manifest(
    destination: Path,
    config: dict[str, Any],
    history_document: dict[str, Any],
    summary: dict[str, Any],
    worlds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_004_social_popularity",
        "llm_calls": 0,
        "world_count": len(worlds),
        "complete": len(worlds) == len(CONDITIONS) * len(config["choice_seeds"]),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "neutral_social_sha256": history_document["sha256"],
        "exact_replay": summary["exact_replay"],
        "classification": summary["classification"]["label"],
        "trajectory_sha256": {
            f"{world['condition_id']}:{world['choice_seed']}": world["trajectory_sha256"] for world in worlds
        },
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_4_config.json",
                "neutral_social_initialization.json",
                "social_signal_reference_s0.jsonl",
                "social_states.jsonl",
                "world_states.jsonl",
                "agent_actions.jsonl",
                "experiment_004_summary.json",
            )
        },
    }


def _log_step(state_file: Any, social_file: Any, action_file: Any, spec: dict[str, Any], seed: int) -> Any:
    def write(logged: int, locations: dict[str, str], social: SocialState, actions: list[dict[str, Any]]) -> None:
        _dump(
            state_file,
            {
                "condition_id": spec["condition_id"],
                "choice_seed": seed,
                "social_weight": spec["social_weight"],
                "signal_mode": spec["signal_mode"],
                "timestep": logged,
                "locations": locations,
                "occupancy": occupancy_of(locations),
                "llm_calls": 0,
            },
        )
        _dump(
            social_file,
            {
                "condition_id": spec["condition_id"],
                "choice_seed": seed,
                "timestep": logged,
                "social_presence_count": social.total,
                "own_presence": social.own,
                "used_for_decision": spec["signal_mode"] == "endogenous",
                "llm_calls": 0,
            },
        )
        for action in actions:
            _dump(action_file, {"condition_id": spec["condition_id"], "choice_seed": seed, "signal_mode": spec["signal_mode"], **action})

    return write


def _write_initial(
    state_file: Any,
    social_file: Any,
    spec: dict[str, Any],
    seed: int,
    locations: dict[str, str],
    social: SocialState,
) -> None:
    _dump(
        state_file,
        {
            "condition_id": spec["condition_id"],
            "choice_seed": seed,
            "social_weight": spec["social_weight"],
            "signal_mode": spec["signal_mode"],
            "timestep": 0,
            "locations": {agent_id: locations[agent_id] for agent_id in sorted(locations)},
            "occupancy": occupancy_of(locations),
            "llm_calls": 0,
        },
    )
    _dump(
        social_file,
        {
            "condition_id": spec["condition_id"],
            "choice_seed": seed,
            "timestep": 0,
            "social_presence_count": social.total,
            "own_presence": social.own,
            "used_for_decision": spec["signal_mode"] == "endogenous",
            "llm_calls": 0,
        },
    )


def _clone_social(social: SocialState) -> SocialState:
    return SocialState(
        {agent_id: dict(social.own[agent_id]) for agent_id in social.own},
        dict(social.total),
    )


def _share(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts[place_id] for place_id in PLACE_IDS)
    if total == 0:
        raise RuntimeError("visit share was requested for an empty window")
    return {place_id: counts[place_id] / total for place_id in PLACE_IDS}


def _hhi(share: dict[str, float]) -> float:
    return sum(value * value for value in share.values())


def _entropy(share: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in share.values() if value > 0.0)


def _unique_leader(counts: dict[str, int]) -> str | None:
    best = max(counts[place_id] for place_id in PLACE_IDS)
    winners = [place_id for place_id in PLACE_IDS if counts[place_id] == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _dominant_counts(labels: list[str | None]) -> dict[str, int]:
    counts = {place_id: 0 for place_id in PLACE_IDS}
    counts["tie"] = 0
    for label in labels:
        counts["tie" if label is None else label] += 1
    return counts


def _tv(left: dict[str, float], right: dict[str, float]) -> float:
    return _tv_lists([left[place_id] for place_id in PLACE_IDS], [right[place_id] for place_id in PLACE_IDS])


def _tv_lists(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(item - other) for item, other in zip(left, right, strict=True))


def _js_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return _js_distance_lists([left[place_id] for place_id in PLACE_IDS], [right[place_id] for place_id in PLACE_IDS])


def _js_distance_lists(left: list[float], right: list[float]) -> float:
    midpoint = [0.5 * (item + other) for item, other in zip(left, right, strict=True)]
    divergence = 0.5 * _kl(left, midpoint) + 0.5 * _kl(right, midpoint)
    return math.sqrt(divergence)


def _kl(source: list[float], target: list[float]) -> float:
    total = 0.0
    for item, other in zip(source, target, strict=True):
        if item > 0.0:
            total += item * math.log(item / other)
    return total


def _kendall(left: dict[str, float], right: dict[str, float]) -> float:
    discordant = 0
    pairs = 0
    for first, second in itertools.combinations(PLACE_IDS, 2):
        pairs += 1
        left_gap = left[first] - left[second]
        right_gap = right[first] - right[second]
        if left_gap == 0.0 or right_gap == 0.0:
            continue
        if left_gap * right_gap < 0.0:
            discordant += 1
    return discordant / pairs


def _occupancy_curve_tv(left: list[tuple[int, ...]], right: list[tuple[int, ...]], start: int, stop: int) -> float:
    values = []
    population = sum(left[0])
    for lhs, rhs in zip(left[start : stop + 1], right[start : stop + 1], strict=True):
        change = sum(abs(item - other) for item, other in zip(lhs, rhs, strict=True))
        values.append(0.5 * change / population)
    return statistics.fmean(values)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((item - mean_left) * (other - mean_right) for item, other in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((item - mean_left) ** 2 for item in left))
    right_scale = math.sqrt(sum((item - mean_right) ** 2 for item in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _category_entropy(labels: list[str]) -> float:
    total = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _strictly_increasing(values: list[Any]) -> bool:
    if any(value is None for value in values):
        return False
    return bool(values[0] < values[1] < values[2])


def _strictly_decreasing(values: list[Any]) -> bool:
    if any(value is None for value in values):
        return False
    return bool(values[0] > values[1] > values[2])


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "mean": statistics.fmean(values),
        "sd": _stdev(values),
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "n": len(values),
    }


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return statistics.fmean(finite)


def _mean_sd(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    return {"mean": None if not finite else statistics.fmean(finite), "sd": _stdev(finite), "n": len(finite)}


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _require_atlas_hash() -> None:
    manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != EXPECTED_ATLAS_SHA256:
        raise RuntimeError("utility atlas hash is not the frozen experiment atlas")


def _refuse_frozen_directories(destination: Path) -> None:
    forbidden = {
        WORLD_V01_DIR.resolve(),
        WORLD_V02_DIR.resolve(),
        WORLD_V03_DIR.resolve(),
        ATLAS_DIR.resolve(),
        CALIBRATION_DIR.resolve(),
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 004 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_004")
    main()
