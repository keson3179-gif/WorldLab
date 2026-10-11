"""Synthetic checks for the Experiment 017 design.

These checks call Experiment 004 and 005 functions on small inputs.
They do not run a world, write into data/, or call a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hashlib
import itertools
import math
import random
import statistics
from typing import Any

from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE
from world.experiment_004 import (
    SocialState,
    _select,
    _utilities as cumulative_utilities,
    commit_social,
    empty_social,
    popularity_scores,
)
from world.experiment_005 import (
    SOCIAL_WEIGHT,
    DecayMemory,
    RollingMemory,
    _population_sd,
    _utilities as memory_utilities,
    centered_scores,
)
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.utility_atlas import load_frozen_atlas

ROOT = _ROOT
DESIGN_DIR = Path(__file__).resolve().parent
HASH_CHUNK_BYTES = 1024 * 1024
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260928
PLACE_COUNT = 5


def run_checks() -> dict[str, Any]:
    report: dict[str, Any] = {
        "record_type": "experiment_017_formula_check",
        "status": "DESIGN_ONLY",
        "run_status": "NOT_RUN",
        "formal_world_executed": False,
        "llm_calls": 0,
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "social_weight_module_default": SOCIAL_WEIGHT,
        "place_order": list(PLACE_IDS),
    }
    report["q_definition"] = _q_definition()
    report["softmax_shift_invariance"] = _softmax_shift()
    report["equal_inputs"] = _equal_inputs()
    report["scale"] = _scale()
    report["five_place_bound"] = _five_place_bound()
    report["historical_memory"] = _historical_memory()
    report["weight_zero_gate"] = _weight_zero_gate()
    report["stored_anchor_hashes"] = _stored_anchor_hashes()
    indexes = paired_seed_indexes(50, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    synthetic_values = [float(seed) for seed in range(50)]
    report["bootstrap_rule"] = {
        "draws": BOOTSTRAP_DRAWS,
        "n_seeds": 50,
        "analysis_rng_seed": BOOTSTRAP_SEED,
        "generator": "random.Random",
        "index_type": "list[list[int]]",
        "sampling": "with_replacement_within_each_row",
        "synthetic_matrix_shape": [len(indexes), len(indexes[0])],
        "unit": "choice_seed",
        "same_row_shared_by_all_cells_within_a_layer": True,
        "layers_analyzed_separately": True,
        "percentile_indexes": list(percentile_indexes(BOOTSTRAP_DRAWS)),
        "interval_label": "pointwise",
        "synthetic_sampling_check_executed": True,
        "formal_research_bootstrap_executed": False,
        "synthetic_paired_constant_check": paired_constant_difference(synthetic_values, 3.5, indexes),
        "synthetic_known_row_mean": resampled_mean([0.0, 1.0, 4.0], [2, 2, 1]),
    }
    return report


def percentile_indexes(draws: int) -> tuple[int, int]:
    return int(0.025 * (draws - 1)), int(0.975 * (draws - 1))


def paired_seed_indexes(n_seeds: int, draws: int, rng_seed: int) -> list[list[int]]:
    generator = random.Random(rng_seed)
    return [
        [generator.randrange(n_seeds) for _column in range(n_seeds)]
        for _draw in range(draws)
    ]


def resampled_mean(values: list[float], indexes: list[int]) -> float:
    return statistics.fmean(values[index] for index in indexes)


def paired_constant_difference(
    values: list[float],
    constant: float,
    indexes: list[list[int]],
) -> dict[str, Any]:
    shifted = [value + constant for value in values]
    gaps = [resampled_mean(shifted, row) - resampled_mean(values, row) for row in indexes]
    return {
        "constant": constant,
        "replicate_count": len(gaps),
        "sample_size": len(indexes[0]),
        "max_abs_gap_from_constant": max(abs(gap - constant) for gap in gaps),
        "input": "synthetic_seed_level_values",
        "formal_research_series_used": False,
    }


def probability_map(utilities: list[float], draw: float = 0.5) -> dict[str, float]:
    _selected, probabilities = _select(list(PLACE_IDS), utilities, draw)
    return probabilities


def half_l1(left: dict[str, float], right: dict[str, float]) -> float:
    return 0.5 * sum(abs(left[place_id] - right[place_id]) for place_id in PLACE_IDS)


def social_from_loo(loo: dict[str, float]) -> SocialState:
    state = empty_social(["focal", "other"])
    for place_id in PLACE_IDS:
        state.total[place_id] = loo[place_id]
        state.own["focal"][place_id] = 0
        state.own["other"][place_id] = loo[place_id]
    return state


def logmax(loo: dict[str, float]) -> dict[str, float]:
    return popularity_scores(social_from_loo(loo), "focal")


def zraw(loo: dict[str, float]) -> dict[str, float]:
    _unclipped, clipped = centered_scores({place_id: float(loo[place_id]) for place_id in PLACE_IDS})
    return clipped


def _q_definition() -> dict[str, Any]:
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    scores = [0.0, 0.0, 0.0, 0.0, 1.0]
    social = [base[index] + SOCIAL_WEIGHT * scores[index] for index in range(PLACE_COUNT)]
    base_probability = probability_map(base)
    social_probability = probability_map(social)
    value = half_l1(social_probability, base_probability)
    return {
        "value": value,
        "uses_reconsideration_softmax": True,
        "includes_stay_put_one_hot": False,
        "persistence_threshold_applied": False,
    }


def _softmax_shift() -> dict[str, Any]:
    utilities = [10.0, 30.0, 20.0, 50.0, 40.0]
    shifted = [value + 80.0 for value in utilities]
    original = probability_map(utilities)
    moved = probability_map(shifted)
    return {
        "max_abs_probability_change": max(abs(original[place_id] - moved[place_id]) for place_id in PLACE_IDS),
        "unchanged": original == moved,
    }


def _equal_inputs() -> dict[str, Any]:
    zeros = {place_id: 0.0 for place_id in PLACE_IDS}
    equal = {place_id: 4.0 for place_id in PLACE_IDS}
    logmax_zero = logmax(zeros)
    zraw_zero = zraw(zeros)
    logmax_equal = logmax(equal)
    zraw_equal = zraw(equal)
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    social_log = [base[index] + SOCIAL_WEIGHT * 1.0 for index in range(PLACE_COUNT)]
    social_z = list(base)
    base_probability = probability_map(base)
    log_probability = probability_map(social_log)
    z_probability = probability_map(social_z)
    return {
        "all_zero_logmax": logmax_zero,
        "all_zero_zraw": zraw_zero,
        "equal_positive_logmax": logmax_equal,
        "equal_positive_zraw": zraw_equal,
        "choice_probability_max_abs_gap": max(
            abs(log_probability[place_id] - z_probability[place_id]) for place_id in PLACE_IDS
        ),
        "q_against_base": half_l1(log_probability, base_probability),
    }


def _scale() -> dict[str, Any]:
    raw = {place_id: float(index + 1) for index, place_id in enumerate(PLACE_IDS)}
    scaled = {place_id: 3.0 * raw[place_id] for place_id in PLACE_IDS}
    log_raw = logmax(raw)
    log_scaled = logmax(scaled)
    z_raw = zraw(raw)
    z_scaled = zraw(scaled)
    return {
        "logmax_max_abs_change": max(abs(log_raw[place_id] - log_scaled[place_id]) for place_id in PLACE_IDS),
        "zraw_max_abs_change": max(abs(z_raw[place_id] - z_scaled[place_id]) for place_id in PLACE_IDS),
        "zraw_uses_population_sd": _uses_population_sd(raw),
    }


def _uses_population_sd(raw: dict[str, float]) -> bool:
    values = [raw[place_id] for place_id in PLACE_IDS]
    center = statistics.fmean(values)
    population = math.sqrt(sum((value - center) ** 2 for value in values) / len(values))
    sample = math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))
    unclipped, _clipped = centered_scores(raw)
    expected = {place_id: (raw[place_id] - center) / population for place_id in PLACE_IDS}
    return population != sample and unclipped == expected


def _five_place_bound() -> dict[str, Any]:
    max_abs = 0.0
    max_clip_gap = 0.0
    near_boundary_with_unchanged_clip = 0
    for values in itertools.product(range(9), repeat=PLACE_COUNT):
        if max(values) == min(values):
            continue
        raw = {place_id: float(value) for place_id, value in zip(PLACE_IDS, values, strict=True)}
        unclipped, clipped = centered_scores(raw)
        for place_id in PLACE_IDS:
            max_abs = max(max_abs, abs(unclipped[place_id]))
            gap = abs(clipped[place_id] - unclipped[place_id])
            max_clip_gap = max(max_clip_gap, gap)
            if abs(unclipped[place_id]) >= 1.9 and gap == 0.0:
                near_boundary_with_unchanged_clip += 1
    spike = {place_id: 0.0 for place_id in PLACE_IDS}
    spike["site_01"] = 5.0
    unclipped, clipped = centered_scores(spike)
    return {
        "analytic_bound": math.sqrt(PLACE_COUNT - 1),
        "grid_values": "0..8",
        "grid_max_abs_unclipped_z": max_abs,
        "grid_max_abs_clip_change": max_clip_gap,
        "near_boundary_scores_unchanged_by_clip": near_boundary_with_unchanged_clip,
        "spike_unclipped_site_01": unclipped["site_01"],
        "spike_clip_identity": clipped == unclipped,
        "population_sd_of_spike_z": _population_sd([unclipped[place_id] for place_id in PLACE_IDS]),
    }


def _historical_memory() -> dict[str, Any]:
    cumulative = empty_social(["a", "b"])
    before = popularity_scores(cumulative, "a")
    cumulative = commit_social(cumulative, {"a": "site_01", "b": "site_02"})
    after = popularity_scores(cumulative, "a")
    rolling = RollingMemory(["a", "b"], 2)
    rolling_before = dict(rolling.loo_raw("a"))
    rolling.commit({"a": "site_01", "b": "site_02"})
    rolling_one = dict(rolling.loo_raw("a"))
    rolling.commit({"a": "site_01", "b": "site_02"})
    rolling.commit({"a": "site_03", "b": "site_03"})
    rolling_after = dict(rolling.loo_raw("a"))
    window = RollingMemory(["a", "b"], 50)
    window.commit({"a": "site_05", "b": "site_05"})
    for _step in range(49):
        window.commit({"a": "site_01", "b": "site_02"})
    length_at_50 = len(window.commits)
    window.commit({"a": "site_03", "b": "site_04"})
    decay = DecayMemory(["a", "b"], 0.99)
    decay_before = dict(decay.loo_raw("a"))
    decay.commit({"a": "site_01", "b": "site_01"})
    first_total = decay.total["site_01"]
    decay.commit({"a": "site_02", "b": "site_01"})
    negative = empty_social(["a"])
    negative.own["a"]["site_01"] = 1
    raised = False
    try:
        popularity_scores(negative, "a")
    except RuntimeError:
        raised = True
    return {
        "cumulative_before_commit": before,
        "cumulative_after_one_commit": after,
        "rolling_before_commit": rolling_before,
        "rolling_after_one_commit": rolling_one,
        "rolling_window_2_after_expiration": rolling_after,
        "rolling_50_length_before_overflow": length_at_50,
        "rolling_50_length_after_51st": len(window.commits),
        "rolling_50_oldest_removed": window.commits[0]["a"] == "site_01" and window.commits[-1]["a"] == "site_03",
        "decay_before_commit": decay_before,
        "decay_first_total_site_01": first_total,
        "decay_second_loo": {place_id: decay.loo_raw("a")[place_id] for place_id in PLACE_IDS},
        "negative_loo_raises": raised,
    }


def _weight_zero_gate() -> dict[str, Any]:
    atlas = load_frozen_atlas()
    locations = load_initial_locations()
    agent_id = sorted(locations)[0]
    origin = locations[agent_id]
    occupancy = occupancy_of(locations)
    matrix = distance_matrix()
    places = place_index()
    ones = {place_id: 1.0 for place_id in PLACE_IDS}
    ramp = {place_id: float(index) for index, place_id in enumerate(PLACE_IDS)}
    base, total_one, social_one = cumulative_utilities(
        atlas, matrix, places, agent_id, origin, occupancy, ones, 0.0
    )
    _base_two, total_ramp, social_ramp = cumulative_utilities(
        atlas, matrix, places, agent_id, origin, occupancy, ramp, 0.0
    )
    _base_heavy, heavy, heavy_social = memory_utilities(
        atlas, matrix, places, agent_id, origin, occupancy, ones
    )
    return {
        "experiment_004_weight_zero_totals_match_base": total_one == base and total_ramp == base,
        "experiment_004_weight_zero_social_term": social_one == social_ramp == {place_id: 0.0 for place_id in PLACE_IDS},
        "experiment_005_utility_uses_module_weight": any(value != 0.0 for value in heavy_social.values()) and heavy != base,
    }


def _stored_anchor_hashes() -> dict[str, Any]:
    manifest_004 = _read_json(ROOT / "data" / "worlds" / "world_v0_4" / "run_manifest.json")
    manifest_005 = _read_json(ROOT / "data" / "worlds" / "world_v0_5" / "run_manifest.json")
    hashes_004 = manifest_004["trajectory_sha256"]
    hashes_005 = manifest_005["trajectory_sha256"]
    pairs = [
        hashes_004[f"S150:{seed}"] == hashes_005[f"CUM:{seed}"]
        for seed in range(50)
    ]
    return {
        "seeds": 50,
        "comparison": "stored_manifest_hashes_only",
        "future_runner_lineage_executed": False,
        "cum_hash_equals_s150_hash": all(pairs),
        "S0:0": hashes_004["S0:0"],
        "S150:0": hashes_004["S150:0"],
        "CUM:0": hashes_005["CUM:0"],
        "RW50:0": hashes_005["RW50:0"],
        "EXP99:0": hashes_005["EXP99:0"],
        "S0:49": hashes_004["S0:49"],
        "S150:49": hashes_004["S150:49"],
        "CUM:49": hashes_005["CUM:49"],
        "RW50:49": hashes_005["RW50:49"],
        "EXP99:49": hashes_005["EXP99:49"],
    }


def write_records() -> None:
    from agents.persona_generator import write_json

    report = run_checks()
    write_json(DESIGN_DIR / "formula_report.json", report)
    write_json(DESIGN_DIR / "source_record.json", _source_record())


def _source_record() -> dict[str, Any]:
    read_files = [
        "world/experiment_004.py",
        "world/experiment_005.py",
        "world/geography.py",
        "world/experiment_002.py",
        "world/dynamics_calibration.py",
        "world/utility_atlas.py",
        "data/worlds/world_v0_4/world_v0_4_config.json",
        "data/worlds/world_v0_4/experiment_004_summary.json",
        "data/worlds/world_v0_4/run_manifest.json",
        "data/worlds/world_v0_5/world_v0_5_config.json",
        "data/worlds/world_v0_5/experiment_005_summary.json",
        "data/worlds/world_v0_5/run_manifest.json",
    ]
    sized_only = [
        "data/worlds/world_v0_4/world_states.jsonl",
        "data/worlds/world_v0_4/social_signal_reference_s0.jsonl",
        "data/worlds/world_v0_4/agent_actions.jsonl",
        "data/worlds/world_v0_5/world_states.jsonl",
        "data/worlds/world_v0_5/agent_actions.jsonl",
    ]
    return {
        "record_type": "experiment_017_design_source_record",
        "status": "DESIGN_ONLY",
        "run_status": "NOT_RUN",
        "not_an_experiment_run_manifest": True,
        "content_hashed": [_file_record(ROOT / relative) for relative in read_files],
        "availability_only": [_size_record(ROOT / relative) for relative in sized_only],
        "not_scanned": "Large JSONL contents were not read. Layer A remains uncomputed.",
        "llm_calls": 0,
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _size_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "content_read": False,
        "content_hashed_this_round": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    write_records()
