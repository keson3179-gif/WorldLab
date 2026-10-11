"""World v0.14 long-horizon persistence tests. No model calls."""

from __future__ import annotations

import hashlib
import json
import tracemalloc
from pathlib import Path

import pytest

import world.experiment_013 as experiment_013
import world.experiment_014 as experiment
from world.dynamics_calibration import load_initial_locations
from world.experiment_004 import trajectory_hash
from world.experiment_014 import (
    ALTERNATION_UPPER_QUARTILE,
    BLEND_TOLERANCE,
    BOOTSTRAP_DRAWS,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    DELAYED_SENTENCE,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    HELDOUT_SEEDS,
    HORIZONS,
    INTERMITTENT_SENTENCE,
    JACCARD_STABLE_MIN,
    ROBUST_SENTENCE,
    S0_LAMBDAS,
    SOCIAL_LAMBDAS,
    TIMESTEPS,
    TRANSIENT_SENTENCE,
    TURNOVER_SENTENCE,
    divergence_duty_cycle,
    divergence_episodes,
    horizon_onset,
    jaccard,
    load_intervention_registry,
    load_lineage_references,
    long_horizon_class,
    onset_stability,
    overall_conclusion,
    simulate_history,
    terminal_attenuation_time,
    terminal_exact_reconvergence_time,
    transition_counts,
)
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = {
    "data/worlds/world_v0_7/run_manifest.json": "dc1e1f26e37c02f96c3331282a9438d6ef9bfa4cae6cf629d566093079a660fb",
    "data/worlds/world_v0_8/run_manifest.json": "b98b35b7d3ee8e9b8e26bffd90b1f47e6a69d8a58e06c89588637ee8124f0aed",
    "data/worlds/world_v0_9/run_manifest.json": "84840c8f74b7297c251007a05073dda07d053f6c964de39deb8de770806a03d4",
    "data/worlds/world_v0_10/run_manifest.json": "81cdc4894894a19843e45263164da42d957b855778c4b05b732a83be5f089318",
    "data/worlds/world_v0_11/run_manifest.json": "cff55b962e9cb7a9edd7ce0bd9d13efbe0d69ebeca02c857bfd2b7537a66cd45",
    "data/worlds/world_v0_12/run_manifest.json": "c6db41cc01effac751991a7ead5af595dc8fb2f9c214a2bfc468d9eea4c7c2fb",
    "data/worlds/world_v0_13/run_manifest.json": "43b06950a1f03f13d8cb6b3e471f72f332366c8380fd835cf517acb68882f419",
    "data/worlds/world_v0_13/experiment_013_summary.json": "251b60eefec578be0d44a1e444fe552a91faba5200d5a94e3063e99805af7bd6",
    "data/worlds/world_v0_13/world_v0_13_config.json": "e4d28e5a6f9215fed49275e61bdead5a6cdda8b819a247e18ef55bdef9970595",
    "data/worlds/world_v0_13/heldout_interventions.jsonl": "006b49b6dbc7252c0c88dfe39c77c2608ed1180b207a973848b77fca3cc9c136",
    "data/worlds/utility_atlas_v1/atlas_manifest.json": "90ff0feb2f846cf521d4a8d9e1b24e0e8f799e4a409a51c84daea374f744f9f1",
}


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


@pytest.fixture(scope="module")
def plans():
    return load_intervention_registry()


def test_rules_thresholds_and_frozen_inputs() -> None:
    assert TIMESTEPS == 2000
    assert HELDOUT_SEEDS == tuple(range(50, 100))
    assert SOCIAL_LAMBDAS == (0.3, 0.35, 0.4, 1.0)
    assert S0_LAMBDAS == (0.0, 1.0)
    assert [item[0] for item in HORIZONS] == ["H500", "H1000", "H1500", "H2000"]
    assert BOOTSTRAP_DRAWS == 10000
    assert BLEND_TOLERANCE == 1e-12
    assert JACCARD_STABLE_MIN == 0.5
    assert (ALTERNATION_UPPER_QUARTILE, CONSECUTIVE_TV_UPPER_QUARTILE) == (0.694375, 0.36148125)
    assert (EXTREME_LATE_HHI, EXTREME_DOMINANT_SHARE, EXTREME_SATURATION) == (0.40, 0.60, 0.10)
    assert long_horizon_class([0.0, 0.0, 0.0, 0.0]) == "STABLE_ATTENUATION"
    assert long_horizon_class([0.03, 0.04, 0.05, 0.06]) == "DURABLE_PERSISTENCE"
    assert long_horizon_class([0.03, 0.04, 0.01, 0.0]) == "TRANSIENT_PERSISTENCE"
    assert long_horizon_class([0.0, 0.0, 0.01, 0.04]) == "DELAYED_PERSISTENCE"
    assert long_horizon_class([0.03, 0.0, 0.04, 0.05]) == "INTERMITTENT"
    onset = horizon_onset({0.3: 0.01, 0.35: 0.02, 0.4: 0.03, 1.0: 0.09})
    assert onset["horizon_onset_band"]["label"] == "(0.35, 0.40]"
    early = horizon_onset({0.3: 0.03, 0.35: 0.04, 0.4: 0.05, 1.0: 0.06})
    assert early["horizon_onset_band"]["label"] == "(-, 0.30]"
    assert horizon_onset({0.3: 0.0, 0.35: 0.04, 0.4: 0.01, 1.0: 0.02})["status"] == "NONMONOTONIC"
    stable = [onset, onset, onset, onset]
    assert onset_stability(stable) == "ONSET_STABLE"
    higher = horizon_onset({0.3: 0.0, 0.35: 0.0, 0.4: 0.0, 1.0: 0.08})
    assert onset_stability([onset, onset, higher, higher]) == "ONSET_SHIFTS_UP"
    assert onset_stability([higher, onset, higher, onset]) == "ONSET_HORIZON_NONMONOTONIC"
    assert jaccard(set(), set()) is None
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert transition_counts([False, True, True, False], [False, False, True, True]) == {"AA": 1, "AP": 1, "PA": 1, "PP": 1}
    values = [0.0] * 501 + [0.03] * 10 + [0.0] * 10 + [0.04] * 5 + [0.0] * (2000 - 525)
    assert divergence_duty_cycle(values) == pytest.approx(15 / 1500)
    episodes = divergence_episodes(values)
    assert episodes["episode_count"] == 2
    assert episodes["max_episode_length"] == 10
    assert episodes["last_divergent_t"] == 525
    assert terminal_attenuation_time(values) == 526
    assert terminal_attenuation_time(values[:-1] + [0.05]) is None
    assert terminal_exact_reconvergence_time([0.0] * 10 + [0.01] + [0.0] * 5) == 11
    assert terminal_exact_reconvergence_time([0.0, 0.2]) is None
    early_only = [0.1] * 22 + [0.0] * 30 + [0.05] * 20
    assert experiment_013.stable_attenuation_time(early_only) == 41
    assert terminal_attenuation_time(early_only) is None
    references = load_lineage_references()
    assert references["h500"]["RW50"]["0.40"] == pytest.approx(0.03191)
    assert references["h500"]["EXP99"]["0.35"] == pytest.approx(0.02793)
    for path, digest in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    sample = {
        "RW50": {
            "onset_stability": "ONSET_STABLE",
            "levels": {"0.40": {"long_horizon_class": "DURABLE_PERSISTENCE", "turnover_interpretation": True, "stable_subset_interpretation": False}},
        },
        "EXP99": {
            "onset_stability": "ONSET_STABLE",
            "levels": {"0.35": {"long_horizon_class": "TRANSIENT_PERSISTENCE", "turnover_interpretation": False, "stable_subset_interpretation": False}},
        },
    }
    text = overall_conclusion(sample)
    assert ROBUST_SENTENCE in text
    assert TRANSIENT_SENTENCE in text
    assert TURNOVER_SENTENCE in text
    assert DELAYED_SENTENCE not in text
    mixed = {
        "RW50": {"onset_stability": "ONSET_SHIFTS_UP", "levels": {"0.40": {"long_horizon_class": "INTERMITTENT", "turnover_interpretation": False, "stable_subset_interpretation": False}}},
        "EXP99": {"onset_stability": "ONSET_STABLE", "levels": {"0.35": {"long_horizon_class": "STABLE_ATTENUATION", "turnover_interpretation": False, "stable_subset_interpretation": False}}},
    }
    mixed_text = overall_conclusion(mixed)
    assert INTERMITTENT_SENTENCE in mixed_text
    assert ROBUST_SENTENCE not in mixed_text


def test_registry_is_reused_and_prefix_matches_experiment_013(atlas, locations, plans) -> None:
    plan = plans[("RW50", 50)]
    assert plan["override_count"] == 2
    assert plan["intervention_timestep"] == 21
    assert plan["override_consumes_rng"] == 0
    assert plan["generator"] == "world.experiment_007.plan_intervention"
    spec = next(item for item in experiment.REGIMES if item["regime_id"] == "RW50")
    references = load_lineage_references()
    donors = {}
    for history in ("CONTROL", "PERTURBED"):
        result = simulate_history(
            atlas,
            locations,
            spec,
            choice_seed_value=50,
            history=history,
            lambda_value=1.0,
            plan=plan,
            donors=None,
            timesteps=500,
            require_donors=False,
        )
        donors[history] = result["endogenous_field"]
        assert trajectory_hash(result["series"]) == references["trajectory_sha256"][f"RW50:50:{history}:1.00"]
        assert result["override_count"] == (2 if history == "PERTURBED" else 0)
    blended = simulate_history(
        atlas,
        locations,
        spec,
        choice_seed_value=50,
        history="PERTURBED",
        lambda_value=0.4,
        plan=plan,
        donors=donors,
        timesteps=500,
    )
    assert blended["trajectory_sha256"] == references["trajectory_sha256"]["RW50:50:PERTURBED:0.40"]
    assert blended["max_blend_error"] <= BLEND_TOLERANCE
    assert blended["rng_draw_counts"] == result["rng_draw_counts"]


def test_nonempty_output_is_refused_without_changing_files(tmp_path) -> None:
    target = tmp_path / "world_v0_14"
    target.mkdir()
    summary = target / "experiment_014_summary.json"
    actions = target / "agent_actions.jsonl"
    summary.write_bytes(b'{"frozen": true}\n')
    actions.write_bytes(b"choice-row\n")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()}
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        experiment.run_experiment(output_dir=target)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()}
    assert after == before
    assert summary.read_bytes() == b'{"frozen": true}\n'
    assert actions.read_bytes() == b"choice-row\n"


def test_file_sha256_streams_fixed_chunks(tmp_path) -> None:
    payload = b"abc" * 100000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert experiment._file_sha256(path, chunk_size=64) == hashlib.sha256(payload).hexdigest()

    def peak(size: int) -> int:
        sample = tmp_path / f"{size}.bin"
        sample.write_bytes(b"x" * size)
        tracemalloc.start()
        experiment._file_sha256(sample, chunk_size=4096)
        _current, high = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return high

    small = peak(200_000)
    large = peak(2_000_000)
    assert large < small + 500_000


def test_recorded_run_replays_selected_worlds(atlas, locations, plans) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_14"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_014_summary.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["world_count"] == 1000
    assert manifest["timesteps"] == 2000
    assert manifest["llm_calls"] == 0
    assert summary["gates"]["first_500_lineage"] is True
    assert summary["gates"]["h500_replication"] is True
    assert summary["gates"]["s0_lambda_equivalence"] is True
    assert summary["gates"]["numerical_stability"] is True
    requested = (("RW50", "PERTURBED", 0.4, 50), ("S0", "CONTROL", 0.0, 50), ("S0", "CONTROL", 1.0, 50))
    for regime_id, history, lambda_value, seed in requested:
        spec = next(item for item in experiment.REGIMES if item["regime_id"] == regime_id)
        donors = {}
        for donor_history in ("CONTROL", "PERTURBED"):
            donors[donor_history] = simulate_history(
                atlas,
                locations,
                spec,
                choice_seed_value=seed,
                history=donor_history,
                lambda_value=1.0,
                plan=plans[(regime_id, seed)],
                donors=None,
                timesteps=2000,
                require_donors=False,
            )["endogenous_field"]
        result = simulate_history(
            atlas,
            locations,
            spec,
            choice_seed_value=seed,
            history=history,
            lambda_value=lambda_value,
            plan=plans[(regime_id, seed)],
            donors=donors,
            timesteps=2000,
        )
        assert result["trajectory_sha256"] == manifest["trajectory_sha256"][f"{regime_id}:{seed}:{history}:{lambda_value:.2f}"]
