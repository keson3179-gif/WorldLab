from __future__ import annotations

from pathlib import Path

from world.choice import softmax_choice
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE
from world.experiment_004 import SocialState, popularity_scores
from world.experiment_005 import centered_scores
from world.experiment_020 import SWITCH_INDEX, source_at
from world.geography import PLACE_IDS
from worldlab_data.phase12 import valid_share_vector
from worldlab_data.phase13 import information_cutoff
from worldlab_observation.adapter import (
    AGENT_COUNT,
    REQUIRED_REPLICATES_FOR_SHARE_SE,
    EndogenousHistory,
    ReportedHistory,
    SocialSignal,
    attribute_gaps,
    conditional_one_step,
    decision_q,
    expected_share,
    fair_one_step_status,
    free_running_step,
    logmax_scores,
    mae_equals_scaled_tv,
    monte_carlo_mean,
    refuse_synthetic_alias,
    replicates_for_share_se,
    scale_tradeoff_same_probability,
    softmax_probabilities,
    stay_adjusted_probability,
    undefined_share,
    zraw_scores,
)


def test_j3_probability_and_share_simplex() -> None:
    probabilities = softmax_probabilities([3.0, 1.0, 0.0])
    assert len(probabilities) == 3
    assert valid_share_vector(probabilities)
    share = expected_share([probabilities, [0.2, 0.3, 0.5]])
    assert valid_share_vector(share)
    _selected, engine = softmax_choice(["a", "b", "c"], [3, 1, 0], REFERENCE_CHOICE_TEMPERATURE, seed=7)
    direct = softmax_probabilities([3.0, 1.0, 0.0])
    assert all(abs(engine[place] - direct[index]) <= 1e-12 for index, place in enumerate(("a", "b", "c")))


def test_nonnegative_share_sums_to_one() -> None:
    stayed = stay_adjusted_probability([0.5, 0.5], 1, 0.5)
    assert min(stayed) >= 0.0
    assert abs(sum(stayed) - 1.0) <= 1e-12
    assert undefined_share(0.0) == "all_zero_group_total"
    assert undefined_share(12.0) is None


def test_same_seed_replays() -> None:
    left = monte_carlo_mean([0.7, 0.2, 0.1], 6, seed=14014, agent_count=AGENT_COUNT)
    right = monte_carlo_mean([0.7, 0.2, 0.1], 6, seed=14014, agent_count=AGENT_COUNT)
    assert left["mean_share"] == right["mean_share"]


def test_different_seeds_estimate_monte_carlo_error() -> None:
    assert replicates_for_share_se(AGENT_COUNT, 0.01) == REQUIRED_REPLICATES_FOR_SHARE_SE == 125
    first = monte_carlo_mean([0.7, 0.2, 0.1], 125, seed=1, agent_count=20)
    second = monte_carlo_mean([0.7, 0.2, 0.1], 125, seed=2, agent_count=20)
    assert first["mean_share"] != second["mean_share"]
    assert first["max_abs_error"] < 0.05
    assert second["max_abs_error"] < 0.05


def test_one_world_discreteness_and_averaging() -> None:
    one = monte_carlo_mean([0.7, 0.2, 0.1], 1, seed=14014, agent_count=20)
    averaged = monte_carlo_mean([0.7, 0.2, 0.1], 125, seed=14014, agent_count=20)
    assert one["one_world_tv"] > averaged["max_abs_error"]
    assert averaged["max_abs_error"] < one["one_world_tv"]


def test_information_cutoffs_and_separated_histories() -> None:
    places = ("a", "b", "c")
    reported = ReportedHistory(places)
    for month, share in (("2013-10", [0.5, 0.3, 0.2]), ("2013-11", [0.4, 0.4, 0.2]), ("2013-12", [0.2, 0.2, 0.6])):
        reported.add_observed(month, share)
    early = conditional_one_step(reported, "2014-01", "ex_post_max_information", None)
    delayed = conditional_one_step(reported, "2014-01", "assumed_publication_delay", None)
    assert early["information_cutoff"] == information_cutoff("2014-01", "ex_post_max_information") == "2013-12"
    assert delayed["information_cutoff"] == "2013-10"
    assert early["source_months"] == ["2013-10", "2013-11", "2013-12"]
    assert delayed["source_months"] == ["2013-10"]
    assert reported.months.keys() == set(early["reported_months_after"])
    endogenous = EndogenousHistory(places)
    try:
        endogenous.add_model_share("t1", [0.2, 0.3, 0.5], "reported_visit_share_conditional")
        raised = False
    except ValueError:
        raised = True
    assert raised
    signal = SocialSignal(places, [1.0, 0.0, 0.0], "reported_visit_share_conditional")
    assert signal.exposure_claim is False
    step = free_running_step(endogenous, "t1", [[0.2, 0.3, 0.5]], "model_endogenous_presence")
    assert step["uses_mid_run_reported_correction"] is False


def test_logmax_and_zraw_match_the_five_place_engine() -> None:
    counts = [0.0, 1.0, 2.0, 4.0, 8.0]
    state = SocialState(
        own={
            "focal": {place_id: 0 for place_id in PLACE_IDS},
            "other": {place_id: counts[index] for index, place_id in enumerate(PLACE_IDS)},
        },
        total={place_id: counts[index] for index, place_id in enumerate(PLACE_IDS)},
    )
    engine = popularity_scores(state, "focal")
    generalized = logmax_scores(counts)
    assert all(abs(engine[place_id] - generalized[index]) <= 1e-12 for index, place_id in enumerate(PLACE_IDS))
    raw = {place_id: counts[index] for index, place_id in enumerate(PLACE_IDS)}
    _unclipped, clipped = centered_scores(raw)
    assert all(abs(clipped[place_id] - zraw_scores(counts)[index]) <= 1e-12 for index, place_id in enumerate(PLACE_IDS))


def test_q_is_probability_distance_and_weight_is_not_separately_identified() -> None:
    value = decision_q([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], weight=10.0)
    assert 0.0 < value < 1.0
    assert decision_q([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], weight=0.0) == 0.0
    assert scale_tradeoff_same_probability([1.0, 2.0, 3.0], [0.0, 0.2, 1.0], 4.0)
    assert mae_equals_scaled_tv([0.5, 0.5], [0.2, 0.8])


def test_real_attributes_and_holdout_identity_gate() -> None:
    gaps = attribute_gaps({"group_id", "site_uid", "year_month", "share"})
    assert gaps["utility_status"] == "unavailable"
    assert gaps["invented_attributes"] is False
    assert "visual_attractiveness" in gaps["missing_attributes"]
    assert fair_one_step_status([True, False, True]) == "unavailable_incomplete_2024_identity"
    try:
        refuse_synthetic_alias("site_01", "stat_example")
        blocked = False
    except ValueError:
        blocked = True
    assert blocked
    assert source_at("EO", SWITCH_INDEX - 1) == "E"
    assert source_at("EO", SWITCH_INDEX) == "O"


def test_runner_does_not_name_holdout_visit_files() -> None:
    source = (Path(__file__).resolve().parents[1] / "worldlab_observation" / "smoke_run.py").read_text(encoding="utf-8")
    assert "staging" not in source
    assert "v4_visits_" not in source
