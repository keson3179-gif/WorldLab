from __future__ import annotations

from pathlib import Path

import pytest

from worldlab_data.phase13 import shift_month
from worldlab_observation.intensity import (
    BETA_GRID,
    HALF_LIFE_MONTHS,
    PHASE16_PARENTS,
    PI_FLOOR,
    PREDICTION_QUANTITY,
    ROLLING_MONTHS,
    TEMPERATURE,
    accepts_attribute_for_year,
    alpha_from_shares,
    calendar_window,
    center_and_scale,
    eligible_internal_months,
    frozen_parents,
    intensity_softmax,
    m0_components,
    m1_shares,
    m2_generation_status,
    prediction_ignores_target,
    refuse_aliased_recent_alpha,
    refuse_independent_social_claim,
    refuse_quantity_alias,
    scale_equivalent,
    select_beta,
    shares_close,
    signal_vector,
    smooth_and_normalize,
)

MEMBERSHIP = ("a", "b", "c")


def _history(start: str, count: int, shares: list[float] | None = None) -> dict[str, dict[str, float]]:
    values = shares or [0.5, 0.3, 0.2]
    history = {}
    month = start
    for _index in range(count):
        history[month] = {uid: values[index] for index, uid in enumerate(MEMBERSHIP)}
        month = shift_month(month, 1)
    return history


def test_alpha_training_only() -> None:
    history = _history("2012-01", 36)
    target = "2014-06"
    components = m0_components(history, MEMBERSHIP, target, "ex_post_max_information")
    assert components["information_cutoff"] == "2014-05"
    assert all(month <= "2014-05" and month < target for month in components["source_months"])
    history["2014-06"] = {"a": 0.9, "b": 0.05, "c": 0.05}
    history["2014-07"] = {"a": 0.1, "b": 0.1, "c": 0.8}
    shadowed = m0_components(history, MEMBERSHIP, target, "ex_post_max_information")
    assert components["source_months"] == shadowed["source_months"]
    assert shares_close(components["shares"], shadowed["shares"])
    assert components["reparameterization_gap"] < 1e-9


def test_softmax_invariance() -> None:
    alpha = alpha_from_shares([0.2, 0.5, 0.3])
    base = intensity_softmax(alpha)
    shifted = intensity_softmax([value + 5.0 for value in alpha])
    assert shares_close(base, shifted)
    assert TEMPERATURE == 1.0
    smoothed = smooth_and_normalize([0.2, 0.5, 0.3])
    assert shares_close(intensity_softmax(alpha_from_shares([0.2, 0.5, 0.3])), smoothed)
    assert PI_FLOOR > 0.0


def test_beta_zero_equals_M0() -> None:
    history = _history("2012-01", 30)
    components = m0_components(history, MEMBERSHIP, "2014-01", "ex_post_max_information")
    signal = signal_vector(history, MEMBERSHIP, "2013-12", "2014-01", "OBS_ROLLING")
    predicted = m1_shares(components, signal, 0.0)
    assert predicted["status"] == "predicted"
    assert shares_close(predicted["shares"], components["shares"])
    assert 0.0 in BETA_GRID


def test_signal_scale_identifiability() -> None:
    alpha = alpha_from_shares([0.2, 0.5, 0.3])
    scores = center_and_scale([0.1, 0.6, 0.3])
    assert scale_equivalent(alpha, scores, 1.0, 2.0)
    cumulative = signal_vector(_history("2012-01", 24), MEMBERSHIP, "2013-12", "2014-01", "OBS_CUM")
    assert cumulative["same_information_as_training_mean"] is True


def test_no_double_counting_claim() -> None:
    with pytest.raises(ValueError, match="agent choice"):
        refuse_quantity_alias("agent_choice_probability")
    with pytest.raises(ValueError, match="social-influence"):
        refuse_independent_social_claim(True)
    with pytest.raises(ValueError, match="same recent window"):
        refuse_aliased_recent_alpha(["2014-04", "2014-05", "2014-06"], ["2014-04", "2014-05", "2014-06"], "OBS_ROLLING")
    assert PREDICTION_QUANTITY == "aggregate_intensity_share"
    refuse_independent_social_claim(False)


def test_observation_memory_calendar() -> None:
    assert ROLLING_MONTHS == 3
    assert HALF_LIFE_MONTHS == 6.0
    assert ROLLING_MONTHS != 50
    assert calendar_window("2014-06", ROLLING_MONTHS) == ["2014-04", "2014-05", "2014-06"]
    signal = signal_vector(_history("2012-01", 30), MEMBERSHIP, "2014-06", "2014-07", "OBS_EXP")
    assert signal["memory_unit"] == "calendar_month"
    assert signal["half_life_months"] == 6.0


def test_lagged_data_cutoff() -> None:
    history = _history("2012-01", 40)
    assert prediction_ignores_target(history, MEMBERSHIP, "2014-06", "assumed_publication_delay")
    delayed = m0_components(history, MEMBERSHIP, "2014-06", "assumed_publication_delay")
    assert delayed["information_cutoff"] == "2014-03"
    assert all(month <= "2014-03" for month in delayed["source_months"])


def test_nested_parameter_selection() -> None:
    history = _history("2012-01", 48)
    selected = select_beta(history, MEMBERSHIP, "2015-01", "ex_post_max_information", "OBS_ROLLING")
    assert selected["status"] == "selected"
    assert selected["beta"] == 0.0
    assert all(month < "2015-01" for month in selected["internal_months"])
    history["2015-01"] = {"a": 0.05, "b": 0.05, "c": 0.9}
    again = select_beta(history, MEMBERSHIP, "2015-01", "ex_post_max_information", "OBS_ROLLING")
    assert again["beta"] == selected["beta"]
    assert again["internal_tv"] == selected["internal_tv"]
    short = _history("2013-06", 8)
    blocked = select_beta(short, MEMBERSHIP, "2014-02", "ex_post_max_information", "OBS_CUM")
    assert blocked["status"] == "insufficient_history"
    assert eligible_internal_months(short, MEMBERSHIP, "2014-02", "ex_post_max_information") == []


def test_group_membership_frozen() -> None:
    assert frozen_parents(["玉山國家公園", "太魯閣國家公園", "雪霸國家公園"]) == PHASE16_PARENTS
    with pytest.raises(RuntimeError, match="frozen"):
        frozen_parents(["玉山國家公園", "太魯閣國家公園"])


def test_historical_attribute_validity() -> None:
    assert accepts_attribute_for_year("current_snapshot_only", 2026, 2014) is False
    assert accepts_attribute_for_year("stable_unconfirmed", 2026, 2014) is False
    assert accepts_attribute_for_year("historically_documented", 2012, 2014) is True
    assert m2_generation_status(0) == "not_run"
    assert m2_generation_status(2) == "not_run"


def test_no_2025_access() -> None:
    source = Path("worldlab_observation/phase16_run.py").read_text(encoding="utf-8")
    assert "staging" not in source
    assert "v4_visits_" not in source
    assert "v4_baseline_scores" not in source
    assert "2025" not in source.split("READ_PATHS")[1].split(")")[0]


def test_original_worldlab_unchanged() -> None:
    source = Path("worldlab_observation/phase16_run.py").read_text(encoding="utf-8")
    for relative in ("world/choice.py", "world/geography.py", "world/experiment_020.py", "worldlab_data/phase13.py"):
        assert relative in source
    assert "protected files changed" in source
