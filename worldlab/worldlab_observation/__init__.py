"""Observation-scale adapter. It does not modify the long-run WorldLab engine."""

from worldlab_observation.adapter import (
    AGENT_COUNT,
    ANALYSIS_ROLE,
    REQUIRED_REPLICATES_FOR_SHARE_SE,
    attribute_gaps,
    decision_q,
    expected_share,
    fair_one_step_status,
    logmax_scores,
    replicates_for_share_se,
    softmax_probabilities,
    zraw_scores,
)

__all__ = [
    "AGENT_COUNT",
    "ANALYSIS_ROLE",
    "REQUIRED_REPLICATES_FOR_SHARE_SE",
    "attribute_gaps",
    "decision_q",
    "expected_share",
    "fair_one_step_status",
    "logmax_scores",
    "replicates_for_share_se",
    "softmax_probabilities",
    "zraw_scores",
]
