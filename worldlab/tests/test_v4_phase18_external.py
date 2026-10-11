from __future__ import annotations

from pathlib import Path

import pytest

from worldlab_observation.external_evidence import (
    PAGEVIEW_END,
    PAGEVIEW_START,
    ProspectiveRegistry,
    classify_article,
    coverage_status,
    month_span,
    select_alternate_candidates,
)


def test_park_article_is_not_the_site() -> None:
    result = classify_article("太魯閣國家公園遊客中心", "太魯閣國家公園", "花蓮縣", "太魯閣國家公園", "太魯閣國家公園位於花蓮", False)
    assert result["accepted"] == "false"
    assert result["match_evidence"] == "park_article_not_the_site"


def test_stem_match_needs_geography() -> None:
    rejected = classify_article("雪見遊憩區", "雪見", "苗栗縣", "雪霸國家公園", "一個沒有地名的句子", False)
    assert rejected["accepted"] == "false"
    accepted = classify_article("雪見遊憩區", "雪見", "苗栗縣", "雪霸國家公園", "位於苗栗縣的雪霸外圍", False)
    assert accepted["accepted"] == "true"
    assert accepted["match_evidence"] == "stem_title_with_geography"


def test_coverage_threshold_is_fixed_before_scores() -> None:
    expected = month_span(PAGEVIEW_START, PAGEVIEW_END)
    assert expected[0] == "2015-07"
    assert expected[-1] == "2023-12"
    assert coverage_status(expected)["usable_series"] is True
    assert coverage_status(expected[:10])["usable_series"] is False


def test_alternate_candidates_ignore_forecast_scores() -> None:
    rows = [
        {"site_uid": "s1", "official_name": "野柳地質公園", "county": "新北市", "parent_area": "北海岸", "match_status": "provisional_rule_match", "first_year": "2012", "last_year": "2023"},
        {"site_uid": "s2", "official_name": "野柳海洋世界", "county": "新北市", "parent_area": "北海岸", "match_status": "provisional_rule_match", "first_year": "2012", "last_year": "2023"},
        {"site_uid": "s3", "official_name": "某遊客中心", "county": "臺東縣", "parent_area": "東部", "match_status": "provisional_rule_match", "first_year": "2012", "last_year": "2023"},
        {"site_uid": "s4", "official_name": "短系列", "county": "臺南市", "parent_area": "南", "match_status": "singleton", "first_year": "2020", "last_year": "2023"},
    ]
    chosen = select_alternate_candidates(rows, set())
    assert [item["official_name"] for item in chosen] == ["野柳地質公園"]


def test_snapshot_cannot_be_overwritten_and_revision_keeps_the_first_hash(tmp_path: Path) -> None:
    registry = ProspectiveRegistry(tmp_path)
    record = _snapshot("snap-1")
    stored = registry.write_snapshot(record, b"first-publication")
    with pytest.raises(FileExistsError):
        registry.write_snapshot(record, b"rewritten")
    revised = registry.revise("snap-1", "snap-2", _snapshot("snap-2"), b"later-revision")
    assert revised["revises"] == "snap-1"
    assert stored["content_sha256"] != revised["content_sha256"]
    assert (tmp_path / "snap-1" / "content.bin").read_bytes() == b"first-publication"


def test_forecast_must_be_frozen_before_the_outcome(tmp_path: Path) -> None:
    registry = ProspectiveRegistry(tmp_path)
    registry.freeze_forecast("f-1", b"prediction", b"config", b"code-v1", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01", "spec-only")
    with pytest.raises(ValueError, match="not frozen"):
        registry.freeze_forecast("f-2", b"prediction", b"config", b"code-v1", "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01", "spec-only")
    manifest = (tmp_path / "forecasts" / "f-1" / "forecast.json").read_bytes()
    registry.record_first_observation("f-1", b"first-outcome", "2026-02-01T00:00:00Z", "2026-02-02T00:00:00Z", "https://example.test/outcome", "status=200")
    assert (tmp_path / "forecasts" / "f-1" / "forecast.json").read_bytes() == manifest
    with pytest.raises(ValueError, match="in place"):
        registry.replace_prediction("f-1", "new-hash")
    other = registry.freeze_forecast("f-3", b"other", b"config", b"code-v1", "2026-01-01T00:00:00Z", "2026-01-15T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01", "spec-only")
    assert other["forecast_id"] == "f-3"
    failure = registry.record_failure({"failure_id": "api-1", "kind": "api_failure", "retrieved_at": "2026-01-01T00:00:00Z"})
    assert failure["kind"] == "api_failure"


def test_runner_does_not_name_holdout_visit_files() -> None:
    source = Path("worldlab_observation/phase18_run.py").read_text(encoding="utf-8")
    assert "staging" not in source
    assert "v4_visits_" not in source
    assert "2025" not in source.split("READ_PATHS")[1].split(")")[0]
    assert "V4_PHASE17_GO_NO_GO.md" in source


def _snapshot(snapshot_id: str) -> dict[str, str]:
    return {
        "snapshot_id": snapshot_id,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "source_url": "https://example.test/visits",
        "http_metadata": "status=200",
        "data_version": "v0",
        "latest_observation_month": "2024-06",
        "available_for_forecast_at": "2026-02-01T00:00:00Z",
        "forecast_origin": "2026-01",
        "model_version": "spec-only",
        "forecast_manifest_hash": "abc",
        "prediction_file_hash": "def",
    }


