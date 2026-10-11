from __future__ import annotations

from pathlib import Path

import pytest

from worldlab_observation.external_evidence import EXTERNAL_TIME_ORDER, ProspectiveRegistry


def test_snapshot_id_cannot_escape_the_registry(tmp_path: Path) -> None:
    registry = ProspectiveRegistry(tmp_path)
    with pytest.raises(ValueError, match="safe identifier"):
        registry.write_snapshot({"snapshot_id": "../outside"}, b"escaped")
    assert not (tmp_path.parent / "outside").exists()


def test_interrupted_snapshot_write_is_not_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ProspectiveRegistry(tmp_path)

    def fail(path: Path, payload: dict[str, str]) -> None:
        del path, payload
        raise OSError("interrupted")

    monkeypatch.setattr("worldlab_observation.external_evidence._write_json", fail)
    with pytest.raises(OSError, match="interrupted"):
        registry.write_snapshot({"snapshot_id": "snap-torn"}, b"partial-body")
    with pytest.raises(RuntimeError, match="incomplete"):
        registry.verify_snapshot("snap-torn")
    with pytest.raises(FileExistsError):
        registry.write_snapshot({"snapshot_id": "snap-torn"}, b"retry")


def test_frozen_forecast_keeps_files_and_rejects_a_second_seal(tmp_path: Path) -> None:
    registry = ProspectiveRegistry(tmp_path)
    stored = registry.freeze_forecast(
        "f-1",
        b"prediction-v1",
        b"config-v1",
        b"code-v1",
        "2026-01-02T08:00:00+08:00",
        "2026-01-02T09:00:00+08:00",
        "2026-02-01T00:00:00Z",
        "2026-01",
        "spec-only",
    )
    assert stored["frozen_at"] == "2026-01-02T01:00:00Z"
    assert stored["external_time_order"] == EXTERNAL_TIME_ORDER
    registry.verify_forecast("f-1")
    manifest = (tmp_path / "forecasts" / "f-1" / "forecast.json").read_bytes()
    prediction = (tmp_path / "forecasts" / "f-1" / "prediction.bin").read_bytes()
    with pytest.raises(ValueError, match="in place"):
        registry.replace_prediction("f-1", "edited")
    registry.record_first_observation("f-1", b"first-bytes", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "https://example.test/a", "status=200")
    with pytest.raises(FileExistsError, match="twice"):
        registry.record_first_observation("f-1", b"other-bytes", "2026-02-01T00:00:00Z", "2026-02-03T00:00:00Z", "https://example.test/a", "status=200")
    revised = registry.revise_forecast("f-1", "f-2", b"prediction-v2", b"config-v2", b"code-v2", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01", "spec-only")
    later = registry.append_observation_revision("f-1", "rev-1", b"revised-bytes", "2026-03-01T00:00:00Z", "https://example.test/b", "status=200")
    assert revised["revises"] == "f-1"
    assert later["revises_first_retrieved_at"] == "2026-02-01T00:00:00Z"
    assert (tmp_path / "forecasts" / "f-1" / "forecast.json").read_bytes() == manifest
    assert (tmp_path / "forecasts" / "f-1" / "prediction.bin").read_bytes() == prediction
    assert (tmp_path / "observations" / "f-1" / "content.bin").read_bytes() == b"first-bytes"


def test_local_edit_fails_hash_verification(tmp_path: Path) -> None:
    registry = ProspectiveRegistry(tmp_path)
    registry.freeze_forecast("f-1", b"prediction-v1", b"config-v1", b"code-v1", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01", "spec-only")
    (tmp_path / "forecasts" / "f-1" / "prediction.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash"):
        registry.verify_forecast("f-1")


def test_naive_timestamp_and_early_retrieval_are_rejected(tmp_path: Path) -> None:
    registry = ProspectiveRegistry(tmp_path)
    with pytest.raises(ValueError, match="timezone"):
        registry.freeze_forecast("f-1", b"p", b"c", b"v", "2026-01-01T00:00:00", "2026-01-02T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01", "spec-only")
    registry.freeze_forecast("f-1", b"p", b"c", b"v", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01", "spec-only")
    with pytest.raises(ValueError, match="before it was published"):
        registry.record_first_observation("f-1", b"early", "2026-02-01T00:00:00Z", "2026-01-15T00:00:00Z", "https://example.test/a", "status=200")
