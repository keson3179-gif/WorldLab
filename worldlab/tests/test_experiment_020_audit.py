from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from world.experiment_017 import file_sha256


def _load():
    path = Path(__file__).resolve().parents[1] / "docs" / "experiment_020_audit" / "audit_experiment_020.py"
    spec = importlib.util.spec_from_file_location("audit_experiment_020", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = _load()


def _finished(elapsed: float = 3.5) -> bytes:
    line = json.dumps({"stage": "finished", "elapsed_seconds": elapsed, "status": "COMPLETE", "llm_calls": 0}, separators=(",", ":"))
    return (line + "\n").encode("utf-8")


def _world(condition: str, schedule: str, eligible: bool, exit_time: int | None, release_time: int | None) -> dict:
    return {
        "condition_id": condition,
        "schedule": schedule,
        "maintenance": {
            "eligible": eligible,
            "maintained_to_horizon": False if eligible else None,
            "released": release_time is not None if eligible else None,
            "late_run_other": False if eligible else None,
            "original_exit": exit_time,
            "release_time": release_time,
        },
    }


def test_median_51_with_a_later_exit_is_not_described_as_all_at_51() -> None:
    statement = AUDIT.describe_exit_times({"51": 978, "52": 1})
    assert statement["median"] == 51
    assert statement["all_at_51"] is False
    assert "t=51–52" in statement["text"]
    assert "中位數為 t=51" in statement["text"]
    assert "每一個合格 seed 都在 t=51 解除集中" not in statement["text"]


def test_interval_zero_coverage_includes_a_small_positive_interval() -> None:
    assert AUDIT.interval_contains_zero(0.0, 0.0) is True
    assert AUDIT.interval_contains_zero(-1e-6, 1e-6) is True
    assert AUDIT.interval_contains_zero(2.0166250000004103e-06, 2.4608343750000926e-05) is False
    assert AUDIT.interval_contains_zero(1.0, 1.0) is False


def test_manifest_all_match_and_exact_finished_append(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    body = b'{"stage":"protocol_ready"}\n'
    progress = root / "progress.jsonl"
    progress.write_bytes(body)
    other = root / "note.json"
    other.write_bytes(b'{"ok":true}\n')
    listed = {"progress.jsonl": file_sha256(progress), "note.json": file_sha256(other)}
    matched = AUDIT.audit_listed_files(root, listed)
    assert matched["manifest_all_matched"] is True
    assert matched["archive_fully_verified"] is True
    assert matched["progress_explained_as_final_finished_append"] is False
    progress.write_bytes(body + _finished())
    explained = AUDIT.audit_listed_files(root, listed)
    assert explained["manifest_all_matched"] is False
    assert explained["checked_count"] == 2
    assert explained["match_count"] == 1
    assert explained["other_listed_files_match"] is True
    assert explained["progress_explained_as_final_finished_append"] is True
    assert explained["archive_fully_verified"] is False
    detail = explained["mismatches"][0]["progress"]
    assert detail["prefix_matches_manifest"] is True
    assert detail["remainder_status"] == "COMPLETE"
    assert detail["remainder_elapsed_seconds"] == 3.5


def test_middle_edits_extra_appends_and_other_records_are_not_the_same_exception(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    body = b'{"stage":"protocol_ready"}\n{"stage":"worlds"}\n'
    progress = root / "progress.jsonl"
    progress.write_bytes(body)
    expected = file_sha256(progress)
    listed = {"progress.jsonl": expected}
    changed = bytearray(body + _finished())
    changed[2] = changed[2] ^ 1
    progress.write_bytes(bytes(changed))
    middle = AUDIT.audit_listed_files(root, listed)
    assert middle["progress_explained_as_final_finished_append"] is False
    extra = body + _finished() + b'{"stage":"finished","status":"COMPLETE"}\n'
    progress.write_bytes(extra)
    appended = AUDIT.audit_listed_files(root, listed)
    assert appended["progress_explained_as_final_finished_append"] is False
    progress.write_bytes(body + b'{"stage":"worlds","status":"COMPLETE"}\n')
    other = AUDIT.audit_listed_files(root, listed)
    assert other["manifest_all_matched"] is False
    assert other["progress_explained_as_final_finished_append"] is False
    assert other["mismatches"][0]["progress"]["remainder_stage"] == "worlds"


def test_byte_comparison_keeps_original_newlines(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    raw = b'{"stage":"protocol_ready"}\n{"stage":"finished","status":"COMPLETE"}\n'
    path.write_bytes(raw)
    assert file_sha256(path) == AUDIT.sha256_bytes(raw)
    assert AUDIT.sha256_bytes(raw) != AUDIT.sha256_bytes(raw.replace(b"\n", b"\r\n"))
    prefix, remainder = AUDIT.split_last_jsonl_record(raw)
    assert prefix + remainder == raw
    assert prefix == b'{"stage":"protocol_ready"}\n'
    assert remainder.endswith(b"\n")
    assert b"\r" not in prefix and b"\r" not in remainder


def test_audit_does_not_change_source_bytes(tmp_path: Path) -> None:
    root = tmp_path / "source"
    seeds = root / "seeds"
    seeds.mkdir(parents=True)
    (seeds / "1.json").write_text(
        json.dumps({"worlds": [_world("CUM_LOGMAX__q25", "EO", True, 52, 52)]}),
        encoding="utf-8",
        newline="\n",
    )
    progress = root / "progress.jsonl"
    progress.write_bytes(b'{"stage":"protocol_ready"}\n')
    before = {path: file_sha256(path) for path in (seeds / "1.json", progress)}
    aggregated = AUDIT.aggregate_maintenance(seeds)
    listed = {"progress.jsonl": before[progress]}
    AUDIT.audit_listed_files(root, listed)
    after = {path: file_sha256(path) for path in before}
    assert after == before
    assert aggregated["conditions"]["CUM_LOGMAX__q25"]["EO"]["original_exit"] == {"52": 1}
    output = tmp_path / "audit_results.json"
    AUDIT.write_audit_json(output, {"ok": True, "llm_calls": 0})
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True
    with pytest.raises(RuntimeError, match="data/"):
        AUDIT.refuse_data_write(AUDIT.DATA_ROOT / "worlds" / "world_v0_20" / "new.json")
