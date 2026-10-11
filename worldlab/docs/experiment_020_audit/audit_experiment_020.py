"""Read-only reporting audit for the frozen Experiment 020 archive.

This module does not resimulate worlds, refit weights, or recompute bootstrap
intervals. It refuses to write under data/.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world.experiment_017 import HASH_CHUNK_BYTES, file_sha256

DATA_ROOT = ROOT / "data"
ARCHIVE = DATA_ROOT / "worlds" / "world_v0_20"
AUDIT_DIR = ROOT / "docs" / "experiment_020_audit"
RUNNER = ROOT / "world" / "experiment_020.py"
GAP_EXPRESSION = "np.mean(np.abs(used_values - own_values))"
STORED_GAP_FIELDS = (
    "field_gap_t1_50",
    "field_gap_t51_100",
    "field_gap_t101_500",
    "field_gap_t1_500",
)


def sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    view = memoryview(payload)
    cursor = 0
    while cursor < len(view):
        digest.update(view[cursor : cursor + HASH_CHUNK_BYTES])
        cursor += HASH_CHUNK_BYTES
    return digest.hexdigest()


def refuse_data_write(path: Path) -> None:
    resolved = path.resolve()
    if DATA_ROOT.resolve() == resolved or DATA_ROOT.resolve() in resolved.parents:
        raise RuntimeError("audit refuses to write into data/")


def write_audit_json(path: Path, payload: dict[str, Any]) -> None:
    refuse_data_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    refuse_data_write(temporary)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _time_key(value: Any) -> str:
    if value is None:
        return "none"
    return str(int(value))


def _bump(counts: dict[str, int], value: Any) -> None:
    key = _time_key(value)
    counts[key] = counts.get(key, 0) + 1


def _empty_schedule() -> dict[str, Any]:
    return {
        "eligible_n": 0,
        "maintained_n": 0,
        "released_n": 0,
        "transfer_n": 0,
        "original_exit": {},
        "release_time": {},
    }


def aggregate_maintenance(seeds_dir: Path) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    seed_count = 0
    for path in sorted(seeds_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed_count += 1
        for row in payload["worlds"]:
            condition = str(row["condition_id"])
            schedule = str(row["schedule"])
            bucket = grouped.setdefault(condition, {}).setdefault(schedule, _empty_schedule())
            record = row["maintenance"]
            if not record["eligible"]:
                continue
            bucket["eligible_n"] += 1
            bucket["maintained_n"] += int(bool(record["maintained_to_horizon"]))
            bucket["released_n"] += int(bool(record["released"]))
            bucket["transfer_n"] += int(bool(record["late_run_other"]))
            _bump(bucket["original_exit"], record["original_exit"])
            _bump(bucket["release_time"], record["release_time"])
    return {"seed_files": seed_count, "conditions": grouped}


def describe_exit_times(counts: dict[str, int]) -> dict[str, Any]:
    observed: list[int] = []
    for key, count in counts.items():
        if key == "none":
            continue
        observed.extend([int(key)] * int(count))
    if not observed:
        return {
            "median": None,
            "minimum": None,
            "maximum": None,
            "all_at_51": False,
            "text": "沒有合格世界的退出時間。",
        }
    median = float(statistics.median(observed))
    minimum = min(observed)
    maximum = max(observed)
    all_at_51 = minimum == 51 and maximum == 51
    median_text = str(int(median)) if median == int(median) else str(median)
    if all_at_51:
        text = "所有合格世界的退出時間都是 t=51。"
    else:
        text = f"所有合格世界都在 t={minimum}–{maximum} 解除集中，退出時間的中位數為 t={median_text}。"
    return {
        "median": median,
        "minimum": minimum,
        "maximum": maximum,
        "all_at_51": all_at_51,
        "text": text,
    }


def interval_contains_zero(low: float, high: float) -> bool:
    return float(low) <= 0.0 <= float(high)


def early_given_o_intervals(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, item in sorted(analysis["summaries"].items()):
        if not name.startswith("early_given_O_"):
            continue
        low = float(item["ci_low"])
        high = float(item["ci_high"])
        rows.append(
            {
                "contrast": name,
                "mean": float(item["mean"]),
                "ci_low": low,
                "ci_high": high,
                "contains_zero": interval_contains_zero(low, high),
                "equivalence_criterion_defined": False,
            }
        )
    return rows


def inspect_field_gap(source_text: str) -> dict[str, Any]:
    endogenous = "used = own_scores" in source_text
    exogenous = "donor_cube[timestep, agent_index, place_index]" in source_text
    expression = GAP_EXPRESSION in source_text
    return {
        "display_name": "used–own field gap",
        "operation": "mean(abs(used_field - recipient_own_field))",
        "stored_field_names": list(STORED_GAP_FIELDS),
        "endogenous_assigns_used_to_own": endogenous,
        "exogenous_reads_donor_cube": exogenous,
        "gap_expression_present": expression,
        "definition_confirmed": endogenous and exogenous and expression,
        "endogenous_gap_is_zero_because_used_equals_own": endogenous,
        "measures_donor_versus_own_only_when_source_is_exogenous": exogenous and expression,
    }


def split_last_jsonl_record(payload: bytes) -> tuple[bytes, bytes]:
    if payload.endswith(b"\n"):
        cut = payload.rfind(b"\n", 0, len(payload) - 1)
    else:
        cut = payload.rfind(b"\n")
    if cut < 0:
        return b"", payload
    return payload[: cut + 1], payload[cut + 1 :]


def explain_progress_append(payload: bytes, expected_hash: str, actual_hash: str) -> dict[str, Any]:
    prefix, remainder = split_last_jsonl_record(payload)
    prefix_hash = sha256_bytes(prefix)
    one_record = remainder.endswith(b"\n") and remainder.count(b"\n") == 1 and b"\r" not in remainder
    record: dict[str, Any] | None = None
    if one_record:
        try:
            parsed = json.loads(remainder[:-1].decode("utf-8"))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            record = parsed
    finished = bool(record and record.get("stage") == "finished" and record.get("status") == "COMPLETE")
    explained = prefix_hash == expected_hash and finished and actual_hash != expected_hash
    return {
        "expected_hash": expected_hash,
        "actual_full_hash": actual_hash,
        "prefix_hash": prefix_hash,
        "prefix_byte_length": len(prefix),
        "remainder_byte_length": len(remainder),
        "prefix_matches_manifest": prefix_hash == expected_hash,
        "remainder_is_one_newline_terminated_record": record is not None,
        "remainder_stage": None if record is None else record.get("stage"),
        "remainder_status": None if record is None else record.get("status"),
        "remainder_elapsed_seconds": None if record is None else record.get("elapsed_seconds"),
        "explained_as_final_finished_append": explained,
    }


def audit_listed_files(root: Path, listed: dict[str, str]) -> dict[str, Any]:
    mismatches = []
    progress_detail = None
    for relative, expected in sorted(listed.items()):
        path = root.joinpath(*str(relative).split("/"))
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        actual = file_sha256(path)
        if actual == expected:
            continue
        detail = None
        if Path(str(relative)).name == "progress.jsonl":
            detail = explain_progress_append(path.read_bytes(), expected, actual)
            progress_detail = detail
        mismatches.append({"path": relative, "expected_hash": expected, "actual_hash": actual, "progress": detail})
    other = [item for item in mismatches if not str(item["path"]).endswith("progress.jsonl")]
    explained = bool(progress_detail and progress_detail["explained_as_final_finished_append"])
    return {
        "checked_count": len(listed),
        "match_count": len(listed) - len(mismatches),
        "manifest_all_matched": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "other_listed_files_match": not other,
        "progress_explained_as_final_finished_append": explained,
        "archive_fully_verified": not mismatches,
    }


def load_run_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload["files"]
    if not isinstance(files, dict):
        raise RuntimeError("run manifest files are not an object")
    return {str(key): str(value) for key, value in files.items()}


def quoted_runner_elapsed(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if "正式執行器耗時" in line]


def run_audit(archive: Path = ARCHIVE, runner: Path = RUNNER) -> dict[str, Any]:
    manifest_path = archive / "run_manifest.json"
    listed = load_run_manifest(manifest_path)
    maintenance = aggregate_maintenance(archive / "seeds")
    analysis = json.loads((archive / "uncertainty_analysis.json").read_text(encoding="utf-8"))
    completion = json.loads((archive / "completion_status.json").read_text(encoding="utf-8"))
    manifest_audit = audit_listed_files(ROOT, listed)
    docs_report = ROOT / "docs" / "EXPERIMENT_020_REPORT.md"
    data_report = archive / "EXPERIMENT_020_REPORT.md"
    progress_detail = None
    for item in manifest_audit["mismatches"]:
        if item.get("progress"):
            progress_detail = item["progress"]
    return {
        "record_type": "experiment_020_reporting_audit",
        "writes_into_data": False,
        "resimulated_worlds": False,
        "recomputed_bootstrap": False,
        "maintenance": maintenance,
        "early_given_o": early_given_o_intervals(analysis),
        "field_gap": inspect_field_gap(runner.read_text(encoding="utf-8")),
        "manifest": manifest_audit,
        "elapsed": {
            "completion_status_seconds": completion.get("elapsed_seconds"),
            "completion_status_meaning": "written before figures, the report, and the manifest",
            "report_generation_quotes": {
                "docs": quoted_runner_elapsed(docs_report.read_text(encoding="utf-8")),
                "data_archive": quoted_runner_elapsed(data_report.read_text(encoding="utf-8")),
            },
            "finished_progress_seconds": None if progress_detail is None else progress_detail.get("remainder_elapsed_seconds"),
            "finished_progress_meaning": "appended to progress.jsonl after the manifest was written",
        },
        "frozen_hashes": {
            "run_manifest": file_sha256(manifest_path),
            "progress_jsonl": file_sha256(archive / "progress.jsonl"),
            "data_report": file_sha256(data_report),
            "completion_status": file_sha256(archive / "completion_status.json"),
            "uncertainty_analysis": file_sha256(archive / "uncertainty_analysis.json"),
            "experiment_020_py": file_sha256(runner),
        },
        "llm_calls": 0,
    }


def main() -> None:
    payload = run_audit()
    write_audit_json(AUDIT_DIR / "audit_results.json", payload)
    manifest = payload["manifest"]
    print(
        f"checked={manifest['checked_count']} matched={manifest['match_count']} "
        f"all_matched={manifest['manifest_all_matched']} "
        f"progress_explained={manifest['progress_explained_as_final_finished_append']} "
        f"other_match={manifest['other_listed_files_match']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
