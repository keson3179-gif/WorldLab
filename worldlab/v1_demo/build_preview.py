"""Extract the 219 saved Experiment 020 gate trajectories into the V1 replay bundle.

This script reads three existing seed files. It does not run a world.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world.experiment_004 import trajectory_hash
from world.experiment_017 import file_sha256, unpack_series
from world.geography import PLACE_IDS, PLACES

from v1_demo.display_rules import LATE_HHI_LABEL, TIME_MARKS, early_text, event_text, maintenance_text

DATA = ROOT / "data" / "worlds" / "world_v0_20"
OUT = Path(__file__).resolve().parent / "export"
SEEDS = (1700, 2199, 2699)
MEMORIES = ("CUM", "RW50", "EXP99")
TRANSFORMS = ("LOGMAX", "ZRAW")
LABELS = ("q05", "q15", "q25")
SCHEDULES = ("EE", "EO", "OE", "OO")
SOURCE_NAMES = (
    "data/worlds/world_v0_20/trajectories/1700.json",
    "data/worlds/world_v0_20/trajectories/2199.json",
    "data/worlds/world_v0_20/trajectories/2699.json",
    "data/worlds/world_v0_20/seeds/1700.json",
    "data/worlds/world_v0_20/seeds/2199.json",
    "data/worlds/world_v0_20/seeds/2699.json",
)


def refuse_data_write(path: Path) -> None:
    resolved = path.resolve()
    data_root = (ROOT / "data").resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise RuntimeError(f"refusing to write into research data: {resolved}")


def write_json(path: Path, payload: Any) -> None:
    refuse_data_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_compact(path: Path, payload: Any) -> None:
    refuse_data_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _metric_block(row: dict[str, Any]) -> dict[str, Any]:
    event = row["event18"]
    maintenance = row["maintenance"]
    late = row["late"]
    return {
        "late_hhi": {
            "value": late["late_hhi"],
            "window": "committed t=101-500 cumulative presence",
            "display_zh": LATE_HHI_LABEL,
            "denominator": "20 agents times committed states t=101 through t=500",
            "role": "saved whole-replay summary",
            "late_dominant": late["late_dominant"],
            "tie": late["tie"],
        },
        "early_formation": {
            "value": bool(row["early_formed"]),
            "display_zh": early_text(bool(row["early_formed"])),
            "window": "18/20 run confirms at or before committed t=50",
            "denominator": "this trajectory",
        },
        "event18": {
            **event,
            "display": event_text(event),
            "window": "first run of 20 committed steps with at least 18 agents in one place",
            "denominator": "this trajectory",
        },
        "maintenance": {
            **maintenance,
            "display": maintenance_text(maintenance),
            "window": "eligibility uses committed t=31-50 and is fixed before the switch",
            "denominator": "this trajectory; ineligible fields stay null",
        },
    }


def _world_record(seed: int, condition: str, schedule: str, packed: dict[str, Any], metrics: dict[str, Any], source_trajectory: str) -> dict[str, Any]:
    series = unpack_series(packed)
    digest = trajectory_hash(series)
    if digest != metrics["trajectory_sha256"]:
        raise RuntimeError(f"trajectory hash mismatch for seed {seed} {condition} {schedule}")
    if packed["places"] != list(PLACE_IDS) or len(packed["agents"]) != 20 or len(packed["steps"]) != 501:
        raise RuntimeError(f"trajectory shape mismatch for seed {seed} {condition} {schedule}")
    if any(len(step) != 20 or any(digit not in "12345" for digit in step) for step in packed["steps"]):
        raise RuntimeError(f"packed step is not 20 legal places for seed {seed} {condition} {schedule}")
    return {
        "experiment": "experiment_020_social_field_source_switching",
        "seed": seed,
        "condition": condition,
        "schedule": schedule,
        "agents": list(packed["agents"]),
        "places": list(packed["places"]),
        "steps": list(packed["steps"]),
        "trajectory_sha256": digest,
        "source_trajectory": source_trajectory,
        "metrics": _metric_block(metrics),
    }


def extract_seed(seed: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    trajectory_path = DATA / "trajectories" / f"{seed}.json"
    metrics_path = DATA / "seeds" / f"{seed}.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_trajectory = f"data/worlds/world_v0_20/trajectories/{seed}.json"
    by_key = {(item["condition_id"], item["schedule"]): item for item in trajectory["worlds"]}
    metric_key = {(item["condition_id"], item["schedule"]): item for item in metrics["worlds"]}
    if set(by_key) != set(metric_key):
        raise RuntimeError(f"seed {seed} trajectory keys do not match metrics keys")
    worlds = []
    inventory = []
    for memory in MEMORIES:
        for transform in TRANSFORMS:
            for label in LABELS:
                condition = f"{memory}_{transform}__{label}"
                for schedule in SCHEDULES:
                    packed = by_key[(condition, schedule)]["series_packed"]
                    row = metric_key[(condition, schedule)]
                    record = _world_record(seed, condition, schedule, packed, row, source_trajectory)
                    worlds.append(record)
                    inventory.append({"seed": seed, "condition": condition, "schedule": schedule, "trajectory_sha256": record["trajectory_sha256"], "source_trajectory": source_trajectory})
    donor_metrics = dict(metrics["donor"])
    donor_metrics["trajectory_sha256"] = metrics["donor_trajectory_sha256"]
    donor = _world_record(seed, "S0", "S0", trajectory["donor_series_packed"], donor_metrics, source_trajectory)
    if donor["trajectory_sha256"] != trajectory["donor_trajectory_sha256"]:
        raise RuntimeError(f"donor hash mismatch for seed {seed}")
    worlds.append(donor)
    inventory.append({"seed": seed, "condition": "S0", "schedule": "S0", "trajectory_sha256": donor["trajectory_sha256"], "source_trajectory": source_trajectory})
    payload = {
        "seed": seed,
        "source_trajectory": source_trajectory,
        "source_trajectory_sha256": file_sha256(trajectory_path),
        "source_metrics": f"data/worlds/world_v0_20/seeds/{seed}.json",
        "source_metrics_sha256": file_sha256(metrics_path),
        "worlds": worlds,
    }
    return payload, inventory


def catalog() -> dict[str, Any]:
    return {
        "status": "V1_PREVIEW",
        "mode": "REPLAY_ONLY",
        "release": "NOT_RELEASED",
        "experiment_021": "NOT_STARTED",
        "banner": "既有實驗回放",
        "experiment": "experiment_020_social_field_source_switching",
        "seeds": list(SEEDS),
        "memories": [
            {"id": "CUM", "label": "CUM：累積全部歷史"},
            {"id": "RW50", "label": "RW50：最近 50 步"},
            {"id": "EXP99", "label": "EXP99：較舊歷史變輕"},
        ],
        "transforms": [
            {"id": "LOGMAX", "label": "LOGMAX：分數較緩和"},
            {"id": "ZRAW", "label": "ZRAW：分數起伏較大"},
        ],
        "q_labels": [
            {"id": "q05", "label": "q05"},
            {"id": "q15", "label": "q15"},
            {"id": "q25", "label": "q25"},
        ],
        "q_note": "這是參考情境的校準標籤，不保證本世界實際 Q 相同。",
        "schedules": [
            {"id": "EE", "label": "EE：全程本世界歷史"},
            {"id": "EO", "label": "EO：t=51 起改用 donor"},
            {"id": "OE", "label": "OE：t=51 起改回本世界"},
            {"id": "OO", "label": "OO：全程 donor"},
            {"id": "S0", "label": "S0：donor 參考回放"},
        ],
        "places": [
            {
                "id": place["place_id"],
                "label": f"場所 {index}",
                "capacity": place["capacity"],
                "visual_attractiveness": place["visual_attractiveness"],
                "novelty": place["novelty"],
                "layout": "schematic",
            }
            for index, place in enumerate(PLACES, start=1)
        ],
        "time_marks": TIME_MARKS,
        "default_selection": {"memory": "CUM", "transform": "LOGMAX", "q": "q25", "seed": 1700, "left": "EE", "right": "EO"},
        "phrases": {
            "not_applicable": "不適用／沒有合格樣本",
            "no_exit_by_horizon": "到 t=500 尚未跌破 18 人門檻",
            "no_release_by_horizon": "到 t=500 尚未解除集中",
            "event_not_formed": "未形成",
        },
        "llm_calls": 0,
        "worlds_resimulated": False,
    }


def manifest_paths() -> list[str]:
    names = [
        "v1_demo/README.md",
        "v1_demo/build_preview.py",
        "v1_demo/serve_preview.py",
        "v1_demo/display_rules.py",
        "v1_demo/__init__.py",
    ]
    for path in sorted((ROOT / "v1_demo" / "static").glob("*")):
        if path.is_file():
            names.append(path.relative_to(ROOT).as_posix())
    export = ROOT / "v1_demo" / "export"
    if export.exists():
        for path in sorted(export.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                names.append(path.relative_to(ROOT).as_posix())
    return names


def write_manifest() -> None:
    payload = {
        "record_type": "v1_preview_manifest",
        "includes_self": False,
        "python": platform.python_version(),
        "worlds_resimulated": False,
        "llm_calls": 0,
        "files": [{"path": path, "sha256": file_sha256(ROOT / path)} for path in manifest_paths()],
    }
    write_json(OUT / "manifest.json", payload)


def verify_manifest() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    listed = [item["path"] for item in manifest["files"]]
    if "v1_demo/export/manifest.json" in listed:
        raise RuntimeError("manifest includes itself")
    if listed != manifest_paths():
        raise RuntimeError("manifest file list drifted")
    for item in manifest["files"]:
        if file_sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {item['path']}")


def main() -> None:
    before = {path: file_sha256(ROOT / path) for path in SOURCE_NAMES}
    inventory: list[dict[str, str]] = []
    for seed in SEEDS:
        payload, rows = extract_seed(seed)
        write_compact(OUT / "replays" / f"{seed}.json", payload)
        inventory.extend(rows)
    keys = {(row["seed"], row["condition"], row["schedule"]) for row in inventory}
    if len(inventory) != 219 or len(keys) != 219:
        raise RuntimeError("the gate subset is not the expected 219 unique trajectories")
    write_json(OUT / "catalog.json", catalog())
    write_json(
        OUT / "build_record.json",
        {
            "python": platform.python_version(),
            "builder": "v1_demo/build_preview.py",
            "builder_sha256": file_sha256(Path(__file__).resolve()),
            "worlds_resimulated": False,
            "llm_calls": 0,
            "sources": [{"path": path, "sha256": before[path]} for path in SOURCE_NAMES],
            "inventory": inventory,
        },
    )
    after = {path: file_sha256(ROOT / path) for path in SOURCE_NAMES}
    if before != after:
        raise RuntimeError("a research source changed while building the preview")
    write_manifest()
    verify_manifest()


if __name__ == "__main__":
    main()
