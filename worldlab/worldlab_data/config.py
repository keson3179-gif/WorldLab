from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


WORLD_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class V4Paths:
    raw_root: Path
    output_root: Path
    reports_root: Path
    manual_crosswalk: Path

    @property
    def interim(self) -> Path:
        return self.output_root / "interim"

    @property
    def processed(self) -> Path:
        return self.output_root / "processed"

    @property
    def curated(self) -> Path:
        return self.output_root / "curated"

    @property
    def raw_pointer(self) -> Path:
        return self.output_root / "raw"

    def ensure_output_dirs(self) -> None:
        for path in (self.interim, self.processed, self.curated, self.raw_pointer, self.reports_root):
            path.mkdir(parents=True, exist_ok=True)


def load_paths(config_path: Path | None = None) -> V4Paths:
    path = config_path or (WORLD_ROOT / "configs" / "v4_data.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    def resolve(value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (WORLD_ROOT / candidate).resolve()

    return V4Paths(
        raw_root=resolve(payload["raw_root"]),
        output_root=resolve(payload["output_root"]),
        reports_root=resolve(payload["reports_root"]),
        manual_crosswalk=resolve(payload["manual_crosswalk"]),
    )
