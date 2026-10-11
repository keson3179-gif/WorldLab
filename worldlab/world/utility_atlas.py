"""Utility Atlas v1 for the reachable World v0.1 input grid.

Place ids and origin ids stay in metadata. The prompt is the independent
utility v3 text, deduplicated by its content hash. A frozen atlas is read-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT, Settings, get_settings
from experiments.experiment_001 import load_base_traits
from experiments.experiment_001e import PROMPT_VERSION, PlaceUtility, build_utility_messages, prompt_sha256
from experiments.reproducibility_check import canonical_sha256
from llm.client import LLMCallError, LLMClient, StructuredLLMClient
from world.geography import PLACE_IDS, crowding, distance_matrix, place_index, travel_distance
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR

ATLAS_ID = "utility_atlas_v1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / ATLAS_ID
POPULATION_LIMIT = 20
COMPONENT_NOTE = "Component scores are diagnostic outputs, not independent construct measures."


def metric_key(value: float) -> str:
    return f"{value:.10f}"


def crowding_level_id(occupancy: int, capacity: int) -> int:
    if occupancy < 0:
        raise ValueError("occupancy cannot be negative")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return min(occupancy, capacity, POPULATION_LIMIT)


def crowding_level_ids(capacity: int) -> list[int]:
    return list(range(crowding_level_id(POPULATION_LIMIT, capacity) + 1))


def distance_levels(matrix: dict[str, dict[str, float]], place_id: str) -> list[float]:
    found: list[float] = []
    for origin in PLACE_IDS:
        value = travel_distance(matrix, origin, place_id)
        if not any(abs(value - kept) < 1e-12 for kept in found):
            found.append(value)
    return sorted(found)


def iter_slots(bases: list[dict[str, Any]], matrix: dict[str, dict[str, float]] | None = None) -> list[dict[str, Any]]:
    active_matrix = distance_matrix() if matrix is None else matrix
    places = place_index()
    traits = {base["base_agent_id"]: base["traits"] for base in bases}
    slots = []
    for agent_id in sorted(traits):
        for place_id in PLACE_IDS:
            place = places[place_id]
            capacity = int(place["capacity"])
            for distance in distance_levels(active_matrix, place_id):
                for level in crowding_level_ids(capacity):
                    attributes = {
                        "crowding": crowding(level, capacity),
                        "visual_attractiveness": float(place["visual_attractiveness"]),
                        "novelty": float(place["novelty"]),
                        "distance": distance,
                    }
                    messages = build_utility_messages(traits[agent_id], attributes, prompt_version=PROMPT_VERSION)
                    slots.append(
                        {
                            "agent_id": agent_id,
                            "place_id": place_id,
                            "distance": distance,
                            "distance_key": metric_key(distance),
                            "crowding_level_id": level,
                            "crowding": attributes["crowding"],
                            "visual_attractiveness": attributes["visual_attractiveness"],
                            "novelty": attributes["novelty"],
                            "prompt_sha256": prompt_sha256(messages),
                        }
                    )
    return slots


def build_atlas(
    settings: Settings | None = None,
    client: LLMClient | None = None,
    *,
    persona_dir: Path | None = None,
    output_dir: Path | None = None,
    cache_path: Path | None = None,
    slots: list[dict[str, Any]] | None = None,
) -> Path:
    active = get_settings() if settings is None else settings
    _require_atlas_settings(active)
    destination = output_dir or OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "atlas_manifest.json"
    existing = _read_manifest(manifest_path)
    if existing is not None and existing.get("frozen") is True:
        _verify_frozen(destination, existing)
        return destination
    source = persona_dir or (active.personas_dir / "persona_set_v1")
    bases = load_base_traits(source, active.world_agent_count)
    planned = slots if slots is not None else iter_slots(bases)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for slot in planned:
        grouped.setdefault(str(slot["prompt_sha256"]), []).append(slot)
    world_cache = _load_score_cache(cache_path or (WORLD_V01_DIR / "utility_cache.jsonl"))
    partial_path = destination / "atlas_partial.jsonl"
    partial = _load_score_cache(partial_path)
    llm = client if client is not None else StructuredLLMClient(active)
    traits = {base["base_agent_id"]: base["traits"] for base in bases}
    resolved: dict[str, dict[str, Any]] = {}
    new_calls = 0
    for index, (digest, group) in enumerate(sorted(grouped.items()), start=1):
        slot = group[0]
        if digest in partial:
            resolved[digest] = {"scores": partial[digest], "source": "llm_resume"}
            continue
        cached = world_cache.get(digest)
        if cached is not None:
            resolved[digest] = {"scores": cached, "source": "cache"}
            continue
        attributes = {
            "crowding": float(slot["crowding"]),
            "visual_attractiveness": float(slot["visual_attractiveness"]),
            "novelty": float(slot["novelty"]),
            "distance": float(slot["distance"]),
        }
        messages = build_utility_messages(traits[slot["agent_id"]], attributes, prompt_version=PROMPT_VERSION)
        if prompt_sha256(messages) != digest:
            raise RuntimeError("atlas prompt hash changed between enumeration and the call")
        parsed = _complete(active, llm, messages)
        scores = _scores(parsed)
        partial[digest] = scores
        append_jsonl(partial_path, {"prompt_sha256": digest, "scores": scores, "source": "llm"})
        resolved[digest] = {"scores": scores, "source": "llm"}
        new_calls += 1
        print(f"atlas {index}/{len(grouped)} new_calls={new_calls}", flush=True)
    records = []
    for digest, group in sorted(grouped.items()):
        item = resolved[digest]
        slot = group[0]
        source = "cache" if item["source"] == "cache" else "llm"
        records.append(
            {
                "prompt_sha256": digest,
                "agent_id": slot["agent_id"],
                "place_id": slot["place_id"],
                "distance": slot["distance"],
                "distance_key": slot["distance_key"],
                "crowding_level_id": slot["crowding_level_id"],
                "crowding": slot["crowding"],
                "visual_attractiveness": slot["visual_attractiveness"],
                "novelty": slot["novelty"],
                "lookup_keys": [
                    {
                        "agent_id": alias["agent_id"],
                        "place_id": alias["place_id"],
                        "distance_key": alias["distance_key"],
                        "crowding_level_id": alias["crowding_level_id"],
                    }
                    for alias in group
                ],
                "overall_utility": item["scores"]["overall_utility"],
                "crowding_fit": item["scores"]["crowding_fit"],
                "visual_fit": item["scores"]["visual_fit"],
                "distance_fit": item["scores"]["distance_fit"],
                "novelty_fit": item["scores"]["novelty_fit"],
                "source": source,
                "component_scores_are_diagnostics_only": True,
            }
        )
    atlas_path = destination / "utility_atlas_v1.jsonl"
    _write_jsonl(atlas_path, records)
    manifest = {
        "atlas_id": ATLAS_ID,
        "frozen": True,
        "prompt_version": PROMPT_VERSION,
        "population_limit": POPULATION_LIMIT,
        "theoretical_slot_count": len(planned),
        "unique_prompt_count": len(grouped),
        "reused_cache_entries": sum(1 for record in records if record["source"] == "cache"),
        "new_llm_calls": sum(1 for record in records if record["source"] == "llm"),
        "sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
        "content_sha256": canonical_sha256(records),
        "component_score_note": COMPONENT_NOTE,
        "crowding_formula": "min(1.0, occupancy / capacity)",
        "place_ids_not_sent": True,
        "origin_ids_not_sent": True,
        "calls_this_process": new_calls,
        "model_configuration": {
            "model": active.model,
            "temperature": active.temperature,
            "seed": active.llm_seed,
            "reasoning_effort": active.reasoning_effort,
            "decision_prompt_version": PROMPT_VERSION,
        },
    }
    write_json(manifest_path, manifest)
    return destination


class FrozenUtilityAtlas:
    """Lookup of overall utility. This object does not call a model."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.index: dict[tuple[str, str, str, int], int] = {}
        for record in records:
            utility = int(record["overall_utility"])
            keys = record.get("lookup_keys") or [
                {
                    "agent_id": record["agent_id"],
                    "place_id": record["place_id"],
                    "distance_key": record["distance_key"],
                    "crowding_level_id": record["crowding_level_id"],
                }
            ]
            for item in keys:
                key = (
                    str(item["agent_id"]),
                    str(item["place_id"]),
                    str(item["distance_key"]),
                    int(item["crowding_level_id"]),
                )
                previous = self.index.get(key)
                if previous is not None and previous != utility:
                    raise RuntimeError("atlas has two utilities for one input key")
                self.index[key] = utility

    def lookup(self, agent_id: str, place_id: str, distance: float, occupancy: int, capacity: int) -> int:
        key = (agent_id, place_id, metric_key(distance), crowding_level_id(occupancy, capacity))
        try:
            return self.index[key]
        except KeyError as exc:
            raise KeyError("utility atlas does not contain this input") from exc


def load_frozen_atlas(path: Path | None = None) -> FrozenUtilityAtlas:
    directory = path or OUTPUT_DIR
    manifest_path = directory / "atlas_manifest.json"
    manifest = _read_manifest(manifest_path)
    if manifest is None or manifest.get("frozen") is not True:
        raise RuntimeError("utility atlas is not frozen")
    _verify_frozen(directory, manifest)
    return FrozenUtilityAtlas(_read_jsonl(directory / "utility_atlas_v1.jsonl"))


def _verify_frozen(directory: Path, manifest: dict[str, Any]) -> None:
    atlas_path = directory / "utility_atlas_v1.jsonl"
    digest = hashlib.sha256(atlas_path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise RuntimeError("frozen utility atlas hash does not match the file")


def _complete(settings: Settings, client: LLMClient, messages: list[dict[str, str]]) -> PlaceUtility:
    error = ""
    for _attempt in range(settings.decision_max_attempts):
        try:
            result = client.complete(messages=messages, response_model=PlaceUtility)
        except (LLMCallError, ValidationError) as exc:
            error = str(exc)
            continue
        if isinstance(result.parsed, PlaceUtility):
            return result.parsed
        error = "parsed object was not a PlaceUtility"
    raise RuntimeError(error or "atlas utility call failed")


def _scores(parsed: PlaceUtility) -> dict[str, int]:
    return {
        "overall_utility": int(parsed.overall_utility),
        "crowding_fit": int(parsed.crowding_fit),
        "visual_fit": int(parsed.visual_fit),
        "distance_fit": int(parsed.distance_fit),
        "novelty_fit": int(parsed.novelty_fit),
    }


def _require_atlas_settings(settings: Settings) -> None:
    if settings.temperature != 0.0:
        raise RuntimeError("utility atlas keeps the LLM temperature at 0")
    if settings.reasoning_effort != "none":
        raise RuntimeError("utility atlas requires reasoning_effort none")
    if settings.llm_seed != 42:
        raise RuntimeError("utility atlas requires llm seed 42")


def _load_score_cache(path: Path) -> dict[str, dict[str, int]]:
    cache: dict[str, dict[str, int]] = {}
    for row in _read_jsonl(path):
        scores = row.get("scores")
        if scores is None:
            scores = {name: row[name] for name in ("overall_utility", "crowding_fit", "visual_fit", "distance_fit", "novelty_fit")}
        parsed = {key: int(value) for key, value in scores.items()}
        digest = str(row["prompt_sha256"])
        previous = cache.get(digest)
        if previous is not None and previous != parsed:
            raise RuntimeError("utility cache has conflicting scores for one prompt")
        cache[digest] = parsed
    return cache


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def main() -> None:
    build_atlas()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.utility_atlas")
    main()
