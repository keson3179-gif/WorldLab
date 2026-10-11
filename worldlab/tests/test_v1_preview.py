"""Contract tests for the V1 replay bundle. They do not run a world experiment."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from urllib.request import Request, urlopen

from world.experiment_004 import trajectory_hash
from world.experiment_017 import file_sha256, unpack_series
from world.geography import PLACE_IDS

from v1_demo.build_preview import SOURCE_NAMES, main, verify_manifest
from v1_demo.display_rules import NOT_APPLICABLE
from v1_demo.serve_preview import make_server

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "v1_demo" / "export"


def test_preview_bundle_matches_saved_trajectories() -> None:
    before = {path: file_sha256(ROOT / path) for path in SOURCE_NAMES}
    main()
    after = {path: file_sha256(ROOT / path) for path in SOURCE_NAMES}
    assert before == after
    verify_manifest()
    record = json.loads((EXPORT / "build_record.json").read_text(encoding="utf-8"))
    assert record["worlds_resimulated"] is False
    assert record["llm_calls"] == 0
    assert len(record["inventory"]) == 219
    keys = {(item["seed"], item["condition"], item["schedule"]) for item in record["inventory"]}
    assert len(keys) == 219
    assert {item["path"] for item in record["sources"]} == set(SOURCE_NAMES)
    for item in record["sources"]:
        assert file_sha256(ROOT / item["path"]) == item["sha256"]
    seen = set()
    ineligible = 0
    occurred = 0
    for seed in (1700, 2199, 2699):
        payload = json.loads((EXPORT / "replays" / f"{seed}.json").read_text(encoding="utf-8"))
        assert len(payload["worlds"]) == 73
        for world in payload["worlds"]:
            identity = (world["seed"], world["condition"], world["schedule"])
            assert identity not in seen
            seen.add(identity)
            assert world["places"] == list(PLACE_IDS)
            assert len(world["agents"]) == 20
            assert len(world["steps"]) == 501
            assert trajectory_hash(unpack_series({"agents": world["agents"], "places": world["places"], "steps": world["steps"]})) == world["trajectory_sha256"]
            maintenance = world["metrics"]["maintenance"]
            if not maintenance["eligible"]:
                ineligible += 1
                assert maintenance["original_exit"] is None
                assert maintenance["display"]["original_exit"] == NOT_APPLICABLE
                assert "0%" not in maintenance["display"]["original_exit"]
            if world["metrics"]["event18"]["occurred"]:
                occurred += 1
                assert world["metrics"]["event18"]["confirm"] == world["metrics"]["event18"]["start"] + 19
            assert world["metrics"]["late_hhi"]["display_zh"] == "第 101–500 步的累積訪問集中度"
            assert world["metrics"]["late_hhi"]["window"] == "committed t=101-500 cumulative presence"
    assert seen == keys
    assert ineligible > 0
    assert occurred > 0
    catalog = json.loads((EXPORT / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["time_marks"]["50"].startswith("t=50")
    assert "t=51" in catalog["time_marks"]["51"]
    assert catalog["status"] == "V1_PREVIEW"
    assert catalog["release"] == "NOT_RELEASED"
    page = (ROOT / "v1_demo" / "static" / "app.js").read_text(encoding="utf-8")
    assert "state.t" in page
    assert "phrases[name]" in page or "phrase(" in page
    assert "0%" not in page


def test_server_reads_without_writing_sources() -> None:
    before = {path: file_sha256(ROOT / path) for path in SOURCE_NAMES}
    manifest_before = file_sha256(EXPORT / "manifest.json")
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/export/catalog.json") as response:
            catalog = json.loads(response.read().decode("utf-8"))
        assert catalog["mode"] == "REPLAY_ONLY"
        with urlopen(f"http://127.0.0.1:{port}/static/index.html") as response:
            page = response.read().decode("utf-8")
        assert "既有實驗回放" in page
        request = Request(f"http://127.0.0.1:{port}/export/catalog.json", method="POST", data=b"{}")
        try:
            urlopen(request)
        except Exception as exc:
            assert getattr(exc, "code", None) == 405
        else:
            raise AssertionError("POST was accepted")
    finally:
        server.shutdown()
        server.server_close()
    assert {path: file_sha256(ROOT / path) for path in SOURCE_NAMES} == before
    assert file_sha256(EXPORT / "manifest.json") == manifest_before


def test_preview_ui_behavior() -> None:
    completed = subprocess.run(
        ["node", str(ROOT / "tests" / "v1_preview_ui_check.mjs")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
