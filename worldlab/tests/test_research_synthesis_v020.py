"""Checks for the v0.20 synthesis. These tests do not run a world experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "research_synthesis_v020" / "build_synthesis.py"
MANIFEST = ROOT / "docs" / "research_synthesis_v020" / "source_manifest.json"
SYNTHESIS = ROOT / "docs" / "RESEARCH_SYNTHESIS_V020.md"

FORBIDDEN = (
    "每一個合格 seed 都在 t=51",
    "全部在 t=51 解除",
    "區間全部包含 0",
    "全部涵蓋 0",
    "manifest 全部吻合",
    "已證明沒有差異",
    "已證明無差異",
    "phase transition",
    "critical point",
    "永久吸收",
    "human realism",
    "t=401-500",
    "t＝401–500",
)


def load_builder():
    spec = importlib.util.spec_from_file_location("research_synthesis_v020", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_claims_resolve_and_keep_cohorts_windows_and_na() -> None:
    builder = load_builder()
    before = {path: file_sha256(ROOT / path) for path in builder.direct_sources()}
    tables = builder.extract_tables()
    claims = builder.verify_claims(tables)
    after = {path: file_sha256(ROOT / path) for path in builder.direct_sources()}
    assert before == after
    assert {row["n"] for row in tables["h2000"]} == {100}
    assert {row["n"] for row in tables["matched_q"]} == {1000}
    assert {row["n"] for row in tables["late_q_019"]} == {1000}
    assert {row["n"] for row in tables["paths_020"]} == {1000}
    assert {row["seeds"] for row in tables["h2000"]} == {"100-199"}
    assert {row["seeds"] for row in tables["matched_q"]} == {"700-1699"}
    assert {row["seeds"] for row in tables["late_q_019"]} == {"700-1699"}
    assert {row["seeds"] for row in tables["paths_020"]} == {"1700-2699"}
    assert {row["hhi_window"] for row in tables["paths_020"]} == {"committed t=101-500 cumulative presence"}
    assert {row["window"] for row in tables["late_q_019"]} == {"decision t=101-500"}
    assert {row["q_window"] for row in tables["matched_q"]} == {"full horizon, S0 reference history"}
    empty = [row for row in tables["maintenance_020"] if row["eligible_n"] == 0]
    assert empty
    assert all(row["maintained_rate"] is None for row in empty)
    assert all(row["plot_as_rate"] is False for row in empty)
    assert any(row["schedule"] == "OE" and row["eligible_n"] == 0 for row in empty)
    ids = {claim["claim_id"] for claim in claims}
    assert ids == {"M01", "M02", "H01", "H02", "C01", "C02", "C03", "I01", "I02", "I03"}
    assert all(claim["verified_this_round"] is True for claim in claims)
    text = SYNTHESIS.read_text(encoding="utf-8")
    for phrase in ("51–52", "used–own", "2019", "1.3206593749999967e-05", "(0.40, 0.60]", "0.025", "NA", "1700", "不是 Experiment 021"):
        assert phrase in text
    for phrase in FORBIDDEN:
        assert phrase not in text


def test_extract_succeeds_when_matplotlib_import_is_blocked() -> None:
    child = f"""
import builtins
import importlib.util
from pathlib import Path
real_import = builtins.__import__

def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("No module named " + name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded
path = Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location("synthesis_v020_without_matplotlib", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tables = module.extract_tables()
if {{row["n"] for row in tables["paths_020"]}} != {{1000}}:
    raise SystemExit("020 sample size changed")
module.verify_claims(tables)
try:
    module.configure_font()
except RuntimeError as exc:
    message = str(exc)
    if "pip install -r" not in message or "plotting-requirements.txt" not in message:
        raise SystemExit(message)
else:
    raise SystemExit("plotting continued without matplotlib")
"""
    completed = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_source_manifest_hashes_match_current_files() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    output_paths = [item["path"] for item in manifest["outputs"]]
    assert "docs/RESEARCH_SYNTHESIS_V020.md" in output_paths
    assert "docs/research_synthesis_v020/plotting-requirements.txt" in output_paths
    assert "docs/research_synthesis_v020/REBUILD.md" in output_paths
    assert "docs/research_synthesis_v020/source_manifest.json" not in output_paths
    assert manifest["worlds_resimulated"] is False
    assert manifest["bootstrap_recomputed"] is False
    assert manifest["llm_calls"] == 0
    assert manifest["historical_trajectories_reverified"] is False
    assert manifest["font"] == "Microsoft JhengHei"
    assert all(item["verified_this_round"] is True for item in manifest["direct_sources"])
    assert all(item["verified_this_round"] is False for item in manifest["indirect_sources"])
    assert file_sha256(SCRIPT) == next(item["sha256"] for item in manifest["outputs"] if item["path"].endswith("build_synthesis.py"))
    for item in manifest["direct_sources"] + manifest["outputs"]:
        assert file_sha256(ROOT / item["path"]) == item["sha256"]
