"""Checks for the v0.16 synthesis tables. These tests do not run a world experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "research_synthesis_v016" / "build_synthesis.py"
MANIFEST = ROOT / "docs" / "research_synthesis_v016" / "source_manifest.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("research_synthesis_v016", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_synthesis_tables_keep_cohorts_separate_and_match_sources() -> None:
    builder = load_builder()
    tables = builder.extract_tables()
    assert {row["n"] for row in tables["cells"]} == {50, 100}
    assert len(tables["identity"]) == 2
    old = next(row for row in tables["cells"] if row["cohort"] == "v0.15" and row["architecture"] == "RW50" and row["lambda"] == "0.40" and row["horizon"] == "H2000")
    new = next(row for row in tables["cells"] if row["cohort"] == "v0.16" and row["architecture"] == "RW50" and row["lambda"] == "0.60" and row["horizon"] == "H2000")
    summary = json.loads((ROOT / "data/worlds/world_v0_16/experiment_016_summary.json").read_text(encoding="utf-8"))
    level = summary["regimes"]["RW50"]["levels"]["0.60"]["horizons"]["H2000"]
    assert old["mean_d"] == 0.0
    assert old["ci_degenerate_zero"] is True
    assert new["mean_d"] == level["D_horizon"]
    assert new["median"] == 0.0
    assert new["persistence_class"] == "persistent residual"
    assert new["n"] == 100
    assert tables["reference"]["observed_overlap"] == 12
    assert tables["reference"]["expected_overlap"] == 12.2
    assert tables["reference"]["hypergeometric_central_95"] == [8, 16]
    assert tables["replication"]["onset"]["overall"] == "FULL_ONSET_REPLICATION"
    builder.verify_claims()


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
spec = importlib.util.spec_from_file_location("synthesis_without_matplotlib", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tables = module.extract_tables()
if {{row["n"] for row in tables["cells"]}} != {{50, 100}}:
    raise SystemExit("cohort sample sizes changed")
module.verify_claims()
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
    assert "docs/RESEARCH_SYNTHESIS_V016.md" in output_paths
    assert "docs/research_synthesis_v016/plotting-requirements.txt" in output_paths
    assert "docs/research_synthesis_v016/REBUILD.md" in output_paths
    assert "docs/research_synthesis_v016/source_manifest.json" not in output_paths
    assert file_sha256(SCRIPT) == manifest["script_sha256"]
    for item in manifest["sources"] + manifest["outputs"]:
        assert file_sha256(ROOT / item["path"]) == item["sha256"]
