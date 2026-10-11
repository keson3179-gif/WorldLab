from __future__ import annotations

import argparse
import json

from worldlab_data.config import load_paths
from worldlab_data.pipeline import run_clean, run_inventory, run_match, run_panel
from worldlab_data.qa_runner import evaluate_gates, write_gate_report
from worldlab_data.reports_v4 import write_final_reports
from worldlab_data.tdx import dry_run_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WorldLab V4 觀察式資料清洗與品質驗收")
    parser.add_argument("command", choices=("inventory", "clean", "match-sites", "build-panel", "qa", "tdx-fetch"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default="")
    args = parser.parse_args(argv)
    paths = load_paths(None if not args.config else __import__("pathlib").Path(args.config))
    if args.command == "inventory":
        print(json.dumps(run_inventory(paths, dry_run=args.dry_run), ensure_ascii=False, indent=2))
        return 0
    if args.command == "clean":
        print(json.dumps(run_clean(paths, dry_run=args.dry_run), ensure_ascii=False, indent=2))
        return 0
    if args.command == "match-sites":
        print(json.dumps(run_match(paths, dry_run=args.dry_run), ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-panel":
        print(json.dumps(run_panel(paths, dry_run=args.dry_run), ensure_ascii=False, indent=2))
        return 0
    if args.command == "qa":
        gates = evaluate_gates(paths)
        fails = write_gate_report(gates, paths.reports_root)
        write_final_reports(paths, gates)
        print(json.dumps({"fail_gates": fails}, ensure_ascii=False))
        return 1 if fails else 0
    if args.command == "tdx-fetch":
        plan = dry_run_plan()
        if args.dry_run or not plan["credentials_present"]:
            plan["will_call_network"] = False
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0 if args.dry_run or not plan["credentials_present"] else 0
        print(json.dumps({"will_call_network": False, "reason": "本輪只留離線 adapter，不在清洗階段呼叫 TDX"}, ensure_ascii=False, indent=2))
        return 0
    return 2
