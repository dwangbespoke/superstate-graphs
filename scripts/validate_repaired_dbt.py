"""Package an explicitly attributed critic repair and execute its dbt oracle.

This does not call a generation model or count the repair as unassisted synthesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from superstate_graphs.dbt_constructor import construct_dbt_task
from superstate_graphs.dbt_runtime import validate_dbt_task

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=ROOT / "results/analysis_v1/shared_start_paths_reviewed.json")
    parser.add_argument("--db", type=Path, default=ROOT / "data/runtime/retail.duckdb")
    args = parser.parse_args()
    raw = args.proposal.read_bytes()
    proposal = json.loads(raw)
    paths = json.loads(args.paths.read_text())["paths"]
    path = next(item for item in paths if item["path_id"] == args.path_id)
    result = construct_dbt_task(
        path, lambda messages: proposal, args.db, args.output, max_repairs=0,
        runtime_validator=lambda directory: validate_dbt_task(directory, timeout_seconds=600),
    )
    receipt = {
        "generation_assistance": "Codex critic repair of a Qwen proposal",
        "interpretation": "Assisted executable example, not an unassisted task-generation success.",
        "proposal_sha256": hashlib.sha256(raw).hexdigest(),
        "proposal_source": str(args.proposal), "repair_notes": json.loads(args.notes.read_text()),
        "result_status": result["status"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "critic_repair.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"]}), flush=True)
    if result["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
