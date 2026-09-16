"""Compile up to a few actual graph paths into mutable dbt tasks; never run on import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from superstate_graphs import analyze
from superstate_graphs.dbt_constructor import construct_dbt_task
from superstate_graphs.dbt_runtime import validate_dbt_task

ROOT = Path(__file__).resolve().parents[1]


def original_judgments(path: dict, analysis_dir: Path) -> list[dict]:
    """Reuse exact saved directional judgments; never substitute a reversed splice."""
    junction = path["junction_evidence"]
    left_id, right_id = junction["left_history_id"], junction["right_history_id"]
    records = []
    for probe_id in junction.get("exact_probe_ids", []):
        filename = analysis_dir / "frozen_judgments" / f"{probe_id}.json"
        if not filename.is_file():
            continue
        record = json.loads(filename.read_text())
        if record.get("left_id") == left_id and record.get("right_id") == right_id:
            judgment = record["judgment"]
            if all(key in judgment for key in ("decision", "local_transfer", "full_segment_transfer")):
                records.append({"probe_id": probe_id, "source": str(filename),
                                "judge_prompt_sha256": record.get("judge_prompt_sha256"),
                                "judgment": judgment})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "results/analysis_v1")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "data/runtime/retail.duckdb")
    parser.add_argument("--output", type=Path, default=ROOT / "results/generated_dbt_tasks")
    parser.add_argument("--endpoint", type=Path, default=ROOT / "results/runtime/reflector.json")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-paths", type=int, default=6)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--runtime-timeout", type=int, default=600)
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--judge-uncached", action="store_true",
                        help="Allow two bounded judge calls when no exact saved junction judgment exists")
    args = parser.parse_args()
    if not 1 <= args.count <= 3 or not args.count <= args.max_paths <= 12:
        parser.error("Use 1–3 tasks and count <= max-paths <= 12 for this POC")
    paths = json.loads((args.paths or args.analysis / "selected_paths.json").read_text())["paths"]
    client = analyze.CachedClient(args.endpoint, args.output / "cache", "mutable_dbt_constructor")
    outcomes, seen = [], set()
    for path in paths:
        if len(outcomes) >= args.max_paths:
            break
        identity = (path["target_superstate"], tuple(path.get("source_task_ids", [])))
        if identity in seen:
            continue
        seen.add(identity)
        path = json.loads(json.dumps(path))
        try:
            originals = path.get("original_replay_judgments") or original_judgments(path, args.analysis)
            if not originals and args.judge_uncached:
                transition = path["transitions"][1]
                witness = Path(transition["witness"])
                if not witness.is_absolute():
                    witness = ROOT / witness
                segment = json.loads(witness.read_text())
                if segment["source_id"] != transition["source_history_id"]:
                    raise ValueError("Witness does not identify the selected source history")
                judgment, attempts = analyze.judge_pair(
                    client, path["junction_prefix_A"], path["junction_prefix_B"], segment,
                    args.output / "junction_attempts" / path["path_id"],
                )
                record = {"path_id": path["path_id"], "judgment": judgment,
                          "structured_attempts": attempts, "source": transition["witness"]}
                analyze.save(args.output / "junction_judgments" / f"{path['path_id']}.json", record)
                originals = [record]
            if not originals:
                result = {"status": "needs_original_junction_judgment",
                          "reason": "No exact cached directional judgment; --judge-uncached permits bounded assessment"}
            elif any(item.get("judgment", item)["decision"]["label"] == "contradicted" for item in originals):
                result = {"status": "unsupported_target", "reason": "Original local decision evidence is contradicted"}
            else:
                path["original_replay_judgments"] = originals
                result = construct_dbt_task(
                    path, lambda messages: client.call(messages, thinking=True, max_tokens=16384,
                        temperature=0.3, response_format={"type": "json_object"}),
                    args.db, args.output / path["path_id"], max_repairs=args.max_repairs,
                    runtime_validator=None if args.package_only else
                        lambda task: validate_dbt_task(task, timeout_seconds=args.runtime_timeout),
                )
            outcomes.append({"path_id": path["path_id"], **result})
        except Exception as error:
            outcomes.append({"path_id": path["path_id"], "status": "error",
                             "error": f"{type(error).__name__}: {error}"})
        analyze.save(args.output / "summary.json", {"attempted": outcomes, "client": client.stats(),
                     "construction_scope": "New task compilation; original literal replay verdicts unchanged"})
        print(json.dumps({"event": "mutable_dbt_task_attempted", "path_id": path["path_id"],
                          "status": outcomes[-1]["status"]}), flush=True)
        completed = {"awaiting_runtime_validation", "validated"} if args.package_only else {"validated"}
        if sum(item["status"] in completed for item in outcomes) >= args.count:
            break


if __name__ == "__main__":
    main()
