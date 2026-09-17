#!/usr/bin/env python3
"""Verify the complete history-assignment census offline and emit a safe receipt.

Checks every prefix hash, transition boundary, completion checkpoint identity,
and previously classified row. This verifies data accounting, not semantic
cluster quality or graph-edge applicability. No model calls or serving keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from superstate_graphs.full_graph import _completion_provenance, _validate_assignment_index
from superstate_graphs.graph_evolution import GraphRuntime, validate_spec


FULL_COUNTS = {"histories": 37532, "transitions": 36502, "rollouts": 1030, "tasks": 103}
CORPUS_FILES = ("rollouts.jsonl", "histories.jsonl", "transitions.jsonl", "splits.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_assignments(run: Path, corpus: Path, *, expected_counts: dict | None = None) -> dict:
    expected_counts = FULL_COUNTS if expected_counts is None else expected_counts
    source_hashes = {}

    def read(name, *, corpus_source=False, jsonl=False):
        raw = ((corpus if corpus_source else run) / name).read_bytes()
        source_hashes[("corpus/" if corpus_source else "run/") + name] = sha(raw)
        return [json.loads(line) for line in raw.splitlines()] if jsonl else json.loads(raw)

    manifest = read("manifest.json", corpus_source=True)
    content = {
        name: read(name, corpus_source=True, jsonl=name.endswith("jsonl")) for name in CORPUS_FILES
    }
    for name in CORPUS_FILES:
        require(
            source_hashes["corpus/" + name] == manifest["artifact_sha256"][name],
            "Immutable corpus artifact hash mismatch: " + name,
        )
    rollouts = content["rollouts.jsonl"]
    histories = {row["history_id"]: row for row in content["histories.jsonl"]}
    require(len(histories) == len(content["histories.jsonl"]), "Duplicate corpus history IDs")
    assignments = read("assignments_all.json")
    initial = read("optimized_assignments_all.json")
    candidate = read("optimized_candidate.json")
    initial_nulls = {hid for hid, row in initial.items() if row["state_id"] is None}
    checkpoint_exists = (run / "completion_checkpoint.json").exists()
    require(
        checkpoint_exists or not initial_nulls,
        "Completion checkpoint missing for fallback assignments",
    )
    if checkpoint_exists:
        checkpoint = read("completion_checkpoint.json")
        graph = checkpoint["graph"]
        require(
            assignments == checkpoint["assignments"],
            "Final assignments differ from completion checkpoint",
        )
        contract = read("run_contract.json")
        models = []
        for prefix in ("", "teacher_"):
            model = {key: contract.get(prefix + key) for key in ("model", "revision")}
            require(
                all(isinstance(v, str) and v for v in model.values()),
                "Missing public pinned model identity",
            )
            models.append(SimpleNamespace(runtime=model))
        runtime = GraphRuntime(models[0], run, teacher_llm=models[1])
        require(
            checkpoint["provenance"]
            == _completion_provenance(runtime, candidate, rollouts, initial),
            "Completion provenance mismatch",
        )
    else:
        graph = {"nodes": validate_spec(candidate)["states"], "routing_stages": [candidate]}
    _validate_assignment_index(initial, rollouts, validate_spec(candidate)["states"])
    _validate_assignment_index(assignments, rollouts, graph["nodes"])
    require(
        set(histories) == set(assignments) == set(initial),
        "Final/initial/corpus history census mismatch",
    )
    counts = {
        "histories": len(histories),
        "transitions": len(content["transitions.jsonl"]),
        "rollouts": len(rollouts),
        "tasks": len({r["task_id"] for r in rollouts}),
    }
    require(
        counts == expected_counts, "Unexpected corpus size for complete assignment verification"
    )
    require(
        all(manifest["statistics"][key] == value for key, value in counts.items()),
        "Manifest corpus statistics mismatch",
    )
    require(len({r["id"] for r in rollouts}) == len(rollouts), "Duplicate corpus rollout IDs")
    splits = content["splits.json"]
    declared = {}
    task_splits = {}
    for split in ("train", "pareto", "test"):
        for rid in splits[split]["rollout_ids"]:
            require(rid not in declared, "Overlapping or duplicate rollout split IDs")
            declared[rid] = split
        for task in splits[split]["task_ids"]:
            require(task not in task_splits, "Overlapping or duplicate task split IDs")
            task_splits[task] = split
    require(set(declared) == {r["id"] for r in rollouts}, "Rollout split census mismatch")
    require(set(task_splits) == {r["task_id"] for r in rollouts}, "Task split census mismatch")
    require(
        graph["routing_stages"][0] == candidate,
        "Completion stages do not begin with selected specification",
    )
    stage_graphs = [validate_spec(stage) for stage in graph["routing_stages"]]
    require(
        graph["nodes"] == [node for stage in stage_graphs for node in stage["states"]],
        "Completion definitions differ from routing-stage chain",
    )
    stage_ids = [{node["id"] for node in stage["states"]} for stage in stage_graphs]
    require(
        len(set().union(*stage_ids)) == sum(len(ids) for ids in stage_ids),
        "Routing stages reuse a state ID",
    )
    nulls, checked_prefixes, newly_assigned = 0, 0, 0
    for rollout in rollouts:
        require(
            rollout["split"] == declared[rollout["id"]] == task_splits[rollout["task_id"]],
            "Rollout/task split mismatch",
        )
        offsets = rollout["history_end_offsets"]
        require(
            len(offsets) == rollout["history_count"] == len(rollout["steps"]) + 1,
            "History boundary count mismatch",
        )
        require(
            offsets == sorted(set(offsets))
            and all(type(end) is int and 0 < end <= len(rollout["transcript"]) for end in offsets),
            "Invalid prefix offsets",
        )
        require(
            sha(rollout["transcript"].encode()) == rollout["transcript_sha256"],
            "Rollout transcript hash mismatch",
        )
        for step, end in enumerate(offsets):
            hid = f"{rollout['id']}:h{step:04d}"
            history, assignment = histories[hid], assignments[hid]
            require(
                all(
                    assignment[field] == history[field]
                    for field in ("history_id", "rollout_id", "task_id", "split", "step")
                ),
                "Assignment/corpus identity mismatch",
            )
            require(history["prefix_char_end"] == end, "History prefix boundary mismatch")
            require(
                sha(rollout["transcript"][:end].encode()) == history["prefix_sha256"],
                "History prefix hash mismatch",
            )
            checked_prefixes += 1
            if hid not in initial_nulls:
                require(
                    assignment == initial[hid], "Completion modified an already classified history"
                )
            elif assignment["state_id"] is not None:
                stage = assignment.get("routing_stage")
                require(
                    type(stage) is int
                    and 1 <= stage < len(stage_ids)
                    and assignment["state_id"] in stage_ids[stage],
                    "Fallback assignment references wrong routing stage",
                )
                newly_assigned += 1
            nulls += assignment["state_id"] is None
    require(nulls == 0, "Final assignments still contain unclassified histories")
    transitions, pairs = set(), set()
    for row in content["transitions.jsonl"]:
        tid = row["transition_id"]
        require(tid not in transitions, "Duplicate transition ID")
        transitions.add(tid)
        source, target = histories[row["source_history_id"]], histories[row["target_history_id"]]
        require(
            source["step"] == row["step"] and target["step"] == row["step"] + 1,
            "Transition crosses a nonconsecutive history boundary",
        )
        require(
            tid == f"{row['rollout_id']}:t{row['step']:04d}",
            "Transition identifier does not match step",
        )
        require(
            all(
                source[field] == target[field] == row[field]
                for field in ("rollout_id", "task_id", "split")
            ),
            "Transition crosses rollout/task/split boundary",
        )
        pairs.add(
            (
                assignments[source["history_id"]]["state_id"],
                assignments[target["history_id"]]["state_id"],
            )
        )
    for key, fingerprint in source_hashes.items():
        root, name = key.split("/", 1)
        require(
            sha(((corpus if root == "corpus" else run) / name).read_bytes()) == fingerprint,
            "Source artifact changed during verification",
        )
    return {
        "format": "independent-full-assignment-verification-v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "zero_null_completion_verified",
        "counts": counts,
        "unassigned": nulls,
        "prefix_hashes_verified": checked_prefixes,
        "initial_unassigned": len(initial_nulls),
        "newly_classified": newly_assigned,
        "retained_initial_assignments": len(initial) - len(initial_nulls),
        "states": len(graph["nodes"]),
        "routing_stages": len(stage_ids),
        "observed_endpoint_pairs": len(pairs),
        "history_counts_by_split": dict(
            sorted(Counter(h["split"] for h in histories.values()).items())
        ),
        "checks": {
            "immutable_corpus_hashes": True,
            "exact_history_and_rollout_census": True,
            "all_prefix_hashes": True,
            "all_transition_boundaries": True,
            "retention_of_previously_classified_rows": True,
            "fallback_stage_membership": True,
            "checkpoint_provenance_and_equality": checkpoint_exists,
        },
        "source_artifact_sha256": source_hashes,
        "verification_script_sha256": sha(Path(__file__).read_bytes()),
        "scope": "Complete accounting and prefix identity; does not certify semantic membership quality, reusable edge contracts, or executable paths",
        "model_calls": 0,
        "formation_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", "--dataset", dest="corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = verify_assignments(args.run, args.corpus)
    except ValueError as error:
        parser.exit(1, f"Verification failed: {error}. No new receipt written.\n")
    except (KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Verification failed ({type(error).__name__}); no new receipt written.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, args.output)
    print(
        f"Verified {receipt['counts']['histories']} complete histories, zero nulls, and {receipt['counts']['transitions']} transition boundaries. Receipt: {args.output}"
    )


if __name__ == "__main__":
    main()
