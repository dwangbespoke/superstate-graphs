"""Offline exact-census verifier checks retention, boundaries, and checkpoint identity."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superstate_graphs.full_corpus import (
    index_messages,
    iter_history_records,
    iter_transition_records,
)
from superstate_graphs.full_graph import _completion_provenance
from superstate_graphs.graph_evolution import GraphRuntime, candidate_from_graph


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_full_corpus_assignments.py"
SPEC = importlib.util.spec_from_file_location("census_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
COUNTS = {"histories": 6, "transitions": 4, "rollouts": 2, "tasks": 2}


def write(root, name, value):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(value))


def lines(root, name, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))


def refresh_manifest(corpus):
    write(
        corpus,
        "manifest.json",
        {
            "statistics": COUNTS,
            "artifact_sha256": {
                name: verifier.sha((corpus / name).read_bytes()) for name in verifier.CORPUS_FILES
            },
        },
    )


def fixture(tmp_path):
    run, corpus = tmp_path / "run", tmp_path / "corpus"
    rollouts = []
    for index, split in enumerate(("train", "test")):
        indexed = index_messages(
            [
                {"role": role, "content": text, "content_json": None, "sequence_number": n}
                for n, (role, text) in enumerate(
                    [
                        ("user", "RAW_PRIVATE_QUERY"),
                        ("assistant", "Inspect"),
                        ("tool", "RAW_PRIVATE_OBSERVATION"),
                        ("assistant", "Check"),
                        ("tool", "Result"),
                    ]
                )
            ]
        )
        rollouts.append(
            {
                "id": f"r{index}",
                "task_id": f"t{index}",
                "split": split,
                "reward": index,
                "reward_provenance": {"kind": "original_verifier"},
                "transcript_sha256": verifier.sha(indexed["transcript"].encode()),
                **indexed,
            }
        )
    histories = [h for rollout in rollouts for h in iter_history_records(rollout)]
    transitions = [t for rollout in rollouts for t in iter_transition_records(rollout)]
    seed_graph = {
        "router_instructions": "Observed facts",
        "states": [
            {
                "id": "S",
                "name": "Initial",
                "description": "Local observed situation",
                "exclusions": [],
            }
        ],
        "edges": [],
    }
    fallback_graph = {
        "router_instructions": "Other observed facts",
        "states": [
            {
                "id": "X1_S",
                "name": "Fallback",
                "description": "Another local situation",
                "exclusions": [],
            }
        ],
        "edges": [],
    }
    candidate, fallback = map(candidate_from_graph, (seed_graph, fallback_graph))
    initial = {
        h["history_id"]: {
            key: h[key] for key in ("history_id", "rollout_id", "task_id", "split", "step")
        }
        | {"state_id": "S" if h["step"] < 2 else None, "evidence": "RAW_PRIVATE_EVIDENCE"}
        for h in histories
    }
    assignments = copy.deepcopy(initial)
    for row in assignments.values():
        if row["state_id"] is None:
            row.update(state_id="X1_S", routing_stage=1, evidence="RAW_FALLBACK_EVIDENCE")
    graph = {
        "nodes": seed_graph["states"] + fallback_graph["states"],
        "routing_stages": [candidate, fallback],
        "edges": [],
    }
    runtime = GraphRuntime(
        SimpleNamespace(runtime={"model": "router", "revision": "r"}),
        run,
        teacher_llm=SimpleNamespace(runtime={"model": "teacher", "revision": "t"}),
    )
    write(
        run,
        "run_contract.json",
        {
            "model": "router",
            "revision": "r",
            "teacher_model": "teacher",
            "teacher_revision": "t",
            "api_key": "RAW_PRIVATE_KEY",
        },
    )
    write(run, "optimized_candidate.json", candidate)
    write(run, "optimized_assignments_all.json", initial)
    write(run, "assignments_all.json", assignments)
    write(
        run,
        "completion_checkpoint.json",
        {
            "provenance": _completion_provenance(runtime, candidate, rollouts, initial),
            "graph": graph,
            "assignments": assignments,
        },
    )
    lines(corpus, "rollouts.jsonl", rollouts)
    lines(corpus, "histories.jsonl", histories)
    lines(corpus, "transitions.jsonl", transitions)
    write(
        corpus,
        "splits.json",
        {
            "train": {"task_ids": ["t0"], "rollout_ids": ["r0"]},
            "test": {"task_ids": ["t1"], "rollout_ids": ["r1"]},
            "pareto": {"task_ids": [], "rollout_ids": []},
        },
    )
    refresh_manifest(corpus)
    return run, corpus


def test_complete_census_receipt_contains_all_counts_and_no_private_content(tmp_path):
    run, corpus = fixture(tmp_path)
    receipt = verifier.verify_assignments(run, corpus, expected_counts=COUNTS)
    assert receipt["status"] == "zero_null_completion_verified"
    assert receipt["counts"] == COUNTS
    assert receipt["prefix_hashes_verified"] == 6
    assert receipt["initial_unassigned"] == receipt["newly_classified"] == 2
    assert receipt["retained_initial_assignments"] == 4
    assert receipt["states"] == receipt["routing_stages"] == receipt["observed_endpoint_pairs"] == 2
    assert "RAW_" not in json.dumps(receipt)
    assert receipt["source_artifact_sha256"]["run/assignments_all.json"] == verifier.sha(
        (run / "assignments_all.json").read_bytes()
    )


@pytest.mark.parametrize(
    "corruption", ["missing", "null", "retention", "stage", "provenance", "prefix", "endpoint"]
)
def test_census_verifier_rejects_corrupt_assignments_or_correlated_records(tmp_path, corruption):
    run, corpus = fixture(tmp_path)
    assignment = json.loads((run / "assignments_all.json").read_text())
    checkpoint = json.loads((run / "completion_checkpoint.json").read_text())
    if corruption == "missing":
        del assignment["r0:h0000"]
    elif corruption == "null":
        assignment["r0:h0002"]["state_id"] = None
    elif corruption == "retention":
        assignment["r0:h0000"]["evidence"] = "Changed old routing evidence"
    elif corruption == "stage":
        assignment["r0:h0002"]["routing_stage"] = 0
    elif corruption == "provenance":
        checkpoint["provenance"]["initial_assignments_hash"] = "0" * 64
    elif corruption in ("prefix", "endpoint"):
        name = "histories.jsonl" if corruption == "prefix" else "transitions.jsonl"
        rows = [json.loads(line) for line in (corpus / name).read_text().splitlines()]
        if corruption == "prefix":
            rows[0]["prefix_sha256"] = "0" * 64
        else:
            rows[0]["target_history_id"] = "r1:h0001"
        lines(corpus, name, rows)
        refresh_manifest(corpus)  # Content checks must fail even when the manifest is updated too.
    checkpoint["assignments"] = assignment
    write(run, "assignments_all.json", assignment)
    write(run, "completion_checkpoint.json", checkpoint)
    with pytest.raises(ValueError):
        verifier.verify_assignments(run, corpus, expected_counts=COUNTS)
