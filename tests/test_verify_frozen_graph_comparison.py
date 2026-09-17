"""Independent arithmetic verifier rejects corrupt artifacts without inference."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superstate_graphs.full_corpus import index_messages
from superstate_graphs.full_graph import evaluate_heldout_comparison
from superstate_graphs.graph_evolution import (
    GraphRuntime,
    candidate_from_graph,
    digest,
    validate_spec,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_frozen_graph_comparison.py"
SPEC = importlib.util.spec_from_file_location("frozen_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def write(root, filename, value):
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(json.dumps(value))


def fixture(tmp_path):
    run, corpus = tmp_path / "run", tmp_path / "corpus"
    graph = {
        "router_instructions": "Use observed evidence",
        "states": [
            {"id": "A", "name": "Observed", "description": "Local evidence", "exclusions": []}
        ],
        "edges": [],
    }
    seed = candidate_from_graph(graph)
    graph["edges"] = [
        {
            "id": "E",
            "source": "A",
            "target": "A",
            "operation": "Inspect",
            "effect": "More evidence",
            "bindings": "Rename source",
        }
    ]
    selected = candidate_from_graph(graph)
    rollouts = []
    # Unequal task sizes make accidental rollout weighting detectable.
    for rid, task in (("a1", "A"), ("a2", "A"), ("b1", "B")):
        indexed = index_messages(
            [
                {"role": role, "content": text, "content_json": None, "sequence_number": i}
                for i, (role, text) in enumerate(
                    [
                        ("user", "RAW_PRIVATE_QUERY"),
                        ("assistant", "Inspect source"),
                        ("tool", "RAW_PRIVATE_OBSERVATION"),
                    ]
                )
            ]
        )
        rollouts.append({"id": rid, "task_id": task, "split": "test", **indexed})

    class OfflineRuntime(GraphRuntime):
        async def batch(self, candidate, examples, *, judge):
            assert judge
            graph = validate_spec(candidate)
            output = []
            for example in examples:
                improved = candidate == selected and example["task_id"] == "A"
                metrics = {
                    "history_coverage": 1.0,
                    "transition_coverage": float(improved),
                    "coherence": 0.6,
                    "outgoing_applicability": 0.4,
                    "complexity": len(verifier.canonical(graph)) / 100000,
                }
                score = (
                    0.25
                    + 0.50 * metrics["transition_coverage"]
                    + 0.15 * 0.6
                    + 0.10 * 0.4
                    - 0.03 * metrics["complexity"]
                )
                output.append(
                    {
                        "rollout_id": example["id"],
                        "task_id": example["task_id"],
                        "split": "test",
                        "candidate_hash": digest(candidate),
                        "cache_provenance": self._cache_provenance(candidate, example, judge=True),
                        "score": score,
                        "metrics": metrics,
                        "assignments": [
                            {
                                "history_id": f"{example['id']}:h{k:04d}",
                                "step": k,
                                "state_id": "A",
                                "evidence": "RAW_PRIVATE_ROUTING",
                            }
                            for k in range(2)
                        ],
                        "membership_supported": [True, True],
                        "transitions": [
                            {"step": 0, "supported": improved, "edge_id": "E" if improved else None}
                        ],
                    }
                )
            return list(reversed(output))  # Pairing must use IDs, not output row order.

    runtime = OfflineRuntime(
        SimpleNamespace(runtime={"model": "test-router", "revision": "r1"}),
        run,
        teacher_llm=SimpleNamespace(runtime={"model": "test-teacher", "revision": "t1"}),
    )
    asyncio.run(
        evaluate_heldout_comparison(runtime, seed, selected, rollouts, run, bootstrap_draws=100)
    )
    write(run, "seed_candidate.json", seed)
    write(run, "optimized_candidate.json", selected)
    write(
        run,
        "gepa_result.json",
        {"candidates": [seed, selected], "val_aggregate_scores": [0.3, 0.4], "best_idx": 1},
    )
    write(
        run,
        "run_contract.json",
        {
            "model": "test-router",
            "revision": "r1",
            "teacher_model": "test-teacher",
            "teacher_revision": "t1",
            "api_key": "RAW_PRIVATE_KEY",
            "endpoint": "RAW_PRIVATE_URL",
        },
    )
    write(
        corpus,
        "splits.json",
        {"test": {"task_ids": ["A", "B"], "rollout_ids": [r["id"] for r in rollouts]}},
    )
    (corpus / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rollouts))
    write(
        corpus,
        "manifest.json",
        {
            "artifact_sha256": {
                name: verifier.file_digest(corpus / name)
                for name in ("rollouts.jsonl", "splits.json")
            }
        },
    )
    return run, corpus


def test_offline_verifier_recomputes_unequal_task_weighting_and_safe_receipt(tmp_path):
    run, corpus = fixture(tmp_path)
    receipt = verifier.verify_comparison(run, corpus)
    assert receipt["status"] == "passed"
    assert (receipt["rollouts"], receipt["original_task_clusters"], receipt["histories"]) == (
        3,
        2,
        6,
    )
    assert receipt["seed_selected_change_scope"] == "edge_only"
    assert receipt["routing_assignment_records_identical"]
    assert receipt["bootstrap"]["draws"] == 100
    assert receipt["maximum_absolute_error"] < 1e-12
    assert receipt["model_calls"] == 0
    assert "RAW_PRIVATE" not in json.dumps(receipt)
    comparison = verifier.read_json(run / "heldout_comparison.json")
    assert comparison["components"]["score"]["mean_difference"] != pytest.approx(
        receipt["score"]["task_balanced_difference"]
    )


@pytest.mark.parametrize(
    "corruption", ["score", "interval", "identity", "census", "corpus", "model"]
)
def test_offline_verifier_rejects_corrupt_comparison_inputs(tmp_path, corruption):
    run, corpus = fixture(tmp_path)
    filename = "heldout_test.json"
    value = verifier.read_json(run / filename)
    if corruption == "score":
        value["rollouts"][0]["score"] += 0.01
    elif corruption == "identity":
        value["evaluation_identity"] = "0" * 64
    elif corruption == "census":
        value["rollouts"][0] = value["rollouts"][1]
    elif corruption == "interval":
        filename = "heldout_comparison.json"
        value = verifier.read_json(run / filename)
        value["components"]["score"]["paired_task_bootstrap_95ci"][0] += 0.01
    elif corruption == "model":
        filename = "run_contract.json"
        value = verifier.read_json(run / filename)
        value["revision"] = "different-model-revision"
    elif corruption == "corpus":
        with (corpus / "rollouts.jsonl").open("a") as stream:
            stream.write("\n")
    write(run, filename, value)
    with pytest.raises(ValueError):
        verifier.verify_comparison(run, corpus)
