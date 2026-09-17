"""Final export preserves every identity while excluding private source text."""

from __future__ import annotations

import gzip
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/export_full_graph_artifacts.py"
SPEC = importlib.util.spec_from_file_location("export_complete_graph", PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
COUNTS = {"tasks": 2, "rollouts": 2, "histories": 6, "transitions": 4}


def write(root: Path, name: str, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_lines(root: Path, name: str, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(tmp_path):
    run, corpus, output = (tmp_path / name for name in ("run", "corpus", "artifacts"))
    state = {
        "id": "S",
        "name": "Local situation",
        "description": "Source is available for inspection",
        "exclusions": [],
    }
    edge = {
        "id": "E",
        "source": "S",
        "target": "S",
        "operation": "Inspect further",
        "effect": "More evidence",
        "bindings": "Rename source",
    }
    candidate = {
        "state_spec": json.dumps({"router_instructions": "Use observed facts", "states": [state]}),
        "edge_spec": json.dumps({"edges": [edge]}),
    }
    selected_hash = exporter.digest_bytes(exporter.encoded(candidate))
    histories, transitions, rollouts, assignments, witnesses = [], [], [], {}, []
    for r in range(2):
        rid, task, split = f"r{r}", f"task{r}", "train" if r == 0 else "test"
        provenance = {
            "kind": "original_verifier" if r == 0 else "manual_trace_review",
            "source": f"rollouts/{rid}/detail.json",
            "source_sha256": "source-hash",
            "reason": "RAW_GRADE_REASON_SECRET",
        }
        rollouts.append(
            {
                "id": rid,
                "rollout_id": rid,
                "task_id": task,
                "split": split,
                "reward": r,
                "reward_provenance": provenance,
                "history_count": 3,
                "query": "RAW_QUERY_SECRET",
                "transcript": "RAW_TRANSCRIPT_SECRET",
            }
        )
        for step in range(3):
            hid = f"{rid}:h{step:04d}"
            h = {
                "history_id": hid,
                "rollout_id": rid,
                "task_id": task,
                "split": split,
                "step": step,
                "prefix_char_end": 10 * (step + 1),
                "prefix_sha256": "hash",
                "reward": r,
                "reward_provenance": provenance,
                "last_observed_history": step == 2,
            }
            histories.append(h)
            assignments[hid] = {
                key: h[key] for key in ("history_id", "rollout_id", "task_id", "split", "step")
            }
            assignments[hid].update({"state_id": "S", "evidence": "RAW_ASSIGNMENT_SECRET"})
            if step < 2:
                tid, target = f"{rid}:t{step:04d}", f"{rid}:h{step + 1:04d}"
                transitions.append(
                    {
                        "transition_id": tid,
                        "source_history_id": hid,
                        "target_history_id": target,
                        "rollout_id": rid,
                        "task_id": task,
                        "split": split,
                        "step": step,
                    }
                )
                witnesses.append(
                    {
                        "transition_id": tid,
                        "source_id": hid,
                        "target_id": target,
                        "rollout_id": rid,
                        "task_id": task,
                        "step": step,
                        "source_history": "RAW_WITNESS_SECRET",
                    }
                )
    write_lines(corpus, "histories.jsonl", histories)
    write_lines(corpus, "transitions.jsonl", transitions)
    write_lines(corpus, "rollouts.jsonl", rollouts)
    write(
        corpus,
        "manifest.json",
        {
            "statistics": COUNTS
            | {"reward_provenance_counts": {"original_verifier": 1, "manual_trace_review": 1}}
        },
    )
    write(
        corpus,
        "splits.json",
        {
            "seed": "test",
            "train": {"task_ids": ["task0"], "rollout_ids": ["r0"]},
            "pareto": {"task_ids": [], "rollout_ids": []},
            "test": {"task_ids": ["task1"], "rollout_ids": ["r1"]},
        },
    )
    write(
        run,
        "run_contract.json",
        {"model": "Qwen/test", "histories": 6, "rollouts": 2, "api_key": "RAW_RUNTIME_SECRET"},
    )
    write(run, "optimized_candidate.json", candidate)
    write(run, "seed_candidate.json", candidate)
    write(
        run,
        "gepa_result.json",
        {
            "candidates": [candidate],
            "parents": [[None]],
            "val_aggregate_scores": [0.7],
            "best_idx": 0,
            "validation_schema_version": 2,
        },
    )
    write(
        run,
        "heldout_test.json",
        {
            "candidate_hash": selected_hash,
            "summary": {"rollouts": 1, "mean_score": 0.6, "metrics": {}},
        },
    )
    write(run, "assignments_all.json", assignments)
    write(
        run,
        "graph.json",
        {
            "nodes": [state],
            "edges": [
                {
                    **edge,
                    "witnesses": witnesses,
                    "traversable": True,
                    "evidence_status": "Sampled support",
                    "audit": {"verdicts": {"supported": 3}, "judge_prompt": "RAW_JUDGE_SECRET"},
                }
            ],
            "routing_stages": [candidate],
            "unassigned_transitions": [],
        },
    )
    moment = {
        "weight": 2,
        "mean": 0.5,
        "population_variance": 0.25,
        "task_cluster_bootstrap": {
            "mean_interval": [0, 1],
            "confidence": 0.95,
            "status": "computed",
        },
    }
    write(
        run,
        "state_reward_variance.json",
        {
            "histories": 6,
            "assigned_histories": 6,
            "missing_assignment_count": 0,
            "unassigned_count": 0,
            "states": [
                {
                    "state_id": "S",
                    "member_histories": 6,
                    "distinct_visiting_rollouts": 2,
                    "distinct_visiting_tasks": 2,
                    "history_weighted": {**moment, "weight": 6},
                    "trajectory_deduplicated": moment,
                    "task_balanced": moment,
                    "per_task": [
                        {
                            "task_id": f"task{i}",
                            "rewarded_rollouts": 1,
                            "mean": i,
                            "population_variance": 0,
                        }
                        for i in range(2)
                    ],
                    "sensitivity_excluding_manual_rewards": {
                        "history_weighted": {"weight": 3, "mean": 0, "population_variance": 0},
                        "task_balanced": {"weight": 1, "mean": 0, "population_variance": 0},
                        "per_task": [
                            {
                                "task_id": "task0",
                                "rewarded_rollouts": 1,
                                "mean": 0,
                                "population_variance": 0,
                            }
                        ],
                        "trajectory_deduplicated": {
                            "weight": 1,
                            "mean": 0,
                            "population_variance": 0,
                        },
                    },
                    "private_evidence": "RAW_VARIANCE_SECRET",
                }
            ],
        },
    )
    write(
        run,
        "independent_audit_summary.json",
        {"planned_checks": 3, "completed_checks": 3, "failed_requests": 0, "by_kind": {}},
    )
    write(run, "task_draft_summary.json", {"selected_examples": [], "draft_count": 0})
    write(run, "completion.json", {"status": "complete", "histories": 6, "unassigned": 0})
    return run, corpus, output


def load_lines(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def test_final_export_preserves_all_memberships_witnesses_and_variance(tmp_path):
    run, corpus, output = fixture(tmp_path)
    result = exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert result["assignment_records"] == 6 and result["witness_records"] == 4
    assignments = load_lines(output / "assignments.jsonl.gz")
    assert {row["history_id"] for row in assignments} == {
        f"r{r}:h{k:04d}" for r in range(2) for k in range(3)
    }
    assert set(load_lines(output / "nodes.jsonl.gz")[0]["member_history_ids"]) == {
        row["history_id"] for row in assignments
    }
    witnesses = load_lines(output / "witnesses.jsonl.gz")
    assert len(witnesses) == 4
    assert set(load_lines(output / "edges.jsonl.gz")[0]["witness_ids"]) == {
        w["transition_id"] for w in witnesses
    }
    variance = json.loads((output / "state_reward_variance.json").read_text())
    assert variance["states"][0]["trajectory_deduplicated"]["population_variance"] == 0.25
    assert variance["states"][0]["trajectory_deduplicated"]["task_cluster_bootstrap"][
        "mean_interval"
    ] == [0, 1]
    assert (
        variance["states"][0]["sensitivity_excluding_manual_rewards"]["trajectory_deduplicated"][
            "weight"
        ]
        == 1
    )
    for name, receipt in result["files"].items():
        raw = (output / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        text = gzip.decompress(raw).decode() if name.endswith(".gz") else raw.decode()
        assert "RAW_" not in text
    assert exporter.read_json(output / "selected_candidate.json") == exporter.read_json(
        run / "optimized_candidate.json"
    )
    assert not load_lines(output / "edges.jsonl.gz")[0]["audit"]["universal_contract_certified"]


def test_export_requires_actual_completion_and_default_entire_corpus(tmp_path):
    run, corpus, output = fixture(tmp_path)
    with pytest.raises(ValueError, match="Full-corpus export expected"):
        exporter.export_artifacts(run, corpus, output)
    (run / "completion.json").unlink()
    with pytest.raises(ValueError, match="complete verified artifacts"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert not output.exists()


def test_gzip_is_deterministic_and_existing_export_cannot_be_overwritten(tmp_path):
    run, corpus, output = fixture(tmp_path)
    a = exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    b = exporter.export_artifacts(run, corpus, tmp_path / "second", expected_counts=COUNTS)
    for name in a["files"]:
        if name.endswith(".gz"):
            assert a["files"][name] == b["files"][name]
    with pytest.raises(FileExistsError):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)


def test_definition_credentials_block_publication_without_changing_definitions(tmp_path):
    run, corpus, output = fixture(tmp_path)
    graph = exporter.read_json(run / "graph.json")
    graph["nodes"][0]["description"] = "api_key=sk-doNotPublish123456789"
    write(run, "graph.json", graph)
    with pytest.raises(ValueError, match="Possible credential"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert not output.exists()
    assert not list(output.parent.glob(".complete-graph-export-*"))


def test_wrong_witness_endpoint_or_assignment_metadata_prevents_export(tmp_path):
    run, corpus, output = fixture(tmp_path)
    graph = exporter.read_json(run / "graph.json")
    graph["edges"][0]["witnesses"][0]["target_id"] = "r1:h0002"
    write(run, "graph.json", graph)
    with pytest.raises(ValueError, match="Witness boundary"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    graph["edges"][0]["witnesses"][0]["target_id"] = "r0:h0001"
    write(run, "graph.json", graph)
    assignments = exporter.read_json(run / "assignments_all.json")
    assignments["r0:h0000"]["step"] = 99
    write(run, "assignments_all.json", assignments)
    with pytest.raises(ValueError, match="metadata disagree"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)


@pytest.mark.parametrize("moment", ["history_weighted", "trajectory_deduplicated", "task_balanced"])
def test_variance_moments_are_recomputed_from_immutable_outcomes(tmp_path, moment):
    run, corpus, output = fixture(tmp_path)
    variance = exporter.read_json(run / "state_reward_variance.json")
    variance["states"][0][moment]["mean"] = 0.7
    write(run, "state_reward_variance.json", variance)
    with pytest.raises(ValueError, match="variance moment mismatch"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert not output.exists()


def test_manual_grade_sensitivity_and_per_task_census_are_verified(tmp_path):
    run, corpus, output = fixture(tmp_path)
    variance = exporter.read_json(run / "state_reward_variance.json")
    variance["states"][0]["sensitivity_excluding_manual_rewards"]["trajectory_deduplicated"][
        "weight"
    ] = 2
    write(run, "state_reward_variance.json", variance)
    with pytest.raises(ValueError, match="variance moment mismatch"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    variance["states"][0]["sensitivity_excluding_manual_rewards"]["trajectory_deduplicated"][
        "weight"
    ] = 1
    variance["states"][0]["per_task"].pop()
    write(run, "state_reward_variance.json", variance)
    with pytest.raises(ValueError, match="per-task rows"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)


def add_task(run):
    root = run / "executable_tasks/path_1"
    root.mkdir(parents=True)
    (root / "instruction.md").write_text("Return the total from the synthetic fixture.")
    (root / "fixture.duckdb").write_bytes(b"UNIT_TEST_FIXTURE_BYTES_NOT_A_REAL_DATABASE")
    (root / "oracle.sql").write_text("SELECT 3 AS total")
    (root / "independent.sql").write_text("SELECT 1+2 AS total")
    write(root, "expected_result.json", {"columns": ["total"], "rows": [[3]]})
    write(
        root,
        "task.json",
        {
            "status": "task",
            "path_id": "path_1",
            "path_transition_ids": ["E"],
            "ordered_output": False,
            "source_history": "RAW_TASK_SOURCE_SECRET",
        },
    )
    write(
        root,
        "source_path.json",
        {
            "path_id": "path_1",
            "state_ids": ["S", "S"],
            "transitions": [
                {
                    "transition_id": "E",
                    "source_history_id": "r0:h0000",
                    "target_history_id": "r0:h0001",
                    "private": "RAW_PATH_SECRET",
                }
            ],
            "private": "RAW_PATH_CONTEXT_SECRET",
        },
    )
    write(
        root,
        "local_validation.json",
        {
            "status": "locally_executed_consistent",
            "reference_query_executed": True,
            "independent_query_executed": True,
            "queries_agree": True,
            "request": "RAW_LOCAL_VALIDATION_REQUEST_SECRET",
        },
    )
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
    write(root, "result.json", {"status": "locally_executed_consistent", "artifact_sha256": hashes})
    write(
        run,
        "executable_task_summary.json",
        {
            "status": "target_reached",
            "requested_examples": 1,
            "target_reached": True,
            "locally_executed_consistent_count": 1,
            "selected_examples": [
                {
                    "status": "locally_executed_consistent",
                    "path_id": "path_1",
                    "local_validation": {
                        "reference_query_executed": True,
                        "independent_query_executed": True,
                        "queries_agree": True,
                    },
                }
            ],
        },
    )
    return root


def test_optional_task_export_uses_receipt_hashes_and_only_allowed_files(tmp_path):
    run, corpus, output = fixture(tmp_path)
    source = add_task(run)
    result = exporter.export_artifacts(
        run, corpus, output, include_tasks=True, expected_counts=COUNTS
    )
    assert result["executable_tasks"] == 1
    target = output / "executable_tasks/path_1"
    assert (target / "fixture.duckdb").read_bytes() == (source / "fixture.duckdb").read_bytes()
    assert set(path.name for path in target.iterdir()) == {
        "instruction.md",
        "fixture.duckdb",
        "oracle.sql",
        "independent.sql",
        "expected_result.json",
        "local_validation.json",
        "task.json",
        "source_path.json",
        "README.md",
    }
    assert "RAW_" not in (target / "task.json").read_text()
    assert "RAW_" not in (target / "local_validation.json").read_text()
    assert exporter.read_json(target / "task.json")["ordered_output"] is False
    assert exporter.read_json(target / "task.json")["path_transition_ids"] == ["E"]
    path = exporter.read_json(target / "source_path.json")
    assert path["graph_edge_ids"] == ["E"]
    assert path["witnesses"][0]["transition_id"] == "r0:t0000"
    assert "RAW_" not in (target / "source_path.json").read_text()


def test_optional_tasks_require_matching_execution_receipts(tmp_path):
    run, corpus, output = fixture(tmp_path)
    source = add_task(run)
    (source / "oracle.sql").write_text("SELECT 999 AS total")
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.export_artifacts(run, corpus, output, include_tasks=True, expected_counts=COUNTS)
    assert not output.exists()


def test_executed_task_requires_graph_path_witness_even_with_matching_file_receipt(tmp_path):
    run, corpus, output = fixture(tmp_path)
    source = add_task(run)
    path = exporter.read_json(source / "source_path.json")
    path["transitions"][0]["target_history_id"] = "r1:h0001"
    write(source, "source_path.json", path)
    receipt = exporter.read_json(source / "result.json")
    receipt["artifact_sha256"]["source_path.json"] = hashlib.sha256(
        (source / "source_path.json").read_bytes()
    ).hexdigest()
    write(source, "result.json", receipt)
    with pytest.raises(ValueError, match="path witness"):
        exporter.export_artifacts(run, corpus, output, include_tasks=True, expected_counts=COUNTS)
    assert not output.exists()


def test_exported_real_synthetic_fixture_remains_executable_with_standalone_verifier(tmp_path):
    from superstate_graphs.graph_task_examples import run_local_validation, verify_submission

    run, corpus, output = fixture(tmp_path)
    source = add_task(run)
    spec = {
        "status": "task",
        "path_id": "path_1",
        "path_transition_ids": ["E"],
        "title": "Unit-test fixture",
        "learner_instruction": "Return the sum as total.",
        "output_columns": ["total"],
        "ordered_output": False,
        "tables": [
            {"name": "numbers", "columns": [{"name": "n", "type": "INTEGER"}], "rows": [[10], [20]]}
        ],
        "reference_sql": "SELECT SUM(n) AS total FROM numbers",
    }
    independent = "SELECT SUM(n * 1) AS total FROM numbers"
    validation = run_local_validation(spec, independent, source)
    assert validation["status"] == "locally_executed_consistent"
    write(source, "task.json", spec)
    (source / "oracle.sql").write_text(spec["reference_sql"])
    (source / "independent.sql").write_text(independent)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.iterdir()
        if path.name != "result.json"
    }
    write(
        source, "result.json", {"status": "locally_executed_consistent", "artifact_sha256": hashes}
    )
    exporter.export_artifacts(run, corpus, output, include_tasks=True, expected_counts=COUNTS)
    task = output / "executable_tasks/path_1"
    assert verify_submission(task, spec["reference_sql"])["status"] == "passed"
    assert verify_submission(task, "SELECT 0 AS total")["status"] != "passed"


def test_paired_comparison_export_keeps_all_ids_and_aggregates_without_inference_rows(tmp_path):
    run, corpus, output = fixture(tmp_path)
    selected = exporter.read_json(run / "heldout_test.json")
    selected.update(
        evaluation_identity="same-eval",
        rollouts=[{"rollout_id": "r1", "feedback": "RAW_EVIDENCE_SECRET"}],
    )
    write(run, "heldout_test.json", selected)
    write(run, "baseline_heldout.json", selected)
    write(
        run,
        "heldout_comparison.json",
        {
            "format": "frozen-heldout-seed-selected-v1",
            "seed_candidate_hash": selected["candidate_hash"],
            "selected_candidate_hash": selected["candidate_hash"],
            "seed_evaluation_identity": "same-eval",
            "selected_evaluation_identity": "same-eval",
            "selected_equals_seed": True,
            "rollouts": 1,
            "original_tasks": 1,
            "common_rollout_ids": ["r1"],
            "before_transductive_completion": True,
            "optimization_uses_test_feedback": False,
            "components": {
                "score": {
                    "seed_mean": 0.6,
                    "selected_mean": 0.6,
                    "task_balanced_difference": 0,
                    "paired_task_bootstrap_95ci": None,
                    "evidence": "RAW_AGGREGATE_SECRET",
                }
            },
            "per_task": {
                "task1": {
                    "rollouts": 1,
                    "components": {
                        "score": {
                            "seed_mean": 0.6,
                            "selected_mean": 0.6,
                            "difference": 0,
                            "private": "RAW_TASK_SECRET",
                        }
                    },
                }
            },
            "bootstrap": {"draws": 0},
            "transcript": "RAW_COMPARISON_SECRET",
        },
    )
    exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    comparison = exporter.read_json(output / "heldout_comparison.json")
    assert comparison["identities_and_census_verified"] is True
    assert comparison["common_rollout_ids"] == ["r1"]
    assert comparison["components"]["score"]["seed_mean"] == 0.6
    assert comparison["per_task"]["task1"]["components"]["score"]["difference"] == 0
    assert "RAW_" not in (output / "heldout_comparison.json").read_text()
    assert not (output / "baseline_heldout.json").exists()
    manifest = exporter.read_json(output / "manifest.json")
    assert "heldout_comparison.json" in manifest["files"]


def test_declared_optimizer_continuation_exports_only_validated_receipt_fields(tmp_path):
    run, corpus, output = fixture(tmp_path)
    receipt = {
        "boundary_after_proposal": 7,
        "reason": "Repair reflection evidence attachment.",
        "old_graph_evolution_sha256": "a" * 64,
        "new_graph_evolution_sha256": "b" * 64,
        "unchanged_evaluation_identity": True,
        "evidence_only_repair": True,
        "archived_checkpoint_sha256": "c" * 64,
        "resumed_at_utc": "2026-09-17T15:00:00+00:00",
    }
    write(
        run,
        "optimizer_continuations.json",
        [{**receipt, "source_path": "RAW_PRIVATE_SOURCE_SECRET"}],
    )
    manifest = exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert exporter.read_json(output / "optimizer_continuations.json") == [receipt]
    provenance = exporter.read_json(output / "provenance.json")
    assert provenance["optimizer_continuations"]["validation_status"] == "valid"
    assert "optimizer_continuations.json" in manifest["files"]
    assert "RAW_" not in (output / "optimizer_continuations.json").read_text()
    assert "RAW_" not in (output / "provenance.json").read_text()


def test_invalid_optimizer_continuation_cannot_be_exported_as_complete(tmp_path):
    run, corpus, output = fixture(tmp_path)
    write(run, "optimizer_continuations.json", {"reason": "RAW_INVALID_RECEIPT_SECRET"})
    with pytest.raises(ValueError, match="optimizer_continuation_receipts"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert not output.exists()


def archive_fixture(run):
    selected = exporter.read_json(run / "optimized_candidate.json")
    candidates = []
    for instruction in ("Initial seed routing", "First accepted branch", "Second accepted branch"):
        state = json.loads(selected["state_spec"])
        state["router_instructions"] = instruction
        candidates.append({**selected, "state_spec": json.dumps(state)})
    candidates.append(selected)
    result = {
        "candidates": candidates,
        "parents": [[None], [0], [0], [1, 2]],
        "val_aggregate_scores": [0.1, 0.3, 0.2, 0.7],
        "best_idx": 3,
        "validation_schema_version": 2,
        "best_outputs_valset": {"private": "RAW_MODEL_OUTPUT_SECRET"},
        "val_subscores": [{"private": "RAW_PER_ROLLOUT_SECRET"}],
        "reflection_dataset": "RAW_REFLECTION_SECRET",
        "run_dir": "/PRIVATE/RUN/PATH",
    }
    write(run, "seed_candidate.json", candidates[0])
    write(run, "gepa_result.json", result)
    return result, candidates[0], selected


def test_export_preserves_seed_accepted_specs_actual_parent_lineage_and_pareto_scores(tmp_path):
    run, corpus, output = fixture(tmp_path)
    recorded, seed, selected = archive_fixture(run)
    manifest = exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert exporter.read_json(output / "seed_candidate.json") == seed
    archive = exporter.read_json(output / "accepted_candidate_archive.json")
    for key in (
        "candidates",
        "parents",
        "val_aggregate_scores",
        "best_idx",
        "validation_schema_version",
    ):
        assert archive[key] == recorded[key]
    assert archive["candidates"][archive["best_idx"]] == selected
    assert archive["candidate_sha256"] == [
        exporter.digest_bytes(exporter.encoded(c)) for c in recorded["candidates"]
    ]
    assert (
        archive["source_gepa_result_sha256"]
        == hashlib.sha256((run / "gepa_result.json").read_bytes()).hexdigest()
    )
    assert {"seed_candidate.json", "accepted_candidate_archive.json"} <= set(manifest["files"])
    assert "RAW_" not in (output / "accepted_candidate_archive.json").read_text()
    assert "/PRIVATE/RUN/PATH" not in (output / "accepted_candidate_archive.json").read_text()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r.pop("parents"), "requires recorded"),
        (lambda r: r["parents"].pop(), "cardinalities"),
        (lambda r: r["val_aggregate_scores"].pop(), "cardinalities"),
        (lambda r: r.update(best_idx=0), "best_idx"),
        (lambda r: r["parents"].__setitem__(1, [1]), "earlier parent"),
        (lambda r: r["parents"].__setitem__(1, [None]), "earlier parent"),
        (lambda r: r["parents"].__setitem__(0, []), "seed parent"),
        (lambda r: r["val_aggregate_scores"].__setitem__(1, float("nan")), "finite numbers"),
        (lambda r: r.update(validation_schema_version=3), "schema version 2"),
    ],
)
def test_archive_refuses_missing_inconsistent_or_invalid_recorded_values(tmp_path, mutate, match):
    run, _, _ = fixture(tmp_path)
    result, seed, selected = archive_fixture(run)
    mutate(result)
    with pytest.raises(ValueError, match=match):
        exporter.accepted_candidate_archive(result, seed, selected)


def test_archive_seed_selected_and_first_maximum_tie_rules_are_verified(tmp_path):
    run, _, _ = fixture(tmp_path)
    result, seed, selected = archive_fixture(run)
    with pytest.raises(ValueError, match="index 0 differs"):
        exporter.accepted_candidate_archive(result, selected, selected)
    with pytest.raises(ValueError, match="best candidate differs"):
        exporter.accepted_candidate_archive(result, seed, seed)
    result["val_aggregate_scores"][1] = 0.7
    with pytest.raises(ValueError, match="first-maximum"):
        exporter.accepted_candidate_archive(result, seed, selected)
    result["best_idx"] = 1
    archive = exporter.accepted_candidate_archive(result, seed, result["candidates"][1])
    assert archive["best_idx"] == 1


def test_archive_applies_candidate_evidence_allowlist_to_all_older_versions(tmp_path):
    run, _, _ = fixture(tmp_path)
    result, seed, selected = archive_fixture(run)
    old_state = json.loads(result["candidates"][1]["state_spec"])
    old_state["states"][0]["private_feedback"] = "RAW_CANDIDATE_FEEDBACK_SECRET"
    result["candidates"][1]["state_spec"] = json.dumps(old_state)
    with pytest.raises(ValueError, match="Unexpected state evidence fields"):
        exporter.accepted_candidate_archive(result, seed, selected)


def test_final_export_does_not_reconstruct_missing_seed_or_lineage(tmp_path):
    run, corpus, output = fixture(tmp_path)
    (run / "seed_candidate.json").unlink()
    with pytest.raises(ValueError, match="recorded seed_candidate"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert not output.exists()
    result = exporter.read_json(run / "gepa_result.json")
    write(run, "seed_candidate.json", result["candidates"][0])
    result = copy.deepcopy(result)
    result.pop("best_idx")
    write(run, "gepa_result.json", result)
    with pytest.raises(ValueError, match="requires recorded"):
        exporter.export_artifacts(run, corpus, output, expected_counts=COUNTS)
    assert not output.exists()
