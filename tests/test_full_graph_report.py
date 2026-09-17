"""Public reports stay honest about partial runs and exclude private evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_full_graph_report.py"
SPEC = importlib.util.spec_from_file_location("full_graph_report", SCRIPT)
report_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_module)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    run, corpus, output = (tmp_path / name for name in ("run", "corpus", "report"))
    corpus.mkdir()
    write(
        corpus,
        "manifest.json",
        {
            "statistics": {
                "tasks": 1,
                "rollouts": 1,
                "messages": 11,
                "histories": 6,
                "transitions": 5,
                "task_split_counts": {"train": 1},
                "rollout_split_counts": {"train": 1},
                "reward_provenance_counts": {"original_verifier": 1},
            }
        },
    )
    (corpus / "histories.jsonl").write_text(
        "\n".join(
            json.dumps({"history_id": f"r:h{i:04d}", "private": "RAW_INDEX_SECRET"})
            for i in range(6)
        )
        + "\n"
    )
    candidate0, candidate1 = (
        {"state_spec": "seed", "edge_spec": "seed"},
        {"state_spec": "best", "edge_spec": "best"},
    )
    candidate_hash = hashlib.sha256(
        json.dumps(candidate1, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write(
        run,
        "run_contract.json",
        {
            "histories": 6,
            "rollouts": 1,
            "model": "Qwen/test",
            "revision": "pinned",
            "max_proposals": 100,
            "minibatch": 6,
            "dataset": "/PRIVATE/LOCAL/PATH",
            "api_key": "RAW_CREDENTIAL_SECRET",
        },
    )
    write(
        run,
        "gepa_result.json",
        {
            "candidates": [candidate0, candidate1],
            "val_aggregate_scores": [0.4, 0.8],
            "best_outputs_valset": {"private": "RAW_GEPA_TRACE_SECRET"},
        },
    )
    write(
        run,
        "heldout_test.json",
        {
            "candidate_hash": candidate_hash,
            "summary": {"rollouts": 1, "mean_score": 0.7, "metrics": {"history_coverage": 0.9}},
            "rollouts": [{"assignments": "RAW_TEST_TRACE_SECRET"}],
        },
    )
    write(run, "completion.json", {"status": "complete", "histories": 6, "unassigned": 0})
    assignments = {
        f"r:h{i:04d}": {
            "history_id": f"r:h{i:04d}",
            "state_id": "A",
            "rollout_id": "r",
            "task_id": "task",
            "evidence": "RAW_ASSIGNMENT_SECRET",
        }
        for i in range(6)
    }
    write(run, "assignments_all.json", assignments)
    witnesses = [
        {
            "transition_id": f"r:t{i:04d}",
            "rollout_id": "r",
            "task_id": "task",
            "source_id": f"r:h{i:04d}",
            "target_id": f"r:h{i + 1:04d}",
            "source_history": "RAW_SOURCE_HISTORY_SECRET",
        }
        for i in range(5)
    ]
    graph = {
        "nodes": [
            {
                "id": "A",
                "name": "<script>alert('node')</script>",
                "description": "Observe facts. api_key=sk-secretABC123456789",
                "exclusions": ["No pending tool result"],
                "transcript": "RAW_NODE_TRANSCRIPT_SECRET",
            }
        ],
        "edges": [
            {
                "id": "E",
                "source": "A",
                "target": "A",
                "operation": "Inspect again",
                "effect": "Additional facts",
                "bindings": "Rename source",
                "witnesses": witnesses,
                "distinct_task_count": 1,
                "traversable": False,
                "audit": {
                    "verdicts": {"supported": 2, "contradicted": 1},
                    "private": "RAW_AUDIT_SECRET",
                },
            }
        ],
        "unassigned_transitions": [],
    }
    write(run, "graph.json", graph)
    write(
        run,
        "state_reward_variance.json",
        {
            "states": [
                {
                    "state_id": "A",
                    "trajectory_deduplicated": {"mean": 0.5, "population_variance": 0.25},
                    "history_weighted": {"mean": 0.4, "population_variance": 0.24},
                    "task_balanced": {"mean": 0.45, "population_variance": 0.2475},
                }
            ]
        },
    )
    write(
        run,
        "independent_audit_summary.json",
        {
            "planned_checks": 3,
            "completed_checks": 3,
            "failed_requests": 0,
            "by_kind": {"edge_source_applicability": {"supported": 2, "contradicted": 1}},
            "independence": {"different_model_from_optimizer": False},
        },
    )
    write(
        run,
        "task_draft_summary.json",
        {
            "requested_examples": 8,
            "attempted_paths": 2,
            "draft_count": 1,
            "executed_task_count": 0,
            "benchmark_validated_count": 0,
            "selected_examples": [
                {
                    "title": "Draft analytics task",
                    "path_id": "path1",
                    "status": "draft",
                    "review_verdict": "plausible",
                    "executed": False,
                    "benchmark_validated": False,
                    "review": "RAW_REVIEW_SECRET",
                    "learner_instruction": "RAW_TASK_INSTRUCTION_SECRET",
                }
            ],
        },
    )
    (run / "events.jsonl").write_text(
        json.dumps({"event": "proposal", "attempt": 1, "kind": "reflection"})
        + "\n"
        + json.dumps({"event": "proposal", "attempt": 2, "kind": "crossover"})
        + "\n"
        + json.dumps({"event": "coverage_accepted", "candidate_hash": candidate_hash})
        + "\n"
        + '{"incomplete_live_append":'
    )
    return run, corpus, output


def test_empty_run_is_partial_and_does_not_invent_results(tmp_path):
    report = report_module.build_report(
        tmp_path / "missing", tmp_path / "missing_corpus", tmp_path / "out"
    )
    assert report["status"] == "partial"
    assert report["corpus"]["histories"] is None
    assert report["optimization"]["best_pareto_score"] is None
    assert report["optimization"]["accepted_revisions"] is None
    assert report["task_drafts"]["executed_task_count"] is None
    assert "PARTIAL" in (tmp_path / "out/README.md").read_text()


def test_completed_report_uses_actual_values_and_separates_evidence_tiers(tmp_path):
    run, corpus, output = fixture(tmp_path)
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "complete"
    assert report["assignments"]["exact_ids_verified"]
    assert report["assignments"]["missing_ids"] == 0
    assert report["optimization"]["proposals_observed"] == 2
    assert report["optimization"]["accepted_revisions"] == 1
    assert report["optimization"]["initial_pareto_score"] == 0.4
    assert report["optimization"]["best_pareto_score"] == 0.8
    assert report["frozen_test"]["mean_score"] == 0.7
    assert report["graph"]["sampled_traversable_edges"] == 0
    node = report["graph"]["nodes"][0]
    assert (node["member_histories"], node["distinct_rollouts"], node["distinct_tasks"]) == (
        6,
        1,
        1,
    )
    assert node["reward_variance"] == 0.25
    edge = report["graph"]["edges"][0]
    assert edge["witness_count"] == 5
    assert len(edge["witness_reference_subset"]) == 3
    assert not edge["universal_contract_certified"]
    assert report["task_drafts"]["executed_task_count"] == 0
    assert not report["task_drafts"]["selected_examples"][0]["executed"]


def test_report_does_not_publish_raw_evidence_credentials_or_executable_model_html(tmp_path):
    run, corpus, output = fixture(tmp_path)
    report_module.build_report(run, corpus, output)
    for name in ("README.md", "report.html", "report.json"):
        text = (output / name).read_text()
        assert "RAW_" not in text
        assert "/PRIVATE/LOCAL/PATH" not in text
        assert "sk-secretABC123456789" not in text
    html = (output / "report.html").read_text()
    assert "<script>alert('node')</script>" not in html
    assert "&lt;script&gt;" in html
    assert "textContent" in html and "innerHTML" not in html
    assert "https://" not in html


def test_completion_flag_does_not_hide_missing_or_unassigned_histories(tmp_path):
    run, corpus, output = fixture(tmp_path)
    assignments = json.loads((run / "assignments_all.json").read_text())
    del assignments["r:h0000"]
    assignments["r:h0001"]["state_id"] = None
    write(run, "assignments_all.json", assignments)
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "partial"
    assert report["assignments"]["missing_ids"] == 1
    assert report["assignments"]["unassigned"] == 1


def test_same_count_wrong_ids_and_duplicate_witnesses_prevent_completion(tmp_path):
    run, corpus, output = fixture(tmp_path)
    assignments = json.loads((run / "assignments_all.json").read_text())
    assignments["wrong:h0000"] = assignments.pop("r:h0000")
    write(run, "assignments_all.json", assignments)
    graph = json.loads((run / "graph.json").read_text())
    graph["edges"][0]["witnesses"][4] = graph["edges"][0]["witnesses"][0]
    write(run, "graph.json", graph)
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "partial"
    assert not report["completion_checks"]["exact_history_census"]
    assert not report["completion_checks"]["transition_census"]


def test_seed_snapshot_and_stale_heldout_are_not_final_results(tmp_path):
    run, corpus, output = fixture(tmp_path)
    heldout = json.loads((run / "heldout_test.json").read_text())
    heldout["candidate_hash"] = "stale"
    write(run, "heldout_test.json", heldout)
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "partial"
    assert not report["frozen_test"]["matches_selected_candidate"]
    assert any("does not match" in warning for warning in report["warnings"])
    (run / "graph.json").unlink()
    write(
        run,
        "seed_candidate.json",
        {
            "state_spec": json.dumps(
                {
                    "router_instructions": "test",
                    "states": [
                        {
                            "id": "A",
                            "name": "Seed state",
                            "description": "Definition",
                            "exclusions": [],
                        }
                    ],
                }
            ),
            "edge_spec": json.dumps({"edges": []}),
        },
    )
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "partial"
    assert report["graph"]["artifact"] == "seed_candidate.json"
    assert report["graph"]["observed_transitions"] is None


def test_malformed_optional_artifact_is_reported_without_leaking_contents(tmp_path):
    run, corpus, output = fixture(tmp_path)
    (run / "state_reward_variance.json").write_text("RAW_PRIVATE_BROKEN_ARTIFACT_SECRET")
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "partial"
    assert report["warnings"]
    assert "RAW_PRIVATE_BROKEN_ARTIFACT_SECRET" not in (output / "report.json").read_text()
