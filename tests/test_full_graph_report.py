"""Public reports stay honest about partial runs and exclude private evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET


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
            "histories": 6,
            "assigned_histories": 6,
            "missing_assignment_count": 0,
            "unassigned_count": 0,
            "states": [
                {
                    "state_id": "A",
                    "member_histories": 6,
                    "distinct_visiting_rollouts": 1,
                    "distinct_visiting_tasks": 1,
                    "trajectory_deduplicated": {
                        "weight": 1,
                        "mean": 0.5,
                        "population_variance": 0.25,
                    },
                    "history_weighted": {"weight": 6, "mean": 0.4, "population_variance": 0.24},
                    "task_balanced": {"weight": 1, "mean": 0.45, "population_variance": 0.2475},
                }
            ],
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


def test_variance_must_cover_every_occupied_state_with_exact_membership_counts(tmp_path):
    run, corpus, output = fixture(tmp_path)
    variance = json.loads((run / "state_reward_variance.json").read_text())
    variance["states"][0]["member_histories"] = 5
    write(run, "state_reward_variance.json", variance)
    report = report_module.collect_report(run, corpus)
    assert report["status"] == "partial"
    assert not report["completion_checks"]["reward_variance_census"]
    variance["states"] = []
    write(run, "state_reward_variance.json", variance)
    assert not report_module.collect_report(run, corpus)["completion_checks"][
        "reward_variance_census"
    ]


def test_recorded_transition_without_an_edge_is_not_a_complete_graph(tmp_path):
    run, corpus, output = fixture(tmp_path)
    graph = json.loads((run / "graph.json").read_text())
    graph["unassigned_transitions"] = [graph["edges"][0]["witnesses"].pop()]
    write(run, "graph.json", graph)
    report = report_module.collect_report(run, corpus)
    assert report["completion_checks"]["transition_census"]
    assert not report["completion_checks"]["all_transitions_have_edges"]
    assert report["status"] == "partial"


def execution_fixture(run, *, status="target_reached", reported_count=1):
    write(
        run,
        "executable_task_summary.json",
        {
            "status": status,
            "requested_examples": 1 if status == "target_reached" else 3,
            "target_reached": status == "target_reached",
            "locally_executed_consistent_count": reported_count,
            "attempts": [
                {"status": "construction_error", "error": "RAW_REQUEST_ERROR_SECRET"},
                {"status": "locally_executed_consistent"},
            ],
            "attempt_status_counts": {"construction_error": 1, "locally_executed_consistent": 1},
            "learner_evaluated": False,
            "official_benchmark_verifier": False,
            "selected_examples": [
                {
                    "path_id": "executed_path",
                    "title": "Executed synthetic example",
                    "status": "locally_executed_consistent",
                    "output_dir": "/PRIVATE/LOCAL/PATH",
                    "learner_evaluated": False,
                    "original_benchmark_reproduced": False,
                    "local_validation": {
                        "reference_query_executed": True,
                        "independent_query_executed": True,
                        "queries_agree": True,
                        "reference_result": "RAW_QUERY_RESULT_SECRET",
                    },
                }
            ],
        },
    )


def test_executed_task_receipts_and_models_are_separate_from_drafts(tmp_path):
    run, corpus, output = fixture(tmp_path)
    execution_fixture(run)
    contract = json.loads((run / "run_contract.json").read_text())
    contract.update(
        teacher_model="Qwen/teacher", teacher_revision="teacher-pin", teacher_reasoning=True
    )
    write(run, "run_contract.json", contract)
    audit = json.loads((run / "independent_audit_summary.json").read_text())
    audit["failed_requests"] = 2
    audit["independence"]["public_model"] = {
        "model": "Qwen/audit",
        "revision": "audit-pin",
        "api_key": "RAW_MODEL_SECRET",
    }
    write(run, "independent_audit_summary.json", audit)
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "complete"
    assert report["task_drafts"]["executed_task_count"] == 0
    assert report["executable_tasks"]["receipt_supported_consistent_count"] == 1
    assert report["executable_tasks"]["target_reached"] is True
    assert report["executable_tasks"]["construction_error_count"] == 1
    assert report["executable_tasks"]["failed_requests"] is None
    assert report["independent_audits"]["failed_requests"] == 2
    assert report["model_provenance"]["teacher"]["name"] == "Qwen/teacher"
    assert report["model_provenance"]["router"]["name"] == "Qwen/test"
    assert report["model_provenance"]["independent_audit"]["revision"] == "audit-pin"
    for name in ("README.md", "report.html", "report.json"):
        public = (output / name).read_text()
        assert "RAW_" not in public and "/PRIVATE/LOCAL/PATH" not in public
        assert "Executed synthetic example" in public


def test_executed_stage_exhaustion_is_complete_but_unreached_and_bad_receipts_are_partial(tmp_path):
    run, corpus, output = fixture(tmp_path)
    execution_fixture(run, status="supported_paths_exhausted")
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "complete"
    assert report["executable_tasks"]["target_reached"] is False
    assert any("exhausted" in warning for warning in report["warnings"])
    execution_fixture(run, reported_count=2)
    report = report_module.collect_report(run, corpus)
    assert report["status"] == "partial"
    assert not report["completion_checks"]["executable_receipts_consistent"]
    execution_fixture(run, status="running")
    report = report_module.collect_report(run, corpus)
    assert not report["completion_checks"]["executable_stage_finalized"]


def test_matrix_represents_every_state_pair_and_aggregates_exact_witness_counts():
    ids = ["A", "B<script>bad</script>", "C"]
    graph = {
        "nodes": [{"id": sid, "name": sid} for sid in ids],
        "edges": [
            {"source": ids[0], "target": ids[1], "witness_count": 3, "sampled_traversable": True},
            {"source": ids[0], "target": ids[1], "witness_count": 2, "sampled_traversable": False},
            {"source": ids[1], "target": ids[2], "witness_count": 7, "sampled_traversable": False},
        ],
    }
    rendered = report_module.adjacency_matrix(graph)
    root = ET.fromstring(rendered)
    assert root.attrib["data-n"] == "3"
    assert "<script>bad" not in rendered
    cells = {
        (element.attrib["data-i"], element.attrib["data-j"]): element.attrib
        for element in root.iter()
        if element.attrib.get("class") == "matrix-cell"
    }
    assert len(cells) == 2  # All nine pairs exist on the grid; zeroes use the blank background.
    assert cells["0", "1"]["data-all"] == "5"
    assert cells["0", "1"]["data-eligible"] == "3"
    assert cells["0", "1"]["data-edges"] == "2"
    assert cells["1", "2"]["data-all"] == "7"
    assert cells["1", "2"]["data-eligible"] == "0"
    labels = [
        element.attrib for element in root.iter() if element.attrib.get("class") == "matrix-node"
    ]
    assert len(labels) == 6
    assert {label["data-node"] for label in labels} == set(ids)
    background = next(
        element for element in root.iter() if element.attrib.get("class") == "matrix-background"
    )
    assert background.attrib["width"] == background.attrib["height"] == "42"


def test_matrix_interactions_use_fixed_safe_javascript_and_filterable_rows(tmp_path):
    run, corpus, output = fixture(tmp_path)
    report_module.build_report(run, corpus, output)
    page = (output / "report.html").read_text()
    for identifier in (
        "matrix-mode",
        "matrix-zoom",
        "matrix-clear",
        "matrix-focus",
        "matrix-detail",
    ):
        assert f'id="{identifier}"' in page
    assert 'data-source="A" data-target="A" data-sampled="0"' in page
    assert 'data-node="A"' in page
    assert "row.hidden = !show" in page
    assert "getScreenCTM" in page and "focusPair" in page
    assert "innerHTML" not in page and "fetch(" not in page


def add_comparison(run):
    result = json.loads((run / "gepa_result.json").read_text())
    seed_hash = hashlib.sha256(
        json.dumps(
            result["candidates"][0], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    heldout = json.loads((run / "heldout_test.json").read_text())
    heldout.update(
        evaluation_identity="selected-eval",
        rollouts=[{"rollout_id": "r", "feedback": "RAW_HELDOUT_SECRET"}],
    )
    write(run, "heldout_test.json", heldout)
    write(
        run,
        "baseline_heldout.json",
        {
            "candidate_hash": seed_hash,
            "evaluation_identity": "seed-eval",
            "rollouts": [{"rollout_id": "r", "transcript": "RAW_BASELINE_SECRET"}],
        },
    )
    comparison = {
        "format": "frozen-heldout-seed-selected-v1",
        "seed_candidate_hash": seed_hash,
        "selected_candidate_hash": heldout["candidate_hash"],
        "seed_evaluation_identity": "seed-eval",
        "selected_evaluation_identity": "selected-eval",
        "rollouts": 1,
        "original_tasks": 1,
        "common_rollout_ids": ["r"],
        "before_transductive_completion": True,
        "optimization_uses_test_feedback": False,
        "components": {
            "score": {
                "seed_mean": 0.4,
                "selected_mean": 0.7,
                "mean_difference": 0.3,
                "task_balanced_difference": 0.3,
                "paired_task_bootstrap_95ci": [0.1, 0.5],
                "private": "RAW_COMPONENT_SECRET",
            }
        },
        "bootstrap": {"draws": 10000, "resampling_units": 1, "private": "RAW_BOOTSTRAP_SECRET"},
        "per_task": {
            "task": {
                "rollouts": 1,
                "components": {
                    "score": {
                        "seed_mean": 0.4,
                        "selected_mean": 0.7,
                        "difference": 0.3,
                        "private": "RAW_TASK_SECRET",
                    }
                },
            }
        },
        "private": "RAW_COMPARISON_SECRET",
    }
    write(run, "heldout_comparison.json", comparison)
    return comparison


def test_paired_heldout_report_verifies_identity_and_omits_inference_evidence(tmp_path):
    run, corpus, output = fixture(tmp_path)
    add_comparison(run)
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "complete"
    comparison = report["heldout_comparison"]
    assert comparison["identities_and_census_verified"]
    assert comparison["components"]["score"]["paired_task_bootstrap_95ci"] == [0.1, 0.5]
    for name in ("README.md", "report.html", "report.json"):
        content = (output / name).read_text()
        assert "RAW_" not in content
    assert "0.1000 to 0.5000" in (output / "report.html").read_text()
    assert "10,000" in (output / "README.md").read_text()


def test_stale_or_different_rollout_comparison_is_not_presented_as_verified(tmp_path):
    run, corpus, output = fixture(tmp_path)
    comparison = add_comparison(run)
    comparison["common_rollout_ids"] = ["wrong"]
    write(run, "heldout_comparison.json", comparison)
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "partial"
    assert not report["heldout_comparison"]["identities_and_census_verified"]
    assert "0.1000 to 0.5000" not in (output / "report.html").read_text()


def cache_record(root, label, *, prompt=100, completion=5, attempts=None):
    key = hashlib.sha256(label.encode()).hexdigest()
    row = {
        "cache_key": key,
        "model": "Qwen/test",
        "model_revision": "pinned",
        "result": {"private": "RAW_CACHED_RESULT_SECRET"},
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        "finish_reason": "stop",
        "finished_at_epoch": 100,
        "api_key": "RAW_CACHE_SECRET",
        "messages": "RAW_REQUEST_SECRET",
    }
    if attempts is not None:
        row["attempt_outcomes"] = attempts
    name = f"llm_cache/{key[:2]}/{key}.json"
    write(root, name, row)
    return name, row


def test_usage_deduplicates_cache_keys_and_keeps_retry_and_process_views_separate(tmp_path):
    run, corpus, output = fixture(tmp_path)
    name, row = cache_record(
        run,
        "with-retry",
        attempts=[
            {"finish_reason": "length", "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            {"finish_reason": "stop", "usage": {"prompt_tokens": 100, "completion_tokens": 5}},
        ],
    )
    archive = run / "operational_restarts/first"
    write(archive, name, row)
    cache_record(run, "legacy-final-only", prompt=20, completion=2)
    write(
        archive,
        "llm_usage.json",
        {
            "requests": 99,
            "prompt_tokens": 9999,
            "completion_tokens": 999,
            "api_key": "RAW_USAGE_SECRET",
        },
    )
    write(run, "llm_usage.json", {"requests": 55, "prompt_tokens": 5555})
    completion = json.loads((run / "completion.json").read_text())
    completion["llm_usage"] = {
        "requests": 1,
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "cache_hits": 2,
        "retries": 0,
        "private": "RAW_COMPLETION_SECRET",
    }
    write(run, "completion.json", completion)
    report = report_module.build_report(run, corpus, output)
    usage = report["usage"]
    assert usage["unique_successful_request_keys"] == 2
    assert usage["duplicate_cache_copies_not_added"] == 1
    assert usage["final_responses"] == {
        "responses": 2,
        "prompt_tokens": 120,
        "completion_tokens": 7,
        "responses_missing_usage": 0,
    }
    assert usage["recorded_response_attempts"] == {
        "responses": 3,
        "prompt_tokens": 220,
        "completion_tokens": 17,
        "responses_missing_usage": 0,
    }
    assert usage["request_chains_with_attempt_ledger"] == 1
    assert usage["request_chains_with_final_response_only"] == 1
    assert len(usage["process_snapshots"]) == 2
    assert usage["process_snapshots"][0]["scope"] == "final_process"
    assert usage["process_snapshots"][0]["counters"]["requests"] == 1
    assert usage["process_snapshots"][1]["counters"]["requests"] == 99
    assert usage["actual_billing_available"] is False
    assert usage["by_model"][0]["final_responses"] == usage["final_responses"]
    for name in ("README.md", "report.html", "report.json"):
        content = (output / name).read_text()
        assert "RAW_" not in content
        assert "Inference usage accounting" in content or name == "report.json"


def test_usage_reports_conflicting_copies_and_missing_records_without_guessing_tokens(tmp_path):
    run = tmp_path / "run"
    name, row = cache_record(run, "conflict")
    row["finished_at_epoch"] = (
        200  # Could be a re-execution, so it cannot be summed as a copied request.
    )
    write(run / "initialization_rejected/older", name, row)
    _, unknown = cache_record(run, "unknown-usage", prompt=None, completion=None)
    invalid = run / "llm_cache/zz/invalid.json"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("RAW_INCOMPLETE_CACHE_SECRET")
    usage = report_module.collect_usage(run)
    assert usage["unique_successful_request_keys"] == 2
    assert usage["conflicting_request_keys_excluded_from_token_totals"] == 1
    assert usage["successful_requests_with_usable_metadata"] == 1
    assert usage["invalid_or_unreadable_cache_files"] == 1
    assert usage["final_responses"]["responses_missing_usage"] == 1
    assert usage["final_responses"]["prompt_tokens"] == 0
    assert usage["recorded_response_attempts"]["completion_tokens"] == 0
    assert "RAW_" not in json.dumps(usage)


def test_usage_separates_invalid_output_responses_from_transport_failures(tmp_path):
    run, corpus, output = fixture(tmp_path)
    name, row = cache_record(
        run,
        "invalid-output-transport-success",
        attempts=[
            {
                "attempt": 1,
                "finish_reason": "length",
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
            {"attempt": 2, "error_type": "APIConnectionError", "error": "RAW_TRANSPORT_SECRET"},
            {
                "attempt": 3,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            },
        ],
    )
    write(run / "operational_restarts/copy", name, row)
    report = report_module.build_report(run, corpus, output)
    usage = report["usage"]
    assert usage["unique_successful_request_keys"] == 1
    assert usage["duplicate_cache_copies_not_added"] == 1
    assert usage["final_responses"]["responses"] == 1
    assert usage["recorded_response_attempts"] == {
        "responses": 2,
        "prompt_tokens": 200,
        "completion_tokens": 15,
        "responses_missing_usage": 0,
    }
    assert usage["recorded_transport_failures"] == 1
    assert usage["by_model"][0]["recorded_transport_failures"] == 1
    assert usage["unclassified_attempt_entries_not_counted_as_responses"] == 0
    for filename in ("README.md", "report.html"):
        content = (output / filename).read_text()
        assert "Recorded transport failures in retained successful request chains: 1" in content
        assert "RAW_" not in content


def test_report_distinguishes_frozen_specification_from_reconstructed_graph(tmp_path):
    run, corpus, output = fixture(tmp_path)
    state = {
        "id": "A",
        "name": "Observed situation",
        "description": "Local facts",
        "exclusions": [],
    }
    seed = {
        "state_spec": json.dumps({"states": [state], "router_instructions": "Route"}),
        "edge_spec": json.dumps({"edges": []}),
    }
    selected = {
        "state_spec": seed["state_spec"],
        "edge_spec": json.dumps(
            {
                "edges": [
                    {
                        "id": f"selected{i}",
                        "source": "A",
                        "target": "A",
                        "operation": "Inspect",
                        "effect": "Facts",
                        "bindings": "Rename",
                    }
                    for i in range(2)
                ]
            }
        ),
    }
    write(run, "seed_candidate.json", seed)
    write(run, "optimized_candidate.json", selected)
    write(
        run,
        "gepa_result.json",
        {"candidates": [seed, selected], "val_aggregate_scores": [0.4, 0.8]},
    )
    heldout = json.loads((run / "heldout_test.json").read_text())
    heldout["candidate_hash"] = hashlib.sha256(
        json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write(run, "heldout_test.json", heldout)
    graph = json.loads((run / "graph.json").read_text())
    graph["nodes"].append({**state, "id": "B"})
    graph["routing_stages"] = [selected]
    graph["optimized_edge_spec"] = json.loads(selected["edge_spec"])["edges"]
    write(run, "graph.json", graph)
    report = report_module.build_report(run, corpus, output)
    assert report["status"] == "complete"
    stages = report["graph_stages"]
    assert stages["initial_seed"]["edges"] == 0
    assert stages["selected_frozen_specification"]["states"] == 1
    assert stages["selected_frozen_specification"]["edges"] == 2
    assert report["graph"]["node_count"] == 2 and report["graph"]["edge_count"] == 1
    assert stages["final_edges_reconstructed"] is True
    assert stages["pre_reconstruction_edge_count"] == 2
    for name in ("README.md", "report.html"):
        text = (output / name).read_text()
        assert "Selected frozen specification" in text
        assert "Final reconstructed witnessed graph" in text
        assert "not the later reconstructed witnessed graph" in text


def test_seed_edges_are_not_labeled_optimized(tmp_path):
    run = tmp_path / "run"
    write(
        run,
        "seed_candidate.json",
        {
            "state_spec": json.dumps(
                {
                    "states": [
                        {"id": "A", "name": "Seed", "description": "Facts", "exclusions": []}
                    ],
                    "router_instructions": "Route",
                }
            ),
            "edge_spec": json.dumps(
                {
                    "edges": [
                        {
                            "id": "E",
                            "source": "A",
                            "target": "A",
                            "operation": "Inspect",
                            "effect": "Facts",
                            "bindings": "Rename",
                        }
                    ]
                }
            ),
        },
    )
    report = report_module.collect_report(run, tmp_path / "corpus")
    assert report["graph"]["artifact"] == "seed_candidate.json"
    assert (
        report["graph"]["edges"][0]["evidence_status"]
        == "Initial seed specification; not yet audited"
    )
    assert not report["graph_stages"]["selected_frozen_specification"]["available"]


def test_audit_reporting_separates_missing_failures_and_non_error_unknowns(tmp_path):
    run, corpus, output = fixture(tmp_path)
    verdicts = {"supported": 1, "unknown": 3, "not_run": 1}
    write(
        run,
        "independent_audit_summary.json",
        {
            "planned_checks": 5,
            "completed_checks": 4,
            "failed_requests": 1,
            "by_kind": {"edge_source_applicability": verdicts},
        },
    )
    write(
        run,
        "independent_audit_jobs.json",
        [
            {
                "audit_id": f"audit{i}",
                "kind": "edge_source_applicability",
                "edge_id": "E",
                "history_ids": [f"r:h{i:04d}"],
            }
            for i in range(5)
        ],
    )
    write(
        run,
        "independent_audit_results.json",
        [
            {
                "audit_id": f"audit{i}",
                "kind": "edge_source_applicability",
                "verdict": "supported" if i == 0 else "unknown",
                "execution_status": "error" if i == 1 else "completed",
                "private": "RAW_AUDIT_DETAILS_SECRET",
            }
            for i in range(4)
        ],
    )
    graph = json.loads((run / "graph.json").read_text())
    graph["edges"][0]["audit"] = {
        "verdicts": verdicts,
        "source_member_count": 6,
        "independent_sampled_source_ids": [f"r:h{i:04d}" for i in range(4)],
    }
    write(run, "graph.json", graph)
    report = report_module.build_report(run, corpus, output)
    audit = report["independent_audits"]
    assert audit["result_records_including_request_errors"] == 4
    assert audit["checks_without_request_errors"] == 3
    assert audit["checks_without_result_records"] == 1
    assert audit["by_kind_execution"]["edge_source_applicability"] == {
        "request_errors": 1,
        "non_error_unknown": 2,
        "not_run": 1,
    }
    edge = report["graph"]["edges"][0]
    assert edge["source_histories_with_audit_records"] == 4
    assert edge["source_member_count"] == 6
    assert edge["source_audit_request_errors"] == 1
    assert edge["source_audit_non_error_unknown"] == 2
    for name in ("README.md", "report.html", "report.json"):
        assert "RAW_" not in (output / name).read_text()
    assert "non-error unknown: 2" in (output / "report.html").read_text()


def continuation_receipt():
    return {
        "boundary_after_proposal": 7,
        "reason": "Attach the producing transition when explaining membership at a history.",
        "old_graph_evolution_sha256": "a" * 64,
        "new_graph_evolution_sha256": "b" * 64,
        "unchanged_evaluation_identity": True,
        "evidence_only_repair": True,
        "archived_checkpoint_sha256": "c" * 64,
        "resumed_at_utc": "2026-09-17T15:00:00Z",
    }


def test_optimizer_continuation_is_explicit_allowlisted_method_disclosure(tmp_path):
    run, corpus, output = fixture(tmp_path)
    receipt = continuation_receipt()
    write(run, "optimizer_continuations.json", [{**receipt, "private": "RAW_CONTINUATION_SECRET"}])
    report = report_module.build_report(run, corpus, output)
    continuation = report["optimizer_continuations"]
    assert report["status"] == "complete"
    assert continuation["validation_status"] == "valid"
    assert continuation["records"] == [receipt]
    assert "not independently established" in continuation["validation_scope"]
    for name in ("README.md", "report.html"):
        text = (output / name).read_text()
        assert "After proposal 7" in text
        assert "unchanged evaluation identity: Yes" in text
        assert "evidence-only repair: Yes" in text
        assert "RAW_" not in text
    assert "RAW_" not in (output / "report.json").read_text()


def test_absent_continuation_receipt_does_not_claim_an_unchanged_run(tmp_path):
    run, corpus, output = fixture(tmp_path)
    report = report_module.build_report(run, corpus, output)
    assert report["optimizer_continuations"]["validation_status"] == "not_declared"
    assert report["optimizer_continuations"]["records"] == []
    assert (
        "does not establish an unchanged optimization run" in (output / "report.html").read_text()
    )


def test_invalid_continuation_fields_make_provenance_incomplete_without_leaking_values(tmp_path):
    run, corpus, output = fixture(tmp_path)
    for field, invalid in (
        ("boundary_after_proposal", True),
        ("reason", " "),
        ("old_graph_evolution_sha256", "RAW_INVALID_HASH_SECRET"),
        ("unchanged_evaluation_identity", "true"),
        ("evidence_only_repair", None),
        ("resumed_at_utc", "2026-09-17T15:00:00"),
    ):
        write(run, "optimizer_continuations.json", [{**continuation_receipt(), field: invalid}])
        report = report_module.collect_report(run, corpus)
        assert report["status"] == "partial"
        assert not report["completion_checks"]["optimizer_continuation_receipts"]
        assert report["optimizer_continuations"]["records"] == []
        assert field in report["optimizer_continuations"]["validation_errors"][0]
        assert "RAW_" not in json.dumps(report)


def test_continuation_false_invariant_flags_are_not_rewritten_as_true(tmp_path):
    run, corpus, output = fixture(tmp_path)
    receipt = {
        **continuation_receipt(),
        "unchanged_evaluation_identity": False,
        "evidence_only_repair": False,
    }
    write(run, "optimizer_continuations.json", [receipt])
    report_module.build_report(run, corpus, output)
    text = (output / "report.html").read_text()
    assert "unchanged evaluation identity: No" in text
    assert "evidence-only repair: No" in text
