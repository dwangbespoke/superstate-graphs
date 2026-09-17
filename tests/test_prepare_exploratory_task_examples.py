"""Offline synthetic selection tests; these are not experimental task outcomes."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from superstate_graphs.graph_analysis import _edge_view, _node_view, aggregate_independent_audits


PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_exploratory_task_examples.py"
SPEC = importlib.util.spec_from_file_location("exploratory_task_experiment", PATH)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_text())


def refresh_audits(run):
    graph = read(run / "graph.json")
    jobs = read(run / "independent_audit_jobs.json")
    results = read(run / "independent_audit_results.json")
    for edge in graph["edges"]:
        edge["traversable"] = False
    summary = aggregate_independent_audits(jobs, results, graph)
    summary["failed_requests"] = sum(row.get("execution_status") == "error" for row in results)
    for edge, row in zip(graph["edges"], summary["edges"]):
        edge["audit"] = {**row, "source_member_count": 6}
        edge["traversable"] = (
            bool(row["verdicts"].get("supported"))
            and not any(row["verdicts"].get(key) for key in ("unknown", "contradicted", "not_run"))
            and edge["proposal"]["universal_source_plausible"]
        )
    write(run / "graph.json", graph)
    write(run / "independent_audit_summary.json", summary)


def fixture(tmp_path):
    run, corpus = tmp_path / "run", tmp_path / "corpus"
    run.mkdir()
    corpus.mkdir()
    histories, transitions, assignments = [], [], {}
    for r in range(2):
        for step in range(3):
            hid = f"r{r}:h{step:04d}"
            h = {"history_id": hid, "rollout_id": f"r{r}", "task_id": f"task{r}", "step": step}
            histories.append(h)
            assignments[hid] = {**h, "state_id": "S", "evidence": "RAW_PREFIX_PRIVATE"}
            if step < 2:
                transitions.append(
                    {
                        "transition_id": f"r{r}:t{step:04d}",
                        "source_history_id": hid,
                        "target_history_id": f"r{r}:h{step + 1:04d}",
                        "rollout_id": f"r{r}",
                        "task_id": f"task{r}",
                        "step": step,
                    }
                )
    for name, rows in (
        ("histories.jsonl", histories),
        ("transitions.jsonl", transitions),
        ("rollouts.jsonl", [{"id": "r0"}, {"id": "r1"}]),
    ):
        (corpus / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(
        corpus / "manifest.json",
        {
            "artifact_sha256": {
                name: experiment.fingerprint(corpus / name)
                for name in ("histories.jsonl", "transitions.jsonl", "rollouts.jsonl")
            }
        },
    )
    node = {"id": "S", "description": "Synthetic local situation", "exclusions": []}
    edges, jobs, results = [], [], []
    for r, eid in enumerate(("E", "F")):
        edge = {
            "id": eid,
            "source": "S",
            "target": "S",
            "operation": "Inspect",
            "effect": "Observation",
            "bindings": "Rename fixture",
            "traversable": False,
            "proposal": {"universal_source_plausible": True},
            "witnesses": [],
        }
        for t in transitions[r * 2 : r * 2 + 2]:
            edge["witnesses"].append(
                {
                    "transition_id": t["transition_id"],
                    "source_id": t["source_history_id"],
                    "target_id": t["target_history_id"],
                    "rollout_id": t["rollout_id"],
                    "task_id": t["task_id"],
                    "step": t["step"],
                }
            )
        edges.append(edge)
        jobs.append(
            {
                "audit_id": "audit_" + eid,
                "kind": "edge_source_applicability",
                "edge_id": eid,
                "state_ids": ["S", "S"],
                "history_ids": [f"r{r}:h0000"],
                "target_history_ids": ["r1:h0002"],
                "source_member_count": 6,
                "source_observed_to_target": True,
            }
        )
        results.append(
            {
                "audit_id": "audit_" + eid,
                "kind": "edge_source_applicability",
                "verdict": "unknown",
                "original_verdict": "supported",
                "schema_and_citations_checked": True,
                "citation_errors": ["Synthetic invalid quote"],
                "verified_evidence": [],
            }
        )
    write(run / "completion.json", {"status": "complete"})
    write(run / "graph.json", {"nodes": [node], "edges": edges})
    write(run / "assignments_all.json", assignments)
    write(run / "independent_audit_jobs.json", jobs)
    write(run / "independent_audit_results.json", results)
    refresh_audits(run)
    path = {
        "path_id": "saved_path",
        "state_ids": ["S"] * 3,
        "nodes": [_node_view(node)] * 3,
        "transitions": [
            {
                **_edge_view(edge),
                "transition_id": edge["id"],
                "source_history_id": f"r{r}:h0000",
                "target_history_id": f"r{r}:h0001",
            }
            for r, edge in enumerate(edges)
        ],
    }
    write(run / "task_path_proposals.json", [path])
    write(
        run / "task_draft_summary.json",
        {"attempts": [{"path_id": "saved_path", "review_verdict": "plausible"}]},
    )
    write(
        run / "executable_task_summary.json",
        {
            "status": "supported_paths_exhausted",
            "attempts": [],
            "locally_executed_consistent_count": 0,
        },
    )
    return run, corpus


def test_unknown_selection_is_explicit_and_does_not_change_primary_artifacts(tmp_path):
    run, corpus = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    plan = experiment.select_paths(run, corpus)
    assert plan["original_supported_edges"] == 0
    assert plan["qualifying_exploratory_edges"] == 2
    assert len(plan["selected_paths"]) == 1
    selected = plan["selected_paths"][0]
    assert selected["selection_origin"] == "saved_path"
    for rows in selected["edge_audit_evidence"].values():
        assert rows[0]["original_verdict"] == "supported"
        assert rows[0]["validated_verdict"] == "unknown"
        assert rows[0]["applicability_label"] == "unresolved_applicability_due_to_citation_errors"
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}
    assert "RAW_PREFIX_PRIVATE" not in json.dumps(experiment.public_plan(plan))


@pytest.mark.parametrize(
    "failure",
    [
        "raw_contradiction",
        "validated_contradiction",
        "proposer_rejection",
        "transport_error",
        "missing_audit",
        "missing_provenance",
        "invalid_witness",
    ],
)
def test_contradicted_rejected_or_missing_evidence_cannot_enter(tmp_path, failure):
    run, corpus = fixture(tmp_path)
    results = read(run / "independent_audit_results.json")
    graph = read(run / "graph.json")
    if failure == "raw_contradiction":
        results[0]["original_verdict"] = "contradicted"
    elif failure == "validated_contradiction":
        results[0].update(
            verdict="contradicted",
            original_verdict="contradicted",
            citation_errors=[],
            verified_evidence=[{"history_id": "r0:h0000", "quote": "Synthetic checked quote"}],
        )
    elif failure == "proposer_rejection":
        graph["edges"][0]["proposal"]["universal_source_plausible"] = False
    elif failure == "transport_error":
        results[0]["execution_status"] = "error"
    elif failure == "missing_audit":
        results.pop(0)
    elif failure == "missing_provenance":
        results[0].pop("original_verdict")
    elif failure == "invalid_witness":
        graph["edges"][0]["witnesses"][0]["transition_id"] = "absent_transition"
    write(run / "graph.json", graph)
    write(run / "independent_audit_results.json", results)
    refresh_audits(run)
    assert experiment.select_paths(run, corpus)["selected_paths"] == []


def test_resampling_is_deterministic_and_preserves_unknown_flags(tmp_path):
    run, corpus = fixture(tmp_path)
    write(run / "task_path_proposals.json", [])
    graph_before = (run / "graph.json").read_bytes()
    assert experiment.select_paths(run, corpus)["selected_paths"] == []
    a = experiment.select_paths(run, corpus, resample_filtered_paths=True)
    b = experiment.select_paths(run, corpus, resample_filtered_paths=True)
    assert a == b
    assert a["selected_paths"]
    assert all(
        item["selection_origin"] == "resampled_filtered_observed_graph"
        and item["resampling_seed"] == 20260919
        for item in a["selected_paths"]
    )
    assert (run / "graph.json").read_bytes() == graph_before


def test_saved_plausible_paths_precede_resampled_paths_and_attempts_are_skipped(tmp_path):
    run, corpus = fixture(tmp_path)
    plan = experiment.select_paths(run, corpus, resample_filtered_paths=True)
    assert plan["selected_paths"][0]["path"]["path_id"] == "saved_path"
    primary = read(run / "executable_task_summary.json")
    primary["attempts"] = [{"path_id": "saved_path", "status": "unsupported"}]
    write(run / "executable_task_summary.json", primary)
    plan = experiment.select_paths(run, corpus, resample_filtered_paths=True)
    assert all(
        [step["transition_id"] for step in item["path"]["transitions"]] != ["E", "F"]
        for item in plan["selected_paths"]
    )


@pytest.mark.parametrize("later_plausible_review", [False, True])
def test_rejected_draft_sequence_cannot_return_with_alternate_witnesses(
    tmp_path, later_plausible_review
):
    run, corpus = fixture(tmp_path)
    reviews = [{"path_id": "saved_path", "review_verdict": "contradicted"}]
    if later_plausible_review:
        reviews.append({"path_id": "saved_path", "review_verdict": "plausible"})
    write(
        run / "task_draft_summary.json",
        {"attempts": reviews},
    )
    plan = experiment.select_paths(run, corpus, resample_filtered_paths=True)
    signatures = {
        tuple(step["transition_id"] for step in item["path"]["transitions"])
        for item in plan["selected_paths"]
    }
    assert ("E", "F") not in signatures
    assert ("F", "E") in signatures
    assert plan["excluded_path_reason_counts"]["draft_review_contradicted"] == 1
    assert (
        plan["excluded_path_reason_counts"]["edge_sequence_rejected_by_primary_draft_review"]
        >= 1
    )


def test_completion_and_immutable_source_guards(tmp_path):
    run, corpus = fixture(tmp_path)
    plan = experiment.select_paths(run, corpus)
    write(run / "completion.json", {"status": "running"})
    with pytest.raises(ValueError, match="not complete"):
        experiment.select_paths(run, corpus)
    with pytest.raises(ValueError, match="changed after"):
        experiment.verify_snapshot(plan, run, corpus)


def test_dry_run_does_not_open_runtime_or_output(tmp_path, monkeypatch, capsys):
    run, corpus = fixture(tmp_path)
    output = tmp_path / "never_created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PATH),
            "--run",
            str(run),
            "--corpus",
            str(corpus),
            "--runtime",
            str(tmp_path / "nonexistent-private-runtime"),
            "--output",
            str(output),
        ],
    )
    experiment.main()
    result = json.loads(capsys.readouterr().out)
    assert len(result["selected_paths"]) == 1
    assert not output.exists()


def test_checkpoint_resume_is_receipt_bound_and_credential_free(tmp_path, monkeypatch):
    run, corpus = fixture(tmp_path)
    plan = experiment.select_paths(run, corpus)
    output, calls = tmp_path / "supplementary", []

    async def constructor(path, provider, llm, directory, **kwargs):
        calls.append(path["path_id"])
        directory.mkdir(parents=True)
        write(directory / "source_path.json", path)
        result = {
            "path_id": path["path_id"],
            "status": "unsupported",
            "error": "SECRET_NOT_FOR_PUBLIC_SUMMARY",
            "artifact_sha256": {
                "source_path.json": experiment.fingerprint(directory / "source_path.json")
            },
        }
        write(directory / "result.json", result)
        return result

    monkeypatch.setattr(experiment, "construct_executable_example", constructor)
    llm = SimpleNamespace(runtime={"model": "synthetic/offline", "api_key": "SECRET_API_KEY"})
    for _ in range(2):
        summary = asyncio.run(experiment.execute_plan(plan, None, llm, output))
    assert calls == ["saved_path"]
    assert summary["status"] == "exploratory_paths_exhausted"
    assert summary["locally_executed_consistent_count"] == 0
    assert "SECRET" not in json.dumps(summary)
    summary["attempts"][0]["locally_executed_consistent"] = True
    write(output / "exploratory_task_summary.json", summary)
    with pytest.raises(ValueError, match="differs from its original receipt"):
        asyncio.run(experiment.execute_plan(plan, None, llm, output))


def test_success_requires_faithfulness_and_actual_execution_receipts(tmp_path):
    result = {
        "path_id": "synthetic",
        "status": "locally_executed_consistent",
        "review_verdict": "unresolved",
    }
    with pytest.raises(ValueError, match="faithful-review and execution"):
        experiment.safe_attempt(result, tmp_path)
    result.update(
        review_verdict="faithful",
        local_validation={
            "reference_query_executed": True,
            "independent_query_executed": True,
            "queries_agree": True,
        },
    )
    with pytest.raises(ValueError, match="missing required artifacts"):
        experiment.safe_attempt(result, tmp_path)


def test_source_mutation_stops_before_constructor_call(tmp_path, monkeypatch):
    run, corpus = fixture(tmp_path)
    plan = experiment.select_paths(run, corpus)
    (run / "completion.json").write_text("{}")

    async def unexpected_constructor(*args, **kwargs):
        raise AssertionError("No model-backed constructor call is allowed")

    monkeypatch.setattr(experiment, "construct_executable_example", unexpected_constructor)
    with pytest.raises(ValueError, match="changed after"):
        asyncio.run(
            experiment.execute_plan(
                plan,
                None,
                SimpleNamespace(runtime={"model": "synthetic/offline"}),
                tmp_path / "supplementary",
                source_guard=lambda: experiment.verify_snapshot(plan, run, corpus),
            )
        )
