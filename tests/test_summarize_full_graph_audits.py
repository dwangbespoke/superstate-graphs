"""Audit accounting tests use synthetic saved records and never rejudge evidence."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


diagnostics = module("audit_diagnostics", ROOT / "scripts/summarize_full_graph_audits.py")
fixtures = module(
    "audit_diagnostics_fixture",
    Path(__file__).with_name("test_prepare_exploratory_task_examples.py"),
)


def fixture(tmp_path):
    run, _ = fixtures.fixture(tmp_path)
    results = fixtures.read(run / "independent_audit_results.json")
    for row in results:
        row["citation_errors"] = [
            "Quote is not an exact prefix substring: PRIVATE_HISTORY_ID",
            "Quote is not an exact prefix substring: OTHER_PRIVATE_HISTORY_ID",
            "Definitive verdict must cite each tested source history",
        ]
    jobs = fixtures.read(run / "independent_audit_jobs.json")
    for index, (raw, validated, error) in enumerate(
        (
            ("supported", "supported", False),
            ("contradicted", "unknown", False),
            ("unknown", "unknown", False),
            (None, "unknown", True),
        )
    ):
        job = {"audit_id": f"coherence{index}", "kind": "within_state_coherence"}
        jobs.append(job)
        row = {**job, "verdict": validated}
        if error:
            row.update(execution_status="error", error="PRIVATE_REQUEST_ERROR")
        else:
            row.update(
                original_verdict=raw,
                schema_and_citations_checked=True,
                citation_errors=["Missing or out-of-scope citation"] if raw != validated else [],
            )
        results.append(row)
    fixtures.write(run / "independent_audit_jobs.json", jobs)
    fixtures.write(run / "independent_audit_results.json", results)
    fixtures.refresh_audits(run)
    return run


def test_raw_validated_failure_and_citation_categories_remain_distinct(tmp_path):
    run = fixture(tmp_path)
    before = {path.name: diagnostics.sha(path) for path in run.iterdir()}
    report = diagnostics.summarize(run)
    assert report["planned_checks"] == report["result_records"] == 6
    assert report["request_failure_records"] == 1
    coherence = report["by_kind"]["within_state_coherence"]
    assert coherence["normal_response_records"] == 3
    assert coherence["validated_verdict_counts"] == {"supported": 1, "unknown": 3}
    assert coherence["definitive_verdicts_downgraded_by_citation_errors"] == 1
    assert {tuple(row.values()) for row in coherence["raw_to_validated"]} == {
        ("supported", "supported", 1),
        ("contradicted", "unknown", 1),
        ("unknown", "unknown", 1),
        ("unavailable_request_failure", "unknown", 1),
    }
    edge = report["by_kind"]["edge_source_applicability"]
    assert edge["citation_error_categories"]["quote_not_exact_prefix_substring"] == {
        "affected_results": 2,
        "error_occurrences": 4,
    }
    assert report["edge_flag_counts_nonexclusive"]["all_raw_source_checks_supported"] == 2
    assert report["edge_flag_counts_nonexclusive"]["all_validated_source_checks_supported"] == 0
    assert report["edge_flag_counts_nonexclusive"]["sampled_supported_eligible"] == 0
    markdown = diagnostics.markdown(report)
    assert "PRIVATE_" not in str(report) + markdown
    assert "INDEPENDENT_AUDIT_REVIEW.md" in markdown
    assert "not semantic accuracy" in markdown
    assert before == {path.name: diagnostics.sha(path) for path in run.iterdir()}


@pytest.mark.parametrize(
    "change", ["missing_result", "aggregate_count", "invalid_downgrade", "flag"]
)
def test_incomplete_or_inconsistent_final_artifacts_are_rejected(tmp_path, change):
    run = fixture(tmp_path)
    if change == "aggregate_count":
        summary = fixtures.read(run / "independent_audit_summary.json")
        summary["completed_checks"] -= 1
        fixtures.write(run / "independent_audit_summary.json", summary)
    elif change == "flag":
        graph = fixtures.read(run / "graph.json")
        graph["edges"][0]["traversable"] = True
        fixtures.write(run / "graph.json", graph)
    else:
        results = fixtures.read(run / "independent_audit_results.json")
        if change == "missing_result":
            results.pop()
        else:
            results[0]["original_verdict"] = "supported"
            results[0]["citation_errors"] = []
        fixtures.write(run / "independent_audit_results.json", results)
    with pytest.raises(ValueError):
        diagnostics.summarize(run)
