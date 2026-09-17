"""Synthetic publisher fixtures: local SQL execution only, never model inference."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from superstate_graphs import graph_task_examples


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


publisher = module("supplementary_publisher", ROOT / "scripts/publish_exploratory_task_examples.py")
selection_tests = module(
    "supplementary_selection_fixture",
    Path(__file__).with_name("test_prepare_exploratory_task_examples.py"),
)
write, read = selection_tests.write, selection_tests.read


def finalized(tmp_path, monkeypatch, *, success=True):
    run, corpus = selection_tests.fixture(tmp_path)
    experiment, output = tmp_path / "experiment", tmp_path / "public" / "exploratory-tasks"
    plan = publisher.experiment_code.select_paths(run, corpus)

    async def construct(path, prefix, llm, directory, **kwargs):
        directory.mkdir(parents=True)
        write(directory / "source_path.json", path)
        write(directory / "generator_response.json", {"reason": "PRIVATE_GENERATOR_TEXT"})
        if success:
            spec = {
                "status": "task",
                "title": "Synthetic sum",
                "path_id": path["path_id"],
                "learner_instruction": "Return the sum of n as total from numbers.",
                "output_columns": ["total"],
                "ordered_output": False,
                "path_transition_ids": [step["transition_id"] for step in path["transitions"]],
                "tables": [
                    {
                        "name": "numbers",
                        "columns": [{"name": "n", "type": "INTEGER"}],
                        "rows": [[1], [2]],
                    }
                ],
                "reference_sql": "SELECT SUM(n) AS total FROM numbers",
                "private_comment": "PRIVATE_TASK_METADATA",
            }
            independent = "SELECT SUM(n + 0) AS total FROM numbers"
            write(directory / "task.json", spec)
            write(
                directory / "independent_solution_review.json",
                {
                    "verdict": "faithful",
                    "independent_sql": independent,
                    "reason": "PRIVATE_REVIEW_PREFIX_QUOTE",
                },
            )
            (directory / "instruction.md").write_text(spec["learner_instruction"] + "\n")
            (directory / "oracle.sql").write_text(spec["reference_sql"] + "\n")
            (directory / "independent.sql").write_text(independent + "\n")
            validation = graph_task_examples.run_local_validation(spec, independent, directory)
            assert validation["status"] == "locally_executed_consistent"
            (directory / "README.md").write_text("PRIVATE_ABSOLUTE_SOURCE_PATH")
            result = {
                "status": validation["status"],
                "review_verdict": "faithful",
                "local_validation": validation,
            }
        else:
            write(directory / "task.json", {"status": "unsupported", "reason": "PRIVATE_RAW_ERROR"})
            result = {"status": "unsupported", "reason": "PRIVATE_RAW_ERROR"}
        result.update(
            {
                "path_id": path["path_id"],
                "public_model": llm.runtime,
                "artifact_sha256": {
                    name: publisher.fingerprint(directory / name)
                    for name in publisher.experiment_code.ARTIFACT_NAMES
                    if (directory / name).is_file()
                },
            }
        )
        write(directory / "result.json", result)
        return result

    monkeypatch.setattr(publisher.experiment_code, "construct_executable_example", construct)
    asyncio.run(
        publisher.experiment_code.execute_plan(
            plan,
            None,
            SimpleNamespace(runtime={"model": "synthetic/offline"}),
            experiment,
            target=1,
        )
    )
    return run, corpus, experiment, output


def test_successful_export_is_separate_safe_self_contained_and_hashed(tmp_path, monkeypatch):
    run, corpus, experiment, output = finalized(tmp_path, monkeypatch)
    source_before = {
        str(p): publisher.fingerprint(p)
        for root in (run, corpus, experiment)
        for p in root.rglob("*")
        if p.is_file()
    }
    result = publisher.publish(experiment, run, corpus, output)
    assert result["examples"] == 1
    summary = read(output / "summary.json")
    assert summary["evidence_label"] == publisher.LABEL
    assert summary["primary_graph_and_counts_unchanged"] is True
    example = output / summary["examples"][0]["directory"]
    provenance = read(example / "provenance.json")
    assert [w["witness_transition_id"] for w in provenance["witnesses"]] == ["r0:t0000", "r1:t0000"]
    audit = provenance["edge_audit_evidence"]["E"][0]
    assert (audit["original_verdict"], audit["validated_verdict"]) == ("supported", "unknown")
    assert audit["applicability_label"] == "unresolved_applicability_due_to_citation_errors"
    verified = graph_task_examples.verify_submission(example, (example / "oracle.sql").read_text())
    assert verified["status"] == "passed"
    manifest = read(output / "manifest.json")
    assert manifest["publication_privacy_review_required"] is True
    for name, info in manifest["files"].items():
        assert publisher.fingerprint(output / name) == info["sha256"]
    for path in output.rglob("*"):
        if path.is_file():
            assert b"PRIVATE_" not in path.read_bytes()
    assert not (example / "generator_response.json").exists()
    assert "[instruction.md](instruction.md)" in (example / "README.md").read_text()
    assert source_before == {
        str(p): publisher.fingerprint(p)
        for root in (run, corpus, experiment)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_zero_successes_still_exports_honest_failed_attempt_counts(tmp_path, monkeypatch):
    run, corpus, experiment, output = finalized(tmp_path, monkeypatch, success=False)
    result = publisher.publish(experiment, run, corpus, output)
    assert result["examples"] == 0 and result["attempts"] == 1
    summary = read(output / "summary.json")
    assert summary["status"] == "exploratory_paths_exhausted"
    assert summary["attempt_status_counts"] == {"unsupported": 1}
    assert summary["target_reached"] is False
    assert not (output / "examples").exists()
    assert b"PRIVATE_RAW_ERROR" not in (output / "summary.json").read_bytes()


@pytest.mark.parametrize(
    "mutation", ["unfinalized", "selection", "result", "artifact", "primary_source"]
)
def test_corrupted_or_unfinished_sources_fail_closed(tmp_path, monkeypatch, mutation):
    run, corpus, experiment, output = finalized(tmp_path, monkeypatch, success=False)
    if mutation == "unfinalized":
        value = read(experiment / "exploratory_task_summary.json")
        value["status"] = "running"
        write(experiment / "exploratory_task_summary.json", value)
    elif mutation == "selection":
        value = read(experiment / "selection.json")
        value["selected_paths"][0]["edge_audit_evidence"]["E"][0]["validated_verdict"] = "supported"
        write(experiment / "selection.json", value)
    elif mutation == "result":
        value = read(experiment / "examples/saved_path/result.json")
        value["status"] = "construction_error"
        write(experiment / "examples/saved_path/result.json", value)
    elif mutation == "artifact":
        (experiment / "examples/saved_path/task.json").write_text("{}")
    elif mutation == "primary_source":
        (run / "completion.json").write_text('{"status":"complete", "unexpected":true}')
    with pytest.raises(ValueError):
        publisher.publish(experiment, run, corpus, output)
    assert not output.exists()


def rewrite_success_receipts(experiment, changed_names):
    directory = experiment / "examples/saved_path"
    result = read(directory / "result.json")
    for name in changed_names:
        result["artifact_sha256"][name] = publisher.fingerprint(directory / name)
    write(directory / "result.json", result)
    summary = read(experiment / "exploratory_task_summary.json")
    receipt = publisher.experiment_code.safe_attempt(result, directory)
    summary["attempts"] = summary["selected_examples"] = [receipt]
    write(experiment / "exploratory_task_summary.json", summary)


def test_success_requires_review_agreement_even_if_hash_receipts_are_rewritten(
    tmp_path, monkeypatch
):
    run, corpus, experiment, output = finalized(tmp_path, monkeypatch)
    review_path = experiment / "examples/saved_path/independent_solution_review.json"
    review = read(review_path)
    review["verdict"] = "unclear"
    write(review_path, review)
    rewrite_success_receipts(experiment, ["independent_solution_review.json"])
    with pytest.raises(ValueError, match="review/execution receipts"):
        publisher.publish(experiment, run, corpus, output)
    assert not output.exists()


def test_recognizable_credentials_in_allowlisted_instruction_block_publication(
    tmp_path, monkeypatch
):
    run, corpus, experiment, output = finalized(tmp_path, monkeypatch)
    directory = experiment / "examples/saved_path"
    spec = read(directory / "task.json")
    spec["learner_instruction"] = "password=synthetic_secret_blocked"
    write(directory / "task.json", spec)
    (directory / "instruction.md").write_text(spec["learner_instruction"] + "\n")
    rewrite_success_receipts(experiment, ["task.json", "instruction.md"])
    with pytest.raises(ValueError, match="credential"):
        publisher.publish(experiment, run, corpus, output)
    assert not output.exists()


def test_decoded_fixture_cells_are_scanned_even_with_updated_file_receipts(tmp_path, monkeypatch):
    run, corpus, experiment, output = finalized(tmp_path, monkeypatch)
    database = graph_task_examples._connect(
        experiment / "examples/saved_path/fixture.duckdb", read_only=False
    )
    try:
        database.execute("ALTER TABLE numbers ALTER COLUMN n TYPE VARCHAR")
        database.execute("UPDATE numbers SET n='password=synthetic_cell_secret' WHERE n='1'")
    finally:
        database.close()
    rewrite_success_receipts(experiment, ["fixture.duckdb"])
    with pytest.raises(ValueError, match="synthetic fixture cell values"):
        publisher.publish(experiment, run, corpus, output)
    assert not output.exists()
