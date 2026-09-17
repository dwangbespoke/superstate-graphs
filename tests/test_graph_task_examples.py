import asyncio
import json
from copy import deepcopy

import pytest

from superstate_graphs.graph_task_examples import (
    compare_query_results,
    construct_executable_example,
    run_local_validation,
    validate_fixture_spec,
    verify_submission,
)


def spec():
    return {
        "status": "task", "title": "Regional order totals", "learner_instruction":
        "Find each customer's region and return region,total for all recorded orders.",
        "output_columns": ["region", "total"], "ordered_output": False,
        "path_transition_ids": ["inspect", "aggregate"],
        "tables": [
            {"name": "customers", "columns": [{"name": "customer_id", "type": "INTEGER"},
                                               {"name": "region", "type": "VARCHAR"}],
             "rows": [[1, "North"], [2, "South"], [3, "North"]]},
            {"name": "orders", "columns": [{"name": "customer_id", "type": "INTEGER"},
                                            {"name": "amount", "type": "INTEGER"}],
             "rows": [[1, 10], [1, 20], [2, 5], [3, 7]]},
        ],
        "reference_sql": "SELECT c.region, SUM(o.amount) AS total FROM customers c "
                         "JOIN orders o ON o.customer_id=c.customer_id GROUP BY c.region",
    }


INDEPENDENT = (
    "SELECT region, SUM(amount) AS total FROM "
    "(SELECT amount, (SELECT region FROM customers c WHERE c.customer_id=o.customer_id) "
    "AS region FROM orders o) GROUP BY region"
)


def test_actual_subprocess_sql_execution_and_standalone_verifier(tmp_path):
    task = spec()
    report = run_local_validation(task, INDEPENDENT, tmp_path)
    assert report["status"] == "locally_executed_consistent"
    assert report["reference_query_executed"] and report["independent_query_executed"]
    assert sorted(report["reference_result"]["rows"]) == [["North", 37], ["South", 5]]
    assert report["original_benchmark_reproduced"] is False
    assert report["learner_evaluated"] is False
    (tmp_path / "task.json").write_text(json.dumps(task))
    verified = verify_submission(tmp_path, task["reference_sql"])
    assert verified["status"] == "passed"
    wrong = verify_submission(tmp_path, "SELECT region, 0 AS total FROM customers")
    assert wrong["status"] == "incorrect_result"


def test_two_queries_disagree_without_becoming_validated(tmp_path):
    report = run_local_validation(spec(), "SELECT region, 0 AS total FROM customers", tmp_path)
    assert report["status"] == "consistency_failed"
    assert report["queries_agree"] is False
    assert not (tmp_path / "fixture.duckdb").exists()


def test_select_only_rejects_destructive_and_multistatement_sql(tmp_path):
    for query in ("DROP TABLE customers", "SELECT 1; SELECT 2", "COPY customers TO '/tmp/nope'"):
        task = spec()
        task["reference_sql"] = query
        with pytest.raises(ValueError, match="SELECT"):
            run_local_validation(task, INDEPENDENT, tmp_path)


def test_external_file_access_is_disabled_inside_the_worker(tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("region,total\nDO_NOT_READ,999\n")
    report = run_local_validation(spec(), f"SELECT * FROM read_csv('{secret}')", tmp_path / "task")
    assert report["status"] == "execution_failed"
    assert report["reference_query_executed"] is True
    assert report["independent_query_executed"] is False
    assert "DO_NOT_READ" not in json.dumps(report)


def test_worker_timeout_is_reported_without_claiming_execution(tmp_path):
    report = run_local_validation(spec(), INDEPENDENT, tmp_path, timeout_seconds=.00001)
    assert report["status"] == "execution_timeout"
    assert not report.get("reference_query_executed", False)


def test_fixture_identifiers_types_and_row_provenance_are_checked():
    malicious = deepcopy(spec())
    malicious["tables"][0]["name"] = "x; DROP TABLE orders"
    with pytest.raises(ValueError, match="identifier"):
        validate_fixture_spec(malicious)
    unsupported_type = deepcopy(spec())
    unsupported_type["tables"][0]["columns"][0]["type"] = "INTEGER); COPY x TO 'path'"
    with pytest.raises(ValueError, match="column type"):
        validate_fixture_spec(unsupported_type)
    with pytest.raises(ValueError, match="path provenance"):
        validate_fixture_spec(spec(), {"transitions": [{"transition_id": "other"}]})


def test_unordered_result_comparison_preserves_duplicate_multiplicity_and_types():
    left = {"columns": ["x"], "rows": [[1], [1], [2]]}
    assert compare_query_results(left, {"columns": ["x"], "rows": [[2], [1.0], [1]]})
    assert not compare_query_results(left, {"columns": ["x"], "rows": [[2], [2], [1]]})
    assert not compare_query_results({"columns": ["x"], "rows": [[True]]},
                                     {"columns": ["x"], "rows": [[1]]})
    assert not compare_query_results(left, {"columns": ["x"], "rows": [[2], [1], [1]]},
                                     ordered=True)


def test_generator_and_blinded_solver_are_schema_constrained_and_execute(tmp_path):
    class FakeModel:
        runtime = {"model": "fake-model", "revision": "r1", "api_key": "NEVER_PERSIST_SECRET"}
        calls = 0

        def shard(self, key):
            return self

        async def complete_json(self, messages, **kwargs):
            self.calls += 1
            assert kwargs.get("schema")
            assert kwargs.get("thinking") is True
            assert "historical/task material has ended" in messages[-1]["content"]
            if messages[0]["content"].startswith("Create a SMALL"):
                response = spec()
                response["reference_sql"] += " /* unique_oracle_text */"
                return response
            content = json.loads(messages[1]["content"])
            assert "reference_sql" not in content["task"]
            assert "unique_oracle_text" not in messages[1]["content"]
            return {"verdict": "faithful", "reason": "Synthetic test response",
                    "independent_sql": INDEPENDENT, "findings": []}

    path = {"path_id": "test-path", "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "transitions": [{"transition_id": "inspect", "source_history_id": "h0",
                             "target_history_id": "h1", "operation": "inspect"},
                            {"transition_id": "aggregate", "source_history_id": "h2",
                             "target_history_id": "h3", "operation": "aggregate"}]}
    prefixes = {"h0": "initial", "h1": "initial observed result", "h2": "second initial",
                "h3": "second initial observed result"}
    model = FakeModel()
    result = asyncio.run(construct_executable_example(path, prefixes.__getitem__, model, tmp_path))
    assert result["status"] == "locally_executed_consistent"
    assert result["executed"]
    assert len(result["fixture_sha256"]) == 64
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "instruction.md").exists()
    assert result["original_benchmark_reproduced"] is False
    assert "NEVER_PERSIST_SECRET" not in (tmp_path / "result.json").read_text()
    assert model.calls == 2
    repeated = asyncio.run(construct_executable_example(path, prefixes.__getitem__, model, tmp_path))
    assert repeated["cache_reused"] is True
    assert model.calls == 2
    (tmp_path / "instruction.md").unlink()
    repaired = asyncio.run(construct_executable_example(path, prefixes.__getitem__, model, tmp_path))
    assert repaired["cache_reused"] is False
    assert repaired["status"] == "locally_executed_consistent"
    assert model.calls == 4
    model.runtime = {**model.runtime, "revision": "r2"}
    revised = asyncio.run(construct_executable_example(path, prefixes.__getitem__, model, tmp_path))
    assert revised["request_key"] != result["request_key"]
    assert model.calls == 6
    prefixes["h3"] += " NEW ACTUAL OBSERVATION"
    changed_prefix = asyncio.run(construct_executable_example(path, prefixes.__getitem__, model, tmp_path))
    assert changed_prefix["request_key"] != revised["request_key"]
    assert model.calls == 8
