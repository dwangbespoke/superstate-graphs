"""Tiny synthetic databases test implementation, not research effectiveness."""

import copy
import json
import subprocess
import sys

import duckdb
import pytest

from superstate_graphs.task_constructor import (
    TaskConstructionError,
    construction_prompt,
    construct_task,
    database_context,
    normalize_path,
    validate_candidate,
    verify_submission,
)


@pytest.fixture
def warehouse(tmp_path):
    db = tmp_path / "test.duckdb"
    connection = duckdb.connect(str(db))
    connection.execute("CREATE TABLE customers(customer_id INTEGER, country VARCHAR)")
    connection.execute("INSERT INTO customers VALUES (1, 'US'), (2, 'UK'), (3, 'US')")
    connection.execute("CREATE TABLE orders(customer_id INTEGER, amount DECIMAL(10,2))")
    connection.execute("INSERT INTO orders VALUES (1, 20), (1, 30), (2, 10), (3, 5)")
    connection.close()
    return db


@pytest.fixture
def path_spec():
    return {
        "path_id": "test_cross_task_path",
        "target_superstate": "aggregate_grain_established",
        "transitions": [
            {"transition_id": "edge_0", "source_task_id": "customer_totals",
             "source_history_id": "a_before", "target_history_id": "a_after",
             "operation": "Aggregate orders to one row per customer",
             "witness": {"observed_sql": "SELECT customer_id, SUM(amount) FROM orders GROUP BY 1"}},
            {"transition_id": "edge_1", "source_task_id": "regional_cohort",
             "source_history_id": "b_before", "target_history_id": "b_after",
             "operation": "Join a customer cohort to country metadata and filter it",
             "witness": {"observed_sql": "SELECT * FROM customers WHERE country = 'US'"}},
        ],
    }


@pytest.fixture
def candidate():
    return {
        "task_id": "high_spend_countries", "title": "High-spend revenue by country",
        "instruction": "Find countries' revenue from customers with total orders >= 25. "
                       "Return country and revenue, with no particular ordering.",
        "prerequisites": ["orders and customers are available"],
        "novel_objective": "Compose customer aggregation with geography instead of source outputs",
        "target_recreation": "Choose the customer grain before the geographic join",
        "target_recreation_limitations": ["No source diagnostic history is replayed"],
        "stages": [
            {"name": "sg_customer_totals", "purpose": "Establish customer grain",
             "source_transition_ids": ["edge_0"],
             "sql": "SELECT customer_id, SUM(amount) AS revenue FROM orders GROUP BY customer_id"},
            {"name": "sg_cohort", "purpose": "Bind qualifying customers to geography",
             "source_transition_ids": ["edge_1"],
             "sql": "SELECT c.country, t.revenue FROM sg_customer_totals t "
                    "JOIN customers c USING (customer_id) WHERE t.revenue >= 25"},
        ],
        "final_sql": "SELECT country, SUM(revenue) AS revenue FROM sg_cohort GROUP BY country",
        "output_columns": ["country", "revenue"], "ordered": False,
        "constraints": [
            {"description": "Every country total comes from a qualifying cohort",
             "sql": "SELECT bool_and(revenue >= 25) FROM sg_result"},
            {"description": "Country names are non-null and unique",
             "sql": "SELECT count(*) = count(DISTINCT country) FROM sg_result"},
        ],
        "alternate_sql": "SELECT country, SUM(revenue) AS revenue FROM "
                         "(SELECT c.customer_id, c.country, SUM(o.amount) AS revenue "
                         "FROM customers c JOIN orders o USING (customer_id) "
                         "GROUP BY c.customer_id, c.country HAVING SUM(o.amount) >= 25) x "
                         "GROUP BY country",
    }


def test_constructed_task_and_standalone_verifier(warehouse, path_spec, candidate, tmp_path):
    output = tmp_path / "constructed"
    result = construct_task(path_spec, lambda prompt: candidate, warehouse, output)
    assert result["report"]["oracle_verifier_check"]["reward"] == 1
    assert result["report"]["result"]["rows"] == [["US", "50.00"]]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["source_task_ids"] == ["customer_totals", "regional_cohort"]
    assert provenance["path"]["transitions"][0]["witness"]
    assert "not independent" in result["report"]["limitations"][0]
    command = subprocess.run(
        [sys.executable, str(output / "verifier.py"), "--db", str(warehouse),
         "--submission", str(output / "oracle.sql")],
        capture_output=True, text=True, check=False,
    )
    assert command.returncode == 0, command.stderr
    assert json.loads(command.stdout)["reward"] == 1
    with pytest.raises(ValueError, match="differs from oracle"):
        verify_submission(output, warehouse, "SELECT 'US' AS country, 999 AS revenue")


def test_execution_feedback_repairs_candidate(warehouse, path_spec, candidate, tmp_path):
    bad = copy.deepcopy(candidate)
    bad["alternate_sql"] = "SELECT 'US' AS country, 123 AS revenue"
    prompts = []
    def llm(prompt):
        prompts.append(prompt)
        return bad if len(prompts) == 1 else candidate
    result = construct_task(path_spec, llm, warehouse, tmp_path / "repaired")
    assert len(prompts) == 2
    assert "EXECUTION/VALIDATION FAILURE" in prompts[1]
    assert "missing row" in prompts[1]
    assert result["report"]["construction_attempts"] == 2


def test_read_only_and_nonvacuous_validation(warehouse, path_spec, candidate):
    bad = copy.deepcopy(candidate)
    bad["stages"][0]["sql"] = "DELETE FROM orders RETURNING *"
    with pytest.raises(ValueError, match="read-only"):
        validate_candidate(bad, path_spec, warehouse)
    connection = duckdb.connect(str(warehouse), read_only=True)
    assert connection.execute("SELECT count(*) FROM orders").fetchone()[0] == 4
    connection.close()
    bad = copy.deepcopy(candidate)
    bad["stages"][1]["sql"] += " AND FALSE"
    with pytest.raises(ValueError, match="empty"):
        validate_candidate(bad, path_spec, warehouse)
    bad = copy.deepcopy(candidate)
    bad["constraints"][0]["sql"] = "SELECT count(*) < 0 FROM sg_result"
    with pytest.raises(ValueError, match="TRUE boolean"):
        validate_candidate(bad, path_spec, warehouse)


def test_witness_provenance_and_stage_dependency_are_required(warehouse, path_spec, candidate):
    bad_path = copy.deepcopy(path_spec)
    bad_path["transitions"][1]["source_task_id"] = "customer_totals"
    with pytest.raises(ValueError, match="two source tasks"):
        normalize_path(bad_path)
    bad = copy.deepcopy(candidate)
    bad["stages"][1]["sql"] = "SELECT country, 50 AS revenue FROM customers"
    with pytest.raises(ValueError, match="consume previous"):
        validate_candidate(bad, path_spec, warehouse)
    bad = copy.deepcopy(candidate)
    bad["stages"][1]["source_transition_ids"] = ["invented_edge"]
    with pytest.raises(ValueError, match="valid source transition"):
        validate_candidate(bad, path_spec, warehouse)


def test_failed_attempts_are_saved(warehouse, path_spec, tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(TaskConstructionError, match="No valid task after 1"):
        construct_task(path_spec, lambda prompt: "invalid JSON", warehouse, output, max_repairs=0)
    attempts = json.loads((output / "construction_attempts.json").read_text())
    assert attempts[0]["status"] == "rejected"
    assert attempts[0]["response"] == "invalid JSON"


def test_schema_retrieval_is_bounded_and_uses_real_identifiers(tmp_path):
    db = tmp_path / "large_catalog.duckdb"
    connection = duckdb.connect(str(db))
    connection.execute("CREATE SCHEMA SALES")
    connection.execute("CREATE TABLE SALES.ORDERS(order_id INTEGER, amount INTEGER)")
    connection.execute("INSERT INTO SALES.ORDERS VALUES (1, 20)")
    connection.execute("CREATE VIEW orders AS SELECT * FROM SALES.ORDERS")
    for index in range(60):
        connection.execute(f"CREATE VIEW unrelated_{index} AS SELECT 1 AS x")
    connection.close()
    context = database_context(
        db, path_spec={"operation": 'SELECT amount FROM "SALES"."ORDERS" JOIN invented_table USING (order_id)'},
        max_tables=4, max_chars=4000,
    )
    assert context["retrieval"]["catalog_relation_count"] == 62
    assert context["retrieval"]["selected_relation_count"] <= 4
    assert context["retrieval"]["sampled_relation_count"] <= 4
    assert context["tables"][0]["qualified_name"] == "SALES.ORDERS"
    assert "invented_table" in context["retrieval"]["unresolved_sql_references"]
    assert "invented_table" not in {table["table"] for table in context["tables"]}
    rendered = json.dumps(context, indent=2, sort_keys=True)
    assert len(rendered) <= 4000
    assert context["retrieval"]["context_characters"] == len(rendered)


def test_large_catalog_without_relevant_evidence_requires_references(tmp_path):
    db = tmp_path / "unrelated.duckdb"
    connection = duckdb.connect(str(db))
    for index in range(41):
        connection.execute(f"CREATE VIEW unrelated_{index} AS SELECT 1 AS x")
    connection.close()
    with pytest.raises(ValueError, match="No path-related table"):
        database_context(db, path_spec={"operation": "inspect available resources"})


def test_prompt_evidence_is_bounded_without_losing_source_identity(path_spec):
    path_spec["transitions"][0]["witness"]["long_observation"] = "large observation " * 10_000
    path_spec = normalize_path(path_spec)
    prompt = construction_prompt(path_spec, {"engine": "DuckDB", "tables": []})
    assert len(prompt) < 60_000
    for transition in path_spec["transitions"]:
        for field in ("source_task_id", "source_history_id", "target_history_id", "transition_id"):
            assert transition[field] in prompt
    assert "full original is retained in provenance.json" in prompt
    with pytest.raises(ValueError, match="strict 10000-character budget"):
        construction_prompt(path_spec, {"large_context": "x" * 12_000}, max_chars=10_000)
