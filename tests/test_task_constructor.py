"""Tiny synthetic databases test implementation, not research effectiveness."""

import copy
import json
import subprocess
import sys

import duckdb
import pytest

from superstate_graphs.task_constructor import (
    TaskConstructionError,
    _prompt_path,
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
        "target_definition": {
            "id": "aggregate_grain_established",
            "description": "Choose aggregate grain before adding geographic dimensions",
            "roles": ["entity", "metric", "dimension"],
            "requirements": ["Source grain and intended output grain are known"],
        },
        "junction_prefix_A": {
            "decision": "Choose aggregation before joining customer metadata",
            "remaining_goal": "Compute eligible-customer revenue",
            "known_facts": ["Orders can repeat customer IDs"],
            "unresolved_questions": ["Whether aggregation should precede the join"],
            "possible_operations": ["Aggregate first", "Join first"],
        },
        "junction_prefix_B": {
            "decision": "Choose customer grain before geographic reporting",
            "remaining_goal": "Report the eligible cohort by country",
            "known_facts": ["Geographic metadata is one row per customer"],
            "unresolved_questions": ["Which grain preserves the customer threshold"],
            "possible_operations": ["Filter at customer grain", "Filter at country grain"],
        },
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
    assert result["status"] == "validated"
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


def test_learned_decision_evidence_precedes_generic_metadata(path_spec):
    path_spec["generic_statistics"] = {"large_unrelated_statistic": "X" * 80_000}
    path_spec["junction_prefix_A"]["evidence"] = "Long raw observation " * 10_000
    path_spec["junction_prefix_B"]["evidence"] = "Another observation " * 10_000
    path_spec["transitions"][0]["witness"]["long_observation"] = "Transition observation " * 10_000
    for side in ("junction_prefix_A", "junction_prefix_B"):
        path_spec[side].update({
            "task_requirements": ["[message 0] Output column must be sales_id"],
            "learner_beliefs": ["[message 3] Learner believes sale_key is missing"],
            "unresolved_conflicts": ["[messages 0, 2, 3] Required alias differs from learner belief"],
            "source_evidence": {
                "task_requirement_excerpt": {"message_index": 0, "source_kind": "task_requirement",
                                             "text": "Required sales_id column. " * 100, "clipped": True},
                "recent_observations": [{"message_index": 2, "source_kind": "terminal_transcript",
                                         "text": "Observed schema output. " * 100, "clipped": True}],
                "explicit_mapping_and_schema_lines": [{
                    "message_index": 2, "source_kind": "terminal_transcript",
                    "pattern": "explicit_sql_alias", "matched_text": "sale_key AS sales_id",
                    "text": "Long context " * 100 + "sale_key AS sales_id", "clipped": True,
                }],
                "selection_policy": "Literal prefix-only snippets",
                "interpretation": "Snippets do not themselves assert semantic facts",
            },
        })
    projected = _prompt_path(path_spec)
    assert projected["target_definition"] == path_spec["target_definition"]
    for side in ("junction_prefix_A", "junction_prefix_B"):
        for field in ("decision", "remaining_goal", "task_requirements", "known_facts",
                      "learner_beliefs", "unresolved_conflicts", "unresolved_questions", "possible_operations"):
            assert projected[side][field] == path_spec[side][field]
        assert projected[side]["evidence"]["note"].startswith("Excerpt only")
        source = projected[side]["source_evidence"]
        assert source["task_requirement_excerpt"]["source_kind"] == "task_requirement"
        assert source["recent_observations"][0]["source_kind"] == "terminal_transcript"
        mapping = source["explicit_mapping_and_schema_lines"][0]
        assert mapping["matched_text"] == "sale_key AS sales_id"
        assert mapping["message_index"] == 2
        assert mapping["constructor_text_clipped"] is True
    assert len(json.dumps(projected, indent=2, sort_keys=True)) <= 18_000
    too_large = copy.deepcopy(path_spec)
    too_large["target_definition"]["requirements"] = ["Core decision requirement " * 1000]
    with pytest.raises(ValueError, match="without losing decision meaning"):
        _prompt_path(too_large)


def test_unsupported_target_returns_receipt_without_task_files(warehouse, path_spec, tmp_path):
    output = tmp_path / "unsupported"
    response = {
        "status": "unsupported_target",
        "targeted_decision": "Identify the active dbt profile before diagnosing model execution",
        "reason": "A SELECT over retail tables cannot recreate choosing project profiles",
        "missing_capabilities": ["Mutable dbt project files and command execution"],
        "required_task_format": "A dbt project task with conflicting profile configurations",
    }
    calls = []
    def llm(prompt):
        calls.append(prompt)
        return response
    result = construct_task(path_spec, llm, warehouse, output)
    assert result["status"] == "unsupported_target"
    assert result["task_id"] is None
    assert len(calls) == 1
    assert "shared topic" in calls[0]
    assert result["report"]["task_files_emitted"] is False
    assert not (output / "instruction.md").exists()
    assert not (output / "oracle.sql").exists()
    assert not (output / "verifier.py").exists()
    assert json.loads((output / "provenance.json").read_text())["path"] == path_spec
    assert json.loads((output / "construction_attempts.json").read_text())[0]["status"] == "unsupported_target"


def test_missing_decision_evidence_cannot_claim_successful_construction(warehouse, path_spec, candidate):
    del path_spec["target_definition"]
    with pytest.raises(ValueError, match="learned target_definition"):
        validate_candidate(candidate, path_spec, warehouse)
