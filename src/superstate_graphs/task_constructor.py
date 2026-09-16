"""Turn witnessed cross-task graph paths into executable, read-only SQL tasks.

The constructor is deliberately a small callable interface: ``construct_task``
accepts a finalized path, a JSON-producing LLM callable, and a real DuckDB file.
It does not collect rollouts, select graph paths, or make model calls itself.
Generated checks and a second SQL formulation are useful consistency checks,
not an independent demonstration that an LLM-authored task is correct.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import decimal
import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Callable

import duckdb

FORMAT_VERSION = "superstate-sql-task-v1"
NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
LIMITATIONS = [
    "The oracle, alternate query, and constraints share a generator; agreement is not independent correctness evidence.",
    "Provenance identifies source witnesses, but semantic preservation of their decision situations still needs review.",
    "These are read-only SQL workflow tasks, not full reproductions of source dbt environments or prefix knowledge states.",
    "No learner difficulty, high continuation variance, or training benefit is established by construction checks.",
]


class TaskConstructionError(ValueError):
    """A candidate could not be made executable and consistent within its budget."""


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False)


def _cell(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (list, dict, tuple)):
        return json.loads(json.dumps(value, default=str))
    return value


def _connect(db_path: str | Path) -> Any:
    return duckdb.connect(str(db_path), read_only=True, config={
        "enable_external_access": "false", "threads": "2", "memory_limit": "1GB",
    })


def _select(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a nonempty string")
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1 or str(statements[0].type).split(".")[-1] != "SELECT":
        raise ValueError("Only one read-only SELECT or WITH query is allowed")
    return sql.strip().rstrip(";").strip()


def _query(connection: Any, sql: str, *, max_rows: int = 1000,
           timeout_seconds: float = 30) -> dict[str, Any]:
    sql = _select(sql)
    timer = threading.Timer(timeout_seconds, connection.interrupt)
    timer.daemon = True
    timer.start()
    try:
        cursor = connection.execute(sql)
        columns = [column[0] for column in cursor.description]
        types = [str(column[1]) for column in cursor.description]
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError(f"Query returned more than {max_rows} rows; aggregate the final output")
        if len(set(columns)) != len(columns):
            raise ValueError("Result columns must have distinct names")
        return {"columns": columns, "types": types,
                "rows": [[_cell(cell) for cell in row] for row in rows]}
    finally:
        timer.cancel()


def _qualify(schema: str, table: str) -> str:
    return '"' + schema.replace('"', '""') + '"."' + table.replace('"', '""') + '"'


def database_context(db_path: str | Path, *, sample_rows: int = 2) -> dict[str, Any]:
    """Read schema and a few real values; no test or reference-solution input."""
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT table_schema, table_name, column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') "
            "ORDER BY table_schema, table_name, ordinal_position"
        ).fetchall()
        tables: dict[tuple[str, str], list[dict[str, str]]] = {}
        for schema, table, column, dtype in rows:
            tables.setdefault((schema, table), []).append({"name": column, "type": dtype})
        context = []
        for (schema, table), columns in tables.items():
            sample = _query(connection, f"SELECT * FROM {_qualify(schema, table)} LIMIT {int(sample_rows)}")
            sample["rows"] = [[cell[:200] if isinstance(cell, str) else cell for cell in row]
                              for row in sample["rows"]]
            context.append({"schema": schema, "table": table, "columns": columns,
                            "sample_rows": sample["rows"]})
        return {"engine": "DuckDB", "tables": context}
    finally:
        connection.close()


def normalize_path(path_spec: dict[str, Any]) -> dict[str, Any]:
    """Require concrete witnesses; preserve the caller's complete provenance."""
    result = json.loads(json.dumps(path_spec, default=str))
    transitions = result.get("transitions", [])
    if len(transitions) < 2:
        raise ValueError("A composed path must contain at least two transitions")
    for index, transition in enumerate(transitions):
        transition.setdefault("transition_id", f"edge_{index}")
        for key in ("source_task_id", "source_history_id", "target_history_id", "operation"):
            if not transition.get(key):
                raise ValueError(f"Transition {index} requires {key}")
    ids = [transition["transition_id"] for transition in transitions]
    if len(set(ids)) != len(ids):
        raise ValueError("Transition IDs must be unique")
    if len({transition["source_task_id"] for transition in transitions}) < 2:
        raise ValueError("Task construction requires witnesses from at least two source tasks")
    if not result.get("target_superstate"):
        raise ValueError("path_spec requires target_superstate")
    result.setdefault("path_id", hashlib.sha256(_json(transitions).encode()).hexdigest()[:12])
    return result


def construction_prompt(path_spec: dict[str, Any], context: dict[str, Any]) -> str:
    return """Construct ONE new executable analytical task in this shared DuckDB warehouse.
The supplied path is a cross-task combination of actual learner transition witnesses.
Reuse their meaningful operation patterns in a coherent workflow with a NEW terminal
objective. A paraphrase of a source instruction is insufficient. Preserve the targeted
decision's prerequisites and ambiguity where possible; do not reveal the solution in
the instruction. Do not claim to preserve a learner's knowledge state just because
SQL executes. Explain any mismatch in target_recreation_limitations.

Return JSON only. Required schema:
{
  "task_id": "short_lowercase_slug",
  "title": "...",
  "instruction": "Complete standalone problem, exact definitions, time bounds, ties,
    null handling, required output columns and ordering. Ask for a single SELECT query.",
  "prerequisites": ["concrete facts/available tables needed"],
  "novel_objective": "How the combined objective differs from both source objectives",
  "target_recreation": "Which targeted decision is required and where",
  "target_recreation_limitations": ["What source context is not reproduced"],
  "stages": [
    {"name": "sg_stage_1", "purpose": "...", "source_transition_ids": ["edge_0"],
     "sql": "SELECT ... FROM actual_source_table"},
    {"name": "sg_stage_2", "purpose": "...", "source_transition_ids": ["edge_1"],
     "sql": "SELECT ... FROM sg_stage_1 ..."}
  ],
  "final_sql": "SELECT ... FROM sg_stage_2 ...",
  "output_columns": ["..."],
  "ordered": false,
  "constraints": [
    {"description": "a semantic invariant of the requested result",
     "sql": "SELECT ... AS valid FROM sg_result"},
    {"description": "another nontrivial semantic invariant",
     "sql": "SELECT ... AS valid FROM sg_result"}
  ],
  "alternate_sql": "A separately formulated SELECT directly over warehouse tables,
     producing the same final columns/results without using sg_stage_* or sg_result"
}

Use at least two stages, citing transitions from at least two DIFFERENT source tasks.
Every stage after the first must consume the immediately previous stage's CTE; the
final SQL must consume the last stage. Keep the final result between 1 and 1000 rows.
Stage names start with sg_ and use lowercase letters, digits and underscores.
SQL may only SELECT/WITH; no writes, extensions, files, PRAGMAs, or external access.
Every intermediate stage must return at least one row. All intermediate stages are
named CTEs visible to subsequent stages. Constraints see the stage CTEs plus sg_result
and each must return exactly one TRUE boolean, inspecting sg_result nontrivially.
Avoid unbounded raw joins; aggregate intermediate data at a specified grain.
The alternate query is a self-consistency check, not independent oracle evidence.

PATH AND ACTUAL WITNESSES:
""" + _json(path_spec) + "\nWAREHOUSE SCHEMA AND REAL SAMPLE VALUES:\n" + _json(context)


def _parse_candidate(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value).strip()
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Constructor must return a JSON object")
    return value


def _references(sql: str, name: str) -> bool:
    # Supplement syntactic checks with actual execution; this is not lineage proof.
    return bool(re.search(r"\b" + re.escape(name) + r"\b", sql, re.IGNORECASE))


def _ctes(stages: list[dict[str, Any]], extra: tuple[str, str] | None = None) -> str:
    terms = [(stage["name"], stage["sql"]) for stage in stages]
    if extra:
        terms.append(extra)
    return "WITH " + ",\n".join(f"{name} AS (\n{_select(sql)}\n)" for name, sql in terms) + "\n"


def _numeric(value: Any) -> decimal.Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = decimal.Decimal(str(value))
        return number if number.is_finite() else None
    except decimal.InvalidOperation:
        return None


def _same_cell(left: Any, right: Any) -> bool:
    if left == right:
        return True
    a, b = _numeric(left), _numeric(right)
    if a is not None and b is not None:
        return abs(a - b) <= max(decimal.Decimal("0.000001"),
                                 max(abs(a), abs(b)) * decimal.Decimal("0.000001"))
    return False


def compare_results(expected: dict[str, Any], actual: dict[str, Any], *, ordered: bool) -> None:
    if expected["columns"] != actual["columns"]:
        raise ValueError(f"Column mismatch: expected {expected['columns']}, got {actual['columns']}")
    if len(expected["rows"]) != len(actual["rows"]):
        raise ValueError(f"Row count mismatch: expected {len(expected['rows'])}, got {len(actual['rows'])}")
    def same_row(a: list[Any], b: list[Any]) -> bool:
        return len(a) == len(b) and all(_same_cell(x, y) for x, y in zip(a, b, strict=True))
    if ordered:
        if not all(same_row(a, b) for a, b in zip(expected["rows"], actual["rows"], strict=True)):
            raise ValueError("Ordered result differs from oracle")
    else:
        # Small aggregate result sets permit a tolerance-aware bag comparison.
        remaining = list(actual["rows"])
        for row in expected["rows"]:
            match = next((i for i, candidate in enumerate(remaining) if same_row(row, candidate)), None)
            if match is None:
                raise ValueError(f"Result differs from oracle; missing row: {row}")
            remaining.pop(match)


def validate_candidate(candidate: dict[str, Any], path_spec: dict[str, Any],
                       db_path: str | Path) -> dict[str, Any]:
    """Execute the whole composition, alternate formulation, and generated checks."""
    for field in ("task_id", "title", "instruction", "novel_objective", "target_recreation"):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            raise ValueError(f"Missing nonempty candidate field: {field}")
    if not NAME.fullmatch(candidate["task_id"]):
        raise ValueError("task_id must be a lowercase alphanumeric/underscore slug")
    if not isinstance(candidate.get("ordered"), bool):
        raise ValueError("ordered must be a boolean")
    stages = candidate.get("stages", [])
    if len(stages) < 2:
        raise ValueError("At least two dependent intermediate stages are required")
    transition_tasks = {edge["transition_id"]: edge["source_task_id"]
                        for edge in path_spec["transitions"]}
    used_tasks: set[str] = set()
    names: set[str] = set()
    for index, stage in enumerate(stages):
        name = stage.get("name", "")
        if not NAME.fullmatch(name) or not name.startswith("sg_") or name == "sg_result" or name in names:
            raise ValueError("Stage names must be unique sg_ identifiers other than sg_result")
        names.add(name)
        _select(stage.get("sql", ""))
        sources = stage.get("source_transition_ids", [])
        if not sources or any(source not in transition_tasks for source in sources):
            raise ValueError(f"Stage {name} lacks valid source transition IDs")
        used_tasks.update(transition_tasks[source] for source in sources)
        if index and not _references(stage["sql"], stages[index - 1]["name"]):
            raise ValueError(f"Stage {name} must consume previous stage {stages[index - 1]['name']}")
    if len(used_tasks) < 2:
        raise ValueError("Stages must use witnesses from at least two different source tasks")
    final_sql = _select(candidate.get("final_sql", ""))
    if not _references(final_sql, stages[-1]["name"]):
        raise ValueError("Final SQL must consume the final stage")
    alternate_sql = _select(candidate.get("alternate_sql", ""))
    if any(_references(alternate_sql, name) for name in names | {"sg_result"}):
        raise ValueError("Alternate SQL must be formulated directly over warehouse tables")
    constraints = candidate.get("constraints", [])
    if len(constraints) < 2:
        raise ValueError("At least two result constraints are required")
    for constraint in constraints:
        sql = _select(constraint.get("sql", ""))
        if not _references(sql, "sg_result") or not constraint.get("description"):
            raise ValueError("Each described constraint must inspect sg_result")
    connection = _connect(db_path)
    stage_reports, check_reports = [], []
    try:
        for index, stage in enumerate(stages):
            prefix = _ctes(stages[:index]) if index else ""
            # Do not retain huge intermediate results; inspect count and a small sample.
            query = prefix + f"SELECT count(*) AS row_count FROM ({_select(stage['sql'])}) AS stage_output"
            row_count = _query(connection, query)["rows"][0][0]
            if row_count == 0:
                raise ValueError(f"Stage {stage['name']} is empty")
            sample = _query(connection, prefix +
                            f"SELECT * FROM ({_select(stage['sql'])}) AS stage_output LIMIT 3")
            stage_reports.append({"name": stage["name"], "row_count": row_count,
                                  "sample": sample, "source_transition_ids": stage["source_transition_ids"]})
        result = _query(connection, _ctes(stages) + final_sql)
        if not result["rows"]:
            raise ValueError("Final query must have a nonempty result")
        if candidate.get("output_columns") != result["columns"]:
            raise ValueError(f"output_columns must match executed result: {result['columns']}")
        alternate = _query(connection, alternate_sql)
        compare_results(result, alternate, ordered=candidate["ordered"])
        for constraint in constraints:
            checked = _query(connection, _ctes(stages, ("sg_result", final_sql)) + constraint["sql"])
            if len(checked["columns"]) != 1 or checked["rows"] != [[True]] or checked["types"] != ["BOOLEAN"]:
                raise ValueError(f"Constraint did not return one TRUE boolean: {constraint['description']}; {checked}")
            check_reports.append({"description": constraint["description"], "passed": True})
        return {"status": "validated", "stage_reports": stage_reports, "result": result,
                "constraint_reports": check_reports, "alternate_agrees": True,
                "witness_source_tasks": sorted(used_tasks), "limitations": LIMITATIONS}
    finally:
        connection.close()


def verify_submission(task_dir: str | Path, db_path: str | Path,
                      submission_sql: str) -> dict[str, Any]:
    """Compare a solver's SELECT to the frozen oracle result on this warehouse."""
    task_dir = Path(task_dir)
    spec = json.loads((task_dir / "task.json").read_text())
    expected = json.loads((task_dir / "expected_result.json").read_text())
    connection = _connect(db_path)
    try:
        actual = _query(connection, submission_sql)
        compare_results(expected, actual, ordered=spec["ordered"])
        for constraint in spec["constraints"]:
            sql = _ctes(spec["stages"], ("sg_result", _select(submission_sql))) + constraint["sql"]
            checked = _query(connection, sql)
            if checked["rows"] != [[True]] or checked["types"] != ["BOOLEAN"]:
                raise ValueError(f"Submission fails constraint: {constraint['description']}")
        return {"passed": True, "reward": 1, "rows": len(actual["rows"]),
                "grading_scope": "Frozen oracle equivalence and generator-authored constraints"}
    finally:
        connection.close()


def _write_task(output: Path, candidate: dict[str, Any], path_spec: dict[str, Any],
                report: dict[str, Any], attempts: list[dict[str, Any]], db_path: Path) -> None:
    oracle = _ctes(candidate["stages"]) + _select(candidate["final_sql"]) + ";\n"
    instruction = candidate["instruction"].strip() + (
        "\n\nSubmit your solution as a single read-only DuckDB SELECT/WITH query in `answer.sql`. "
        "The evaluator runs it against the supplied warehouse.\n"
    )
    receipt = {"database_file_name": db_path.name, "database_bytes": db_path.stat().st_size,
               "database_mtime_ns": db_path.stat().st_mtime_ns,
               "identity_note": "Filesystem receipt only; source dataset manifest should supply a content hash."}
    provenance = {"format_version": FORMAT_VERSION, "path": path_spec,
                  "source_task_ids": report["witness_source_tasks"], "database": receipt,
                  "stage_bindings": [{key: stage[key] for key in ("name", "purpose", "source_transition_ids")}
                                     for stage in candidate["stages"]],
                  "novel_objective": candidate["novel_objective"],
                  "target_recreation": candidate["target_recreation"],
                  "target_recreation_limitations": candidate.get("target_recreation_limitations", [])}
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "instruction.md": instruction, "oracle.sql": oracle,
        "alternate.sql": _select(candidate["alternate_sql"]) + ";\n",
        "task.json": _json(candidate) + "\n", "provenance.json": _json(provenance) + "\n",
        "expected_result.json": _json(report["result"]) + "\n",
        "validation.json": _json(report) + "\n", "construction_attempts.json": _json(attempts) + "\n",
        # Copy this module so a task can be evaluated with only Python + duckdb.
        "verifier.py": Path(__file__).read_text(),
    }
    for name, contents in files.items():
        (output / name).write_text(contents)
    (output / "README.md").write_text(
        f"# {candidate['title']}\n\n"
        "Give the solver only instruction.md and the warehouse. Keep oracle.sql, task.json, "
        "expected_result.json, provenance, and the verifier-side files hidden until grading.\n\n"
        "```bash\npython verifier.py --db /path/to/retail.duckdb --submission answer.sql\n```\n\n"
        "This task is a composed read-only SQL workflow prototype. See provenance.json for the "
        "actual cross-task witnesses and validation.json for execution evidence and limitations.\n"
    )


def construct_task(path_spec: dict[str, Any], llm: Callable[[str], str | dict[str, Any]],
                   db_path: str | Path, output_dir: str | Path, *, max_repairs: int = 2,
                   context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate, execute, repair, and export one task from a finalized graph path.

    At most ``max_repairs + 1`` LLM calls are made through the supplied callable.
    All unsuccessful attempts are recorded. No model provider is assumed.
    """
    db_path, output = Path(db_path), Path(output_dir)
    if max_repairs < 0:
        raise ValueError("max_repairs cannot be negative")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty task directory: {output}")
    path_spec = normalize_path(path_spec)
    prompt = construction_prompt(path_spec, context if context is not None else database_context(db_path))
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_repairs + 1):
        response = llm(prompt)
        candidate: dict[str, Any] | None = None
        try:
            candidate = _parse_candidate(response)
            report = validate_candidate(candidate, path_spec, db_path)
        except (ValueError, KeyError, TypeError, duckdb.Error) as error:
            attempts.append({"attempt": attempt + 1, "status": "rejected", "error": str(error),
                             "candidate": candidate, "response": response if candidate is None else None})
            prompt = (construction_prompt(path_spec, context if context is not None else database_context(db_path))
                      + "\nPREVIOUS CANDIDATE:\n" + _json(candidate if candidate is not None else response)
                      + "\nEXECUTION/VALIDATION FAILURE:\n" + str(error)
                      + "\nReturn the FULL corrected JSON task. Keep the intended cross-task composition.")
            continue
        attempts.append({"attempt": attempt + 1, "status": "accepted"})
        report.update({"format_version": FORMAT_VERSION, "construction_attempts": len(attempts),
                       "path_id": path_spec["path_id"], "target_superstate": path_spec["target_superstate"]})
        _write_task(output, candidate, path_spec, report, attempts, db_path)
        # The exported grading route must accept the actual composed oracle.
        report["oracle_verifier_check"] = verify_submission(output, db_path, (output / "oracle.sql").read_text())
        (output / "validation.json").write_text(_json(report) + "\n")
        return {"task_id": candidate["task_id"], "output_dir": str(output), "report": report}
    output.mkdir(parents=True, exist_ok=True)
    (output / "construction_attempts.json").write_text(_json(attempts) + "\n")
    raise TaskConstructionError(f"No valid task after {len(attempts)} attempts; see {output / 'construction_attempts.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a submitted SELECT against a constructed SQL task.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        report = verify_submission(args.task_dir, args.db, args.submission.read_text())
    except (ValueError, duckdb.Error) as error:
        report = {"passed": False, "reward": 0, "error": str(error)}
    print(_json(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
