"""Executable synthetic DuckDB tasks grounded in concrete learned graph paths.

Only a restricted JSON fixture and SELECT SQL are model-authored.  Python, shell,
and verifier code are never generated or executed by a model.  A separate prompt
derives a second query without seeing the first.  Query agreement is local
consistency evidence, not a proof of semantic fidelity or a benchmark replication.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import decimal
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import duckdb

from .graph_analysis import SOURCE_DATA_SUFFIX, task_draft_messages
from .graph_llm import qwen_generation_policy
from .graph_schemas import SHORT, TEXT, TEXTS, obj


FORMAT = "synthetic-path-duckdb-v1"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
TYPES = {"INTEGER", "BIGINT", "DOUBLE", "VARCHAR", "DATE", "BOOLEAN", "DECIMAL(18,2)"}
FIXTURE_SCHEMA = obj({
    "status": {"type": "string", "enum": ["task", "unsupported"]},
    "reason": SHORT, "title": TEXT, "learner_instruction": TEXT,
    "output_columns": TEXTS, "ordered_output": {"type": "boolean"},
    "path_transition_ids": TEXTS, "preserved_challenges": TEXTS, "adaptations": TEXTS,
    "tables": {"type": "array", "maxItems": 4, "items": obj({
        "name": TEXT, "columns": {"type": "array", "minItems": 1, "maxItems": 12,
                                  "items": obj({"name": TEXT, "type": {
                                      "type": "string", "enum": sorted(TYPES)}})},
        "rows": {"type": "array", "minItems": 2, "maxItems": 60,
                 "items": {"type": "array", "items": {"type": ["string", "number", "boolean", "null"]}}},
    })},
    "reference_sql": TEXT,
})
INDEPENDENT_SQL_SCHEMA = obj({
    "verdict": {"type": "string", "enum": ["faithful", "unsupported", "unclear"]},
    "reason": SHORT, "independent_sql": TEXT, "findings": TEXTS,
    "preserved_local_challenges": TEXTS,
})
LIMITATIONS = [
    "This is a newly synthesized fixture, not the source benchmark environment or a restored history.",
    "Two separately prompted SQL formulations agree locally; both may share modeling errors.",
    "The verifier checks final query results, not whether a learner followed the intended reasoning path.",
    "No learner success rate, conditional continuation variance, training benefit, or universal graph contract is established.",
]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value))
    temporary.replace(path)


def _quote(name: str) -> str:
    if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
        raise ValueError(f"Invalid fixture identifier: {name!r}")
    return '"' + name + '"'


def validate_fixture_spec(spec: Mapping[str, Any], path: Mapping[str, Any] | None = None
                          ) -> dict[str, Any]:
    if spec.get("status") not in {"task", "unsupported"}:
        raise ValueError("Generator must return task or unsupported")
    if spec["status"] == "unsupported":
        return dict(spec)
    for key in ("title", "learner_instruction", "reference_sql"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"Task requires a nonempty {key}")
    columns = spec.get("output_columns")
    if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
        raise ValueError("Task must specify distinct ordered output column names")
    for column in columns:
        _quote(column)
    tables = spec.get("tables")
    if not isinstance(tables, list) or not 1 <= len(tables) <= 4:
        raise ValueError("Use one through four fixture tables")
    names, total_rows = set(), 0
    for table in tables:
        name = table.get("name")
        _quote(name)
        if name.lower() in names:
            raise ValueError("Duplicate fixture table name")
        names.add(name.lower())
        definitions = table.get("columns")
        if not isinstance(definitions, list) or not 1 <= len(definitions) <= 12:
            raise ValueError("Use one through twelve typed columns per table")
        column_names = set()
        for column in definitions:
            _quote(column.get("name"))
            if column["name"].lower() in column_names or column.get("type") not in TYPES:
                raise ValueError("Duplicate column or unsupported column type")
            column_names.add(column["name"].lower())
        rows = table.get("rows")
        if not isinstance(rows, list) or not 2 <= len(rows) <= 60:
            raise ValueError("Use two through sixty explicit fixture rows per table")
        total_rows += len(rows)
        for row in rows:
            if not isinstance(row, list) or len(row) != len(definitions):
                raise ValueError("Fixture row width disagrees with its columns")
            for cell in row:
                if cell is not None and not isinstance(cell, (str, int, float, bool)):
                    raise ValueError("Fixture cells must be JSON scalars")
                if isinstance(cell, float) and not math.isfinite(cell):
                    raise ValueError("Fixture numeric values must be finite")
                if isinstance(cell, str) and len(cell) > 4096:
                    raise ValueError("Fixture strings are limited to 4096 characters")
    if total_rows > 160:
        raise ValueError("Fixture exceeds the 160-row budget")
    if path is not None:
        expected = [t["transition_id"] for t in path["transitions"]]
        if spec.get("path_transition_ids") != expected:
            raise ValueError("Fixture task must preserve its ordered path provenance")
    _select(spec["reference_sql"])
    return dict(spec)


def _select(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 40_000:
        raise ValueError("SQL must be nonempty and at most 40,000 characters")
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1 or str(statements[0].type).split(".")[-1] != "SELECT":
        raise ValueError("Only one SELECT/WITH query is permitted")
    return sql


def _cell(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return {"decimal": str(value)}
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Query produced a nonfinite numeric result")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported query result type {type(value).__name__}")


def _query(connection: Any, sql: str) -> dict[str, Any]:
    cursor = connection.execute(_select(sql))
    rows = cursor.fetchmany(501)
    if len(rows) > 500:
        raise ValueError("Task result exceeds 500 rows")
    columns = [column[0] for column in cursor.description]
    if len(set(columns)) != len(columns):
        raise ValueError("Query result has duplicate column names")
    return {"columns": columns, "rows": [[_cell(value) for value in row] for row in rows]}


def _numeric(value: Any) -> decimal.Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return decimal.Decimal(str(value))
    if isinstance(value, dict) and set(value) == {"decimal"}:
        return decimal.Decimal(value["decimal"])
    return None


def _same_cell(left: Any, right: Any) -> bool:
    a, b = _numeric(left), _numeric(right)
    if a is not None or b is not None:
        return a is not None and b is not None and abs(a - b) <= decimal.Decimal("0.00000001")
    return type(left) is type(right) and left == right


def compare_query_results(left: Mapping[str, Any], right: Mapping[str, Any], *,
                           ordered: bool = False) -> bool:
    """Preserve duplicate rows; numerical tolerance is absolute 1e-8 only."""
    if left["columns"] != right["columns"] or len(left["rows"]) != len(right["rows"]):
        return False

    def same_row(a: Sequence[Any], b: Sequence[Any]) -> bool:
        return len(a) == len(b) and all(_same_cell(x, y) for x, y in zip(a, b))

    if ordered:
        return all(same_row(a, b) for a, b in zip(left["rows"], right["rows"]))
    pending = list(right["rows"])
    for row in left["rows"]:
        for index, other in enumerate(pending):
            if same_row(row, other):
                pending.pop(index)
                break
        else:
            return False
    return True


def _connect(path: Path, *, read_only: bool) -> Any:
    return duckdb.connect(str(path), read_only=read_only, config={
        "enable_external_access": "false", "threads": "1", "memory_limit": "256MB",
        "allow_unsigned_extensions": "false",
    })


def _worker(request_path: Path, response_path: Path) -> None:
    started = time.monotonic()
    request = json.loads(request_path.read_text())
    report: dict[str, Any] = {"status": "execution_failed", "reference_query_executed": False,
                              "independent_query_executed": False}
    try:
        database = Path(request["database"])
        if request["mode"] == "construct":
            spec = validate_fixture_spec(request["spec"])
            connection = _connect(database, read_only=False)
            try:
                for table in spec["tables"]:
                    columns = ", ".join(_quote(c["name"]) + " " + c["type"] for c in table["columns"])
                    connection.execute("CREATE TABLE " + _quote(table["name"]) + " (" + columns + ")")
                    placeholders = ",".join("?" for _ in table["columns"])
                    connection.executemany("INSERT INTO " + _quote(table["name"]) +
                                           " VALUES (" + placeholders + ")", table["rows"])
            finally:
                connection.close()
            connection = _connect(database, read_only=True)
            try:
                reference = _query(connection, spec["reference_sql"])
                report["reference_query_executed"] = True
                independent = _query(connection, request["independent_sql"])
                report["independent_query_executed"] = True
                report["reference_result"], report["independent_result"] = reference, independent
                report["nonempty_result"] = bool(reference["rows"])
                report["output_schema_matches"] = reference["columns"] == spec["output_columns"]
                report["queries_agree"] = compare_query_results(
                    reference, independent, ordered=bool(spec.get("ordered_output", False)))
                report["status"] = ("locally_executed_consistent" if all(report[key] for key in
                                     ("nonempty_result", "output_schema_matches", "queries_agree"))
                                    else "consistency_failed")
            finally:
                connection.close()
        elif request["mode"] == "verify":
            connection = _connect(database, read_only=True)
            try:
                actual = _query(connection, request["submission_sql"])
                passed = compare_query_results(actual, request["expected_result"],
                                               ordered=request.get("ordered_output", False))
                report.update({"status": "passed" if passed else "incorrect_result",
                               "submission_query_executed": True, "result": actual})
            finally:
                connection.close()
        else:
            raise ValueError("Unknown worker mode")
    except Exception as error:
        report["error_type"], report["error"] = type(error).__name__, str(error)
    report["elapsed_seconds"] = time.monotonic() - started
    _write(response_path, report)


def _run_worker(request: Mapping[str, Any], work: Path, timeout_seconds: float) -> dict[str, Any]:
    _write(work / "request.json", request)
    # Do not pass serving/API credentials into the SQL worker.
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                   "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                   "PYTHONNOUSERSITE": "1"}
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "superstate_graphs.graph_task_examples", "_worker",
             str(work / "request.json"), str(work / "response.json")],
            capture_output=True, text=True, timeout=timeout_seconds, env=environment)
    except subprocess.TimeoutExpired:
        return {"status": "execution_timeout", "timeout_seconds": timeout_seconds}
    if completed.returncode != 0 or not (work / "response.json").exists():
        return {"status": "execution_failed", "worker_exit_code": completed.returncode,
                "worker_stderr": completed.stderr[-2000:]}
    return json.loads((work / "response.json").read_text())


def run_local_validation(spec: Mapping[str, Any], independent_sql: str, output: Path | str, *,
                         timeout_seconds: float = 30) -> dict[str, Any]:
    spec = validate_fixture_spec(spec)
    if spec["status"] != "task":
        raise ValueError("Cannot execute an unsupported task specification")
    _select(independent_sql)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="local-check-", dir=output) as temporary:
        work = Path(temporary)
        request = {"mode": "construct", "database": str(work / "fixture.duckdb"),
                   "spec": spec, "independent_sql": independent_sql}
        report = _run_worker(request, work, timeout_seconds)
        if report["status"] == "locally_executed_consistent":
            shutil.copy2(work / "fixture.duckdb", output / "fixture.duckdb")
            _write(output / "expected_result.json", report["independent_result"])
    report.update({"format": FORMAT, "synthetic_fixture": True, "learner_evaluated": False,
                   "original_benchmark_reproduced": False, "official_benchmark_verifier": False,
                   "universal_graph_contract_certified": False, "limitations": LIMITATIONS})
    _write(output / "local_validation.json", report)
    return report


def verify_submission(task_dir: Path | str, sql: str, *, timeout_seconds: float = 30
                       ) -> dict[str, Any]:
    output = Path(task_dir).resolve()
    spec = json.loads((output / "task.json").read_text())
    expected = json.loads((output / "expected_result.json").read_text())
    with tempfile.TemporaryDirectory(prefix="submission-check-") as temporary:
        return _run_worker({"mode": "verify", "database": str(output / "fixture.duckdb"),
                            "submission_sql": sql, "expected_result": expected,
                            "ordered_output": bool(spec.get("ordered_output", False))},
                           Path(temporary), timeout_seconds)


GENERATOR_SYSTEM = """Create a SMALL, RUNNABLE SYNTHETIC DuckDB task faithfully motivated by this
actual graph path and its complete source histories. It is a new fixture, not a benchmark clone.
Preserve the unresolved local challenge and the bounded operations; do not replace an environment,
dbt configuration, or dependency-repair challenge with an unrelated SELECT problem. If a read-only
DuckDB fixture cannot preserve the path, return {status:'unsupported',reason:'...'}.
For unsupported, fill other schema fields with empty strings/lists and false as appropriate.

Otherwise return JSON with status:'task',title,learner_instruction,output_columns,ordered_output,
path_transition_ids in exact path order,preserved_challenges,adaptations,tables,reference_sql.
Use 1-4 tables, 2-60 explicit rows per table, at most160 total rows. Each table is
{name,columns:[{name,type}],rows:[[scalar,...]]}. Identifier names use letters/digits/underscores.
Allowed types: INTEGER,BIGINT,DOUBLE,VARCHAR,DATE,BOOLEAN,DECIMAL(18,2). Include enough counterexamples
or contrasting records that the local decision matters; a coincidentally correct shortcut should
not solve the task. Explain business semantics sufficient to determine a unique correct result,
but do not reveal an intended join-key discovery or other answer in the learner instruction.
The learner receives only instruction.md and fixture.duckdb and submits one SELECT/WITH query.
Specify output column names and whether output row order matters. reference_sql must be one SELECT
query, nonempty in result, at most500 result rows, with no external access or extension requirement.
Never generate Python, shell commands, or claims of executed validation.
"""


REVIEWER_SYSTEM = """Independently solve and assess this proposed synthetic DuckDB task. You see
its instruction, complete fixture data, output schema, and source-path evidence, but NOT the
generator's reference SQL. Do not assume the generator's preservation claims are true. Determine
whether the task is unambiguous, fixtures make the intended local challenge meaningful, and the
task actually preserves the path's local operations. Return JSON {verdict:'faithful|unsupported|unclear',
reason,independent_sql,findings:[],preserved_local_challenges:[]}.
For faithful, independently derive one SELECT/WITH query matching the ordered output columns.
Do not claim the query has run or that this reproduces the original benchmark. A different
downstream business goal is permitted; changing the immediate challenge is not.
"""


async def construct_executable_example(path: Mapping[str, Any],
                                        prefix_provider: Callable[[str], str], llm: Any,
                                        output: Path | str, *, timeout_seconds: float = 30,
                                        thinking: bool = True,
                                        ) -> dict[str, Any]:
    """Produce one grounded task, independently solve it, then execute both queries."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = json.loads(task_draft_messages(path, prefix_provider)[1]["content"])
    messages = [{"role": "system", "content": GENERATOR_SYSTEM},
                {"role": "user", "content": _json(context)},
                {"role": "user", "content": SOURCE_DATA_SUFFIX}]
    settings = getattr(llm, "runtime", {})
    public_model = {key: settings[key] for key in
                    ("model", "revision", "model_revision", "tokenizer_revision", "quantization")
                    if key in settings}
    public_model.setdefault("model", getattr(llm, "model", type(llm).__qualname__))
    provenance = {"implementation": "executable-path-task-v2", "path": dict(path),
                  "generator_messages": messages, "generator_schema": FIXTURE_SCHEMA,
                  "reviewer_system": REVIEWER_SYSTEM, "reviewer_schema": INDEPENDENT_SQL_SCHEMA,
                  "public_model": public_model, "thinking": thinking,
                  "generator_max_tokens": 16384 if thinking else 12000,
                  "reviewer_max_tokens": 8192,
                  "generator_generation_policy": qwen_generation_policy(
                      thinking=thinking, max_tokens=16384 if thinking else 12000,
                      temperature=.2),
                  "reviewer_generation_policy": qwen_generation_policy(
                      thinking=thinking, max_tokens=8192),
                  "seed": 17,
                  "timeout_seconds": timeout_seconds}
    request_key = hashlib.sha256(_json(provenance).encode()).hexdigest()
    completed_path = output / "result.json"
    if completed_path.exists():
        previous = json.loads(completed_path.read_text())
        reusable = {"locally_executed_consistent", "unsupported",
                    "independent_review_not_supported", "consistency_failed"}
        hashes = previous.get("artifact_sha256", {})
        intact = bool(hashes) and all(
            (output / name).is_file() and
            hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
            for name, expected in hashes.items())
        if (previous.get("request_key") == request_key
                and previous.get("status") in reusable and intact):
            return {**previous, "cache_reused": True}
        # Preserve the previous receipt even when this invocation repairs missing
        # artifacts or changes model/evidence. Do not call old files current evidence.
        archive = output / "previous_receipts"
        archive.mkdir(exist_ok=True)
        old_key = previous.get("request_key", "unversioned")
        _write(archive / f"{old_key}.json", previous)
    _write(output / "source_path.json", dict(path))
    try:
        raw = await llm.shard("sql-generator:" + path["path_id"]).complete_json(
            messages, schema=FIXTURE_SCHEMA, max_tokens=16384 if thinking else 12000,
            thinking=thinking, temperature=.2, cache_namespace="synthetic-path-task-v2")
        _write(output / "generator_response.json", raw)
        spec = validate_fixture_spec(raw, path)
        spec.update({"format": FORMAT, "path_id": path["path_id"], "synthetic_fixture": True})
        _write(output / "task.json", spec)
        if spec["status"] == "unsupported":
            result = {"status": "unsupported", "reason": spec.get("reason"), "executed": False}
        else:
            learner_spec = {key: value for key, value in spec.items() if key != "reference_sql"}
            independent = await llm.shard("sql-independent:" + path["path_id"]).complete_json(
                [{"role": "system", "content": REVIEWER_SYSTEM},
                 {"role": "user", "content": _json({"task": learner_spec, "source_evidence": context})},
                 {"role": "user", "content": SOURCE_DATA_SUFFIX}],
                schema=INDEPENDENT_SQL_SCHEMA, max_tokens=8192,
                thinking=thinking, cache_namespace="synthetic-independent-sql-v2")
            _write(output / "independent_solution_review.json", independent)
            if independent.get("verdict") != "faithful":
                result = {"status": "independent_review_not_supported", "executed": False,
                          "review_verdict": independent.get("verdict"),
                          "reason": independent.get("reason")}
            else:
                sql = independent.get("independent_sql")
                _select(sql)
                (output / "instruction.md").write_text(spec["learner_instruction"] + "\n")
                (output / "oracle.sql").write_text(spec["reference_sql"] + "\n")
                (output / "independent.sql").write_text(sql + "\n")
                report = await asyncio.to_thread(run_local_validation, spec, sql, output,
                                                 timeout_seconds=timeout_seconds)
                result = {"status": report["status"], "executed": report.get("reference_query_executed", False),
                          "review_verdict": "faithful", "local_validation": report}
                if report["status"] == "locally_executed_consistent":
                    fingerprint = hashlib.sha256((output / "fixture.duckdb").read_bytes()).hexdigest()
                    result["fixture_sha256"] = fingerprint
                    (output / "README.md").write_text(
                        "# Synthetic DuckDB task\n\nGive the learner only `instruction.md` and "
                        "`fixture.duckdb`. Keep reference queries and expected results hidden.\n\n"
                        "Verify a submitted SQL file with:\n\n"
                        "```sh\npython -m superstate_graphs.graph_task_examples verify "
                        f"--task-dir '{output}' --submission answer.sql\n```\n\n"
                        "Two separately prompted reference queries were executed locally and agreed. "
                        "This does not establish original-benchmark fidelity, learner success, or "
                        "universal graph validity. The verifier checks final output only.\n")
    except Exception as error:
        result = {"status": "construction_error", "executed": False,
                  "error_type": type(error).__name__, "error": str(error)[:1000]}
    result.update({"path_id": path["path_id"], "output_dir": str(output),
                   "request_key": request_key, "public_model": public_model,
                   "thinking": thinking, "cache_reused": False,
                   "synthetic_fixture": True, "learner_evaluated": False,
                   "original_benchmark_reproduced": False, "limitations": LIMITATIONS})
    required = ["source_path.json", "generator_response.json", "task.json"]
    if result["status"] != "unsupported":
        required.append("independent_solution_review.json")
    if result["status"] in {"locally_executed_consistent", "consistency_failed"}:
        required += ["local_validation.json", "instruction.md", "oracle.sql", "independent.sql"]
    if result["status"] == "locally_executed_consistent":
        required += ["fixture.duckdb", "expected_result.json", "README.md"]
    result["artifact_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in required if (output / name).is_file()}
    _write(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("_worker")
    worker.add_argument("request", type=Path)
    worker.add_argument("response", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--task-dir", type=Path, required=True)
    verify.add_argument("--submission", type=Path, required=True)
    verify.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.command == "_worker":
        _worker(args.request, args.response)
    else:
        result = verify_submission(args.task_dir, args.submission.read_text(),
                                   timeout_seconds=args.timeout_seconds)
        print(_json(result))
        raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
