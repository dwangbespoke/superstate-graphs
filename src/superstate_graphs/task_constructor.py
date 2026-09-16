"""Turn witnessed cross-task graph paths into executable, read-only SQL tasks.

The constructor is deliberately a small callable interface: ``construct_task``
accepts a finalized path, a JSON-producing LLM callable, and a real DuckDB file.
It does not collect rollouts, select graph paths, or make model calls itself.
Generated checks and a second SQL formulation are useful consistency checks,
not an independent demonstration that an LLM-authored task is correct.
"""

from __future__ import annotations

import argparse
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
DEFAULT_SCHEMA_CHAR_BUDGET = 24_000
DEFAULT_PROMPT_CHAR_BUDGET = 60_000
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
        types = ["BOOLEAN" if str(column[1]).upper() in {"BOOL", "BOOLEAN"}
                 else str(column[1]) for column in cursor.description]
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


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def _terms(text: str) -> set[str]:
    stop = {"select", "from", "where", "group", "order", "table", "schema", "source",
            "history", "transition", "task", "stage", "selecting", "main", "raw", "data"}
    return {word.rstrip("s") for word in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text.lower())
            if len(word) > 2 and word not in stop}


def _rank_tables(inventory: list[tuple[str, str, str]], path_spec: dict[str, Any] | None
                 ) -> tuple[list[dict[str, Any]], list[str]]:
    """Rank only catalogued identifiers; an unresolved mention is never a new table."""
    text = "\n".join(_strings(path_spec or {}))
    # Remove identifier quote delimiters, while retaining the exact catalog spelling below.
    normalized = re.sub(r'["`]', "", text).lower()
    sql_refs = set(re.findall(
        r"\b(?:from|join)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)", normalized))
    catalog = {f"{schema}.{table}".lower() for schema, table, _ in inventory}
    bare = {table.lower() for _, table, _ in inventory}
    unresolved = sorted(ref for ref in sql_refs if ref not in catalog and ref not in bare)
    terms = _terms(text)
    ranked = []
    for schema, table, table_type in inventory:
        qualified = f"{schema}.{table}"
        key, name = qualified.lower(), table.lower()
        if (schema.lower() == "main" and key not in sql_refs and name not in sql_refs
                and any(ref.endswith("." + name) for ref in sql_refs)):
            # A qualified witness already identifies another relation; do not spend
            # its context budget on an unreferenced main-view namesake.
            continue
        reasons: list[str] = []
        score = 0
        if key in sql_refs:
            score += 1000
            reasons.append("qualified SQL reference")
        elif name in sql_refs:
            score += 800
            reasons.append("unqualified SQL reference (catalog candidate)")
        if re.search(r"(?<![a-z0-9_])" + re.escape(key) + r"(?![a-z0-9_])", normalized):
            score += 400
            reasons.append("qualified identifier in path evidence")
        elif re.search(r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])", normalized):
            score += 150
            reasons.append("table identifier in path evidence")
        overlap = terms.intersection(_terms(qualified))
        if overlap:
            score += 8 * len(overlap)
            reasons.append("lexical overlap: " + ", ".join(sorted(overlap)))
        if score:
            # Prefer underlying tables to thousands of duplicate main views, but retain
            # explicitly referenced views. Backups remain eligible only when warranted.
            if table_type == "BASE TABLE":
                score += 20
            if schema.lower() != "main":
                score += 5
            if re.search(r"(?:^|_)(?:hist|backup|tmp|stg|v[0-9]+)(?:_|$)", name):
                score -= 10
            ranked.append({"schema": schema, "table": table, "table_type": table_type,
                           "qualified_name": qualified, "score": score, "selection_reasons": reasons})
    if not ranked and len(inventory) <= 40:
        ranked = [{"schema": schema, "table": table, "table_type": table_type,
                   "qualified_name": f"{schema}.{table}", "score": 0,
                   "selection_reasons": ["small-catalog fallback; no matching path reference"]}
                  for schema, table, table_type in inventory]
    ranked.sort(key=lambda item: (-item["score"], item["qualified_name"]))
    return ranked, unresolved


def database_context(db_path: str | Path, *, path_spec: dict[str, Any] | None = None,
                     sample_rows: int = 2, max_tables: int = 24,
                     max_chars: int = DEFAULT_SCHEMA_CHAR_BUDGET) -> dict[str, Any]:
    """Retrieve a bounded relevant schema, without sampling the whole warehouse.

    Only a compact table-name inventory is loaded globally. Columns and samples
    are fetched for at most ``max_tables`` catalogued relations. Exact SQL/table
    mentions rank above lexical similarity. Returned metadata makes omissions and
    ambiguous/unresolved references visible; no physical table is invented.
    """
    if not 1 <= max_tables <= 40 or not 0 <= sample_rows <= 3 or max_chars < 2000:
        raise ValueError("Require 1–40 tables, 0–3 sample rows, and at least 2000 context characters")
    connection = _connect(db_path)
    try:
        inventory = connection.execute(
            "SELECT table_schema, table_name, table_type FROM information_schema.tables "
            "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') "
            "ORDER BY table_schema, table_name"
        ).fetchall()
        ranked, unresolved = _rank_tables(inventory, path_spec)
        if not ranked:
            raise ValueError("No path-related table matched the large warehouse catalog; "
                             "supply concrete SQL/table mentions in path witnesses")
        context: dict[str, Any] = {
            "engine": "DuckDB", "tables": [],
            "retrieval": {
                "catalog_relation_count": len(inventory), "relevant_candidate_count": len(ranked),
                "max_tables": max_tables, "max_characters": max_chars,
                "unresolved_sql_references": unresolved[:25],
                "unresolved_sql_reference_count": len(unresolved),
                "selection": "Exact witnessed SQL/identifiers, then lexical overlap; base tables preferred",
                "interpretation": "A bounded retrieval, not the full schema. Unresolved references may be CTEs, dbt models, or unavailable tables.",
                "sampled_relation_count": 0, "omitted_relevant_relation_count": len(ranked),
            },
        }
        sampled = 0
        for relation in ranked[:max_tables]:
            schema, table = relation["schema"], relation["table"]
            column_rows = connection.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                [schema, table],
            ).fetchall()
            record = dict(relation)
            # Wide retail tables have >100 columns; compact names→types retains
            # their whole schema without verbose repeated metadata keys.
            record["columns"] = dict(column_rows)
            record["sample_rows"] = []
            if sample_rows:
                try:
                    evidence = "\n".join(_strings(path_spec or {})).lower()
                    sample_columns = sorted(
                        [column for column, _ in column_rows],
                        key=lambda column: (
                            -int(bool(re.search(r"\b" + re.escape(column.lower()) + r"\b", evidence))),
                            -int(column.lower().endswith("_id")), column,
                        ),
                    )[:12]
                    selection = ", ".join('"' + column.replace('"', '""') + '"'
                                          for column in sample_columns)
                    sample = _query(connection,
                                    f"SELECT {selection} FROM {_qualify(schema, table)} LIMIT {sample_rows}")
                    sampled += 1
                    record["sample_columns"] = sample["columns"]
                    record["sample_rows"] = [
                        [cell[:120] if isinstance(cell, str) else cell for cell in row]
                        for row in sample["rows"]
                    ]
                except (ValueError, duckdb.Error) as error:
                    record["sample_error"] = str(error)[:300]
            context["tables"].append(record)
            if len(_json(context)) > max_chars - 200:
                record["sample_rows"] = []
                record["samples_omitted_for_budget"] = True
            if len(_json(context)) > max_chars - 200:
                context["tables"].pop()
                continue
        context["retrieval"]["sampled_relation_count"] = sampled
        context["retrieval"]["selected_relation_count"] = len(context["tables"])
        context["retrieval"]["omitted_relevant_relation_count"] = len(ranked) - len(context["tables"])
        context["retrieval"]["context_characters"] = 0
        for _ in range(3):
            context["retrieval"]["context_characters"] = len(_json(context))
        if not context["tables"] or len(_json(context)) > max_chars:
            raise ValueError("Relevant schema cannot fit the context budget; increase max_chars or narrow path")
        return context
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


def _excerpt(value: Any, max_chars: int) -> Any:
    serialized = value if isinstance(value, str) else _json(value)
    if len(serialized) <= max_chars:
        return value
    return {"excerpt": serialized[:max_chars], "original_characters": len(serialized),
            "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "note": "Excerpt only; full original is retained in provenance.json"}


def _prompt_path(path_spec: dict[str, Any], *, max_chars: int = 18_000) -> dict[str, Any]:
    """Bound prompt evidence while retaining all source IDs and full disk provenance."""
    identity_fields = {"transition_id", "source_task_id", "source_history_id", "target_history_id"}
    for field_budget in (3000, 1800, 1000, 500, 250):
        projected: dict[str, Any] = {
            "path_id": path_spec["path_id"],
            "target_superstate": _excerpt(path_spec["target_superstate"], field_budget),
            "transitions": [],
        }
        additional = {key: value for key, value in path_spec.items()
                      if key not in {"path_id", "target_superstate", "transitions"}}
        if additional:
            projected["additional_path_metadata"] = _excerpt(additional, field_budget)
        for transition in path_spec["transitions"]:
            record = {key: transition[key] for key in identity_fields}
            record["operation"] = _excerpt(transition["operation"], field_budget)
            evidence = {key: value for key, value in transition.items()
                        if key not in identity_fields | {"operation"}}
            record["evidence"] = _excerpt(evidence, field_budget)
            projected["transitions"].append(record)
        if len(_json(projected)) <= max_chars:
            return projected
    raise ValueError("Path identities/evidence exceed prompt budget; select a shorter path")


def construction_prompt(path_spec: dict[str, Any], context: dict[str, Any], *,
                        max_chars: int = DEFAULT_PROMPT_CHAR_BUDGET) -> str:
    prompt = """Construct ONE new executable analytical task in this shared DuckDB warehouse.
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
The schema is a relevance-filtered subset. Use exact qualified table names and
provided columns; do not invent missing tables or silently replace unresolved names.
Every intermediate stage must return at least one row. All intermediate stages are
named CTEs visible to subsequent stages. Constraints see the stage CTEs plus sg_result
and each must return exactly one TRUE boolean, inspecting sg_result nontrivially.
Avoid unbounded raw joins; aggregate intermediate data at a specified grain.
The alternate query is a self-consistency check, not independent oracle evidence.

PATH AND ACTUAL WITNESSES:
""" + _json(_prompt_path(path_spec)) + "\nWAREHOUSE SCHEMA AND REAL SAMPLE VALUES:\n" + _json(context)
    if len(prompt) > max_chars:
        raise ValueError(f"Construction prompt exceeds strict {max_chars}-character budget: {len(prompt)}")
    return prompt


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
                report: dict[str, Any], attempts: list[dict[str, Any]], db_path: Path,
                context: dict[str, Any]) -> None:
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
        "schema_context.json": _json(context) + "\n",
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
                   context: dict[str, Any] | None = None,
                   max_prompt_chars: int = DEFAULT_PROMPT_CHAR_BUDGET) -> dict[str, Any]:
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
    context = context if context is not None else database_context(db_path, path_spec=path_spec)
    base_prompt = construction_prompt(path_spec, context, max_chars=max_prompt_chars)
    prompt = base_prompt
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
            repair_suffix = ("\nEXECUTION/VALIDATION FAILURE:\n" + str(error)[:2000]
                             + "\nReturn the FULL corrected JSON task. Keep the intended cross-task composition.")
            remaining = max_prompt_chars - len(base_prompt) - len(repair_suffix) - 400
            if remaining < 500:
                raise ValueError("Prompt budget leaves insufficient room for repair feedback") from error
            previous = _excerpt(candidate if candidate is not None else response, min(12_000, remaining))
            prompt = base_prompt + "\nPREVIOUS CANDIDATE:\n" + _json(previous) + repair_suffix
            if len(prompt) > max_prompt_chars:
                raise ValueError("Repair prompt exceeds strict character budget") from error
            continue
        attempts.append({"attempt": attempt + 1, "status": "accepted"})
        report.update({"format_version": FORMAT_VERSION, "construction_attempts": len(attempts),
                       "path_id": path_spec["path_id"], "target_superstate": path_spec["target_superstate"],
                       "schema_retrieval": context.get("retrieval", {"provided_by_caller": True}),
                       "prompt_characters": len(prompt), "max_prompt_characters": max_prompt_chars})
        _write_task(output, candidate, path_spec, report, attempts, db_path, context)
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
