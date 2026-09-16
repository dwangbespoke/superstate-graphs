import collections, datetime as dt, decimal, json, math, threading
def _cell(value):
    if isinstance(value, decimal.Decimal):
        return {"decimal": str(value.normalize())}
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return {"temporal": value.isoformat()}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    if isinstance(value, (tuple, list)):
        return [_cell(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _cell(item) for key, item in value.items()}
    return value

def query_result(database, sql, max_rows=10000):
    import duckdb

    statements = duckdb.extract_statements(sql)
    if len(statements) != 1 or str(statements[0].type).split(".")[-1] != "SELECT":
        raise ValueError("Reference and result queries must be one read-only SELECT")
    con = duckdb.connect(database, read_only=True, config={
        "enable_external_access": "false", "threads": "2", "memory_limit": "1GB",
    })
    timer = threading.Timer(45, con.interrupt)
    timer.daemon = True
    timer.start()
    try:
        cursor = con.execute(sql)
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError("Result exceeds 10,000-row validation limit")
        if not rows or len(set(columns)) != len(columns):
            raise ValueError("Result must be nonempty with distinct column names")
        return {"columns": columns, "rows": [[_cell(cell) for cell in row] for row in rows]}
    finally:
        timer.cancel()
        con.close()

def compare_results(actual, expected, ordered=False):
    if actual["columns"] != expected["columns"]:
        return {"matches": False, "reason": "column names/order differ"}
    def key(row):
        return json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False)
    actual_rows = [key(row) for row in actual["rows"]]
    expected_rows = [key(row) for row in expected["rows"]]
    match = (actual_rows == expected_rows if ordered else
             collections.Counter(actual_rows) == collections.Counter(expected_rows))
    return {"matches": match, "reason": "equal" if match else "row values/multiplicities differ",
            "actual_rows": len(actual_rows), "expected_rows": len(expected_rows)}

def relation_result(database, schema, relation):
    """A view with the right rows does not satisfy a materialized-table task."""
    import duckdb

    con = duckdb.connect(database, read_only=True, config={
        "enable_external_access": "false", "threads": "2", "memory_limit": "1GB",
    })
    try:
        rows = con.execute(
            "SELECT table_type FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?", [schema, relation],
        ).fetchall()
        if rows != [("BASE TABLE",)]:
            raise ValueError(f"Target must be a materialized BASE TABLE; found {rows!r}")
    finally:
        con.close()
    qualified = '"' + schema.replace('"', '""') + '"."' + relation.replace('"', '""') + '"'
    return query_result(database, "SELECT * FROM " + qualified)

from pathlib import Path
import subprocess, sys
def main():
    expected = json.loads(Path('/tests/expected.json').read_text())
    spec = json.loads(Path('/tests/check_spec.json').read_text())
    run = subprocess.run(['dbt', 'run', '--project-dir', '/app/sg_project', '--profiles-dir',
                          '/app/sg_project', '--no-use-colors'], capture_output=True, text=True, timeout=180)
    print(run.stdout[-12000:]); print(run.stderr[-4000:])
    if run.returncode:
        raise RuntimeError('dbt run failed')
    actual = relation_result('/app/database/retail.duckdb', spec['target_schema'], spec['target_relation'])
    result = compare_results(actual, expected, spec['ordered'])
    print(json.dumps(result))
    if not result['matches']:
        raise RuntimeError('Output differs from the pristine reference')
if __name__ == '__main__':
    main()
