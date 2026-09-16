"""Bounded execution checks and Harbor packaging for generated mutable dbt tasks.

Generated files run only in a disposable Modal sandbox. Passing these checks
establishes executable oracle/result agreement, not preservation of a decision
from the source graph or independent correctness of the generated specification.
"""
from __future__ import annotations

import collections
import datetime as dt
import decimal
import hashlib
import inspect
import json
import math
import re
import shutil
import threading
from pathlib import Path, PurePosixPath
from typing import Any

from .stage import BASE_IMAGE

PROJECT_ROOT = "/app/sg_project"
DATABASE = "/app/database/retail.duckdb"
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
LIMITATIONS = [
    "Execution agreement does not establish semantic preservation of the targeted decision.",
    "The reference query and oracle share a generator; this is not independent correctness evidence.",
    "Results are checked on one pinned warehouse, not adversarial or perturbed data.",
    "No learner difficulty, pooled variance, or training benefit is established by these checks.",
]


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


def _shared_source() -> str:
    return "\n".join([
        "import collections, datetime as dt, decimal, json, math, threading",
        inspect.getsource(_cell), inspect.getsource(query_result), inspect.getsource(compare_results),
    ])


def load_spec(task_dir: Path) -> dict[str, Any]:
    spec = json.loads((task_dir / "task.json").read_text())
    for field in ("task_id", "project_name", "profile_name", "model_name", "target_relation", "target_schema"):
        if not IDENTIFIER.fullmatch(spec.get(field, "")):
            raise ValueError(f"Invalid {field}; use a lowercase identifier")
    if spec["model_name"] != spec["target_relation"]:
        raise ValueError("model_name must equal target_relation")
    if not spec["target_schema"].startswith("sg_"):
        raise ValueError("Use a fresh sg_ target schema")
    if not isinstance(spec.get("instruction"), str) or not spec["instruction"].strip():
        raise ValueError("instruction is required")
    if not isinstance(spec.get("output_columns"), list) or not spec["output_columns"]:
        raise ValueError("output_columns must be nonempty")
    if len(set(spec["output_columns"])) != len(spec["output_columns"]):
        raise ValueError("output_columns must be distinct")
    if not isinstance(spec.get("ordered", False), bool):
        raise ValueError("ordered must be boolean")
    # Parse the reference locally, but never execute generated SQL on the host.
    import duckdb
    statements = duckdb.extract_statements(spec["reference_sql"])
    if len(statements) != 1 or str(statements[0].type).split(".")[-1] != "SELECT":
        raise ValueError("reference_sql must contain exactly one SELECT")
    for field in ("starting_files", "oracle_files"):
        files = spec.get(field)
        if not isinstance(files, dict) or not files:
            raise ValueError(f"{field} must be a nonempty path-to-text mapping")
        if len(files) > 30 or sum(len(str(text)) for text in files.values()) > 200000:
            raise ValueError(f"{field} exceeds bounded project size")
        for name, text in files.items():
            path = PurePosixPath(name)
            if (path.is_absolute() or ".." in path.parts or not path.parts or
                    str(path) != name or "\\" in name or not isinstance(text, str)):
                raise ValueError(f"Unsafe project path in {field}: {name!r}")
            if path.suffix not in {".sql", ".yml", ".yaml", ".md", ".txt"}:
                raise ValueError(f"Unsupported dbt project file type: {name!r}")
    merged = spec["starting_files"] | spec["oracle_files"]
    if not {"dbt_project.yml", "profiles.yml"}.issubset(merged):
        raise ValueError("Oracle requires dbt_project.yml and profiles.yml")
    if not any(name.startswith("models/") and name.endswith(".sql") for name in merged):
        raise ValueError("Oracle needs a dbt model")
    return spec


def _write_files(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def materialize_dbt_task(task_dir: Path) -> Path:
    """Write a Harbor task package; only starting files enter the learner image."""
    task_dir = Path(task_dir)
    spec = load_spec(task_dir)
    for directory in ("environment", "solution", "tests"):
        path = task_dir / directory
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    _write_files(task_dir / "environment/project", spec["starting_files"])
    _write_files(task_dir / "solution/oracle", spec["oracle_files"])
    (task_dir / "instruction.md").write_text(spec["instruction"] + "\n")
    (task_dir / "environment/Dockerfile").write_text(
        f"FROM {BASE_IMAGE}\nWORKDIR {PROJECT_ROOT}\nCOPY project/ {PROJECT_ROOT}/\n"
        f"ENV DB_TYPE=duckdb DUCKDB_PATH={DATABASE}\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[metadata]\ncategory = "data-engineering"\n'
        'difficulty = "unknown"\n[agent]\ntimeout_sec = 1200.0\n'
        '[verifier]\ntimeout_sec = 300.0\n[environment]\nbuild_timeout_sec = 600.0\n'
        'cpus = 2\nmemory_mb = 4096\nstorage_mb = 10240\nallow_internet = false\n')
    (task_dir / "solution/solve.sh").write_text(
        '#!/bin/bash\nset -euo pipefail\n'
        f'cp -a /solution/oracle/. {PROJECT_ROOT}/\ncd {PROJECT_ROOT}\n'
        f'dbt run --project-dir {PROJECT_ROOT} --profiles-dir {PROJECT_ROOT} --no-use-colors\n')
    checker = _shared_source() + '''
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
    actual = query_result('/app/database/retail.duckdb', spec['target_sql'])
    result = compare_results(actual, expected, spec['ordered'])
    print(json.dumps(result))
    if not result['matches']:
        raise RuntimeError('Output differs from the pristine reference')
if __name__ == '__main__':
    main()
'''
    (task_dir / "tests/check_task.py").write_text(checker)
    (task_dir / "tests/check_spec.json").write_text(json.dumps({
        "target_sql": f'SELECT * FROM "{spec["target_schema"]}"."{spec["target_relation"]}"',
        "ordered": spec.get("ordered", False),
    }, indent=2))
    # Expected results do not exist until the sandbox computes them on pristine data.
    (task_dir / "tests/test.sh").write_text(
        '#!/bin/bash\nmkdir -p /logs/verifier\n'
        'echo 0 > /logs/verifier/reward.txt\n'
        'if python3 /tests/check_task.py; then echo 1 > /logs/verifier/reward.txt; else exit 1; fi\n')
    for script in (task_dir / "solution/solve.sh", task_dir / "tests/test.sh"):
        script.chmod(0o755)
    return task_dir


_REMOTE_RUNNER = r'''
import hashlib, os, shutil, subprocess, time
from pathlib import Path
SPEC = json.loads(Path('/tmp/sg_spec.json').read_text())
ROOT = Path('/app/sg_project')
DB = Path('/app/database/retail.duckdb')
RECEIPT = {'status': 'rejected', 'errors': [], 'phases': {}, 'started_at_epoch': time.time()}
def files(mapping):
    for name, text in mapping.items():
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
def dbt(phase):
    try:
        run = subprocess.run(['dbt', 'run', '--project-dir', str(ROOT), '--profiles-dir', str(ROOT),
                              '--no-use-colors'], cwd=ROOT, capture_output=True, text=True,
                             timeout=150, env={**os.environ, 'DB_TYPE': 'duckdb', 'DUCKDB_PATH': str(DB),
                                               'DBT_SEND_ANONYMOUS_USAGE_STATS': 'false'})
        result = {'returncode': run.returncode, 'stdout': run.stdout[-24000:], 'stderr': run.stderr[-8000:]}
    except subprocess.TimeoutExpired:
        result = {'returncode': None, 'error': 'dbt run exceeded 150 seconds'}
    RECEIPT['phases'][phase] = result
    return result
def compare(expected):
    try:
        actual = query_result(str(DB), 'SELECT * FROM "' + SPEC['target_schema'] + '"."' + SPEC['target_relation'] + '"')
        return compare_results(actual, expected, SPEC.get('ordered', False))
    except Exception as exc:
        return {'matches': False, 'error': type(exc).__name__ + ': ' + str(exc)[:2000]}
try:
    pristine_sha = hashlib.sha256()
    with DB.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024): pristine_sha.update(chunk)
    RECEIPT['pristine_database_sha256'] = pristine_sha.hexdigest()
    expected = query_result(str(DB), SPEC['reference_sql'])
    if expected['columns'] != SPEC['output_columns']:
        raise ValueError('Reference columns differ from output_columns')
    RECEIPT['reference'] = {'columns': expected['columns'], 'row_count': len(expected['rows'])}
    before = compare(expected)
    RECEIPT['phases']['before_starting_files'] = before
    if before['matches']:
        raise ValueError('Pristine warehouse already solves the task')
    shutil.copy2(DB, '/tmp/sg_pristine.duckdb')
    ROOT.mkdir(parents=True, exist_ok=True)
    files(SPEC['starting_files'])
    starting_run = dbt('starting_dbt')
    starting = compare(expected)
    RECEIPT['phases']['starting_result'] = starting
    if starting['matches']:
        raise ValueError('Starting project already solves the task')
    # The oracle is checked independently of any mutations from the starter run.
    DB.unlink()
    Path(str(DB) + '.wal').unlink(missing_ok=True)
    shutil.copy2('/tmp/sg_pristine.duckdb', DB)
    shutil.rmtree(ROOT)
    ROOT.mkdir()
    files(SPEC['starting_files'])
    files(SPEC['oracle_files'])
    oracle_run = dbt('oracle_dbt')
    if oracle_run['returncode'] != 0:
        raise ValueError('Oracle dbt run failed')
    oracle_result = compare(expected)
    RECEIPT['phases']['oracle_result'] = oracle_result
    if not oracle_result['matches']:
        raise ValueError('Oracle relation does not equal the pristine reference result')
    Path('/tmp/sg_expected.json').write_text(json.dumps(expected, allow_nan=False))
    RECEIPT['status'] = 'validated'
except Exception as exc:
    RECEIPT['errors'].append(type(exc).__name__ + ': ' + str(exc)[:4000])
finally:
    RECEIPT['finished_at_epoch'] = time.time()
    Path('/tmp/sg_receipt.json').write_text(json.dumps(RECEIPT, indent=2, allow_nan=False))
'''


def validate_dbt_task(task_dir: Path, *, timeout_seconds: int = 600) -> dict[str, Any]:
    """Execute one task in a fresh finite sandbox and always terminate it."""
    if not 360 <= timeout_seconds <= 1200:
        raise ValueError("Sandbox timeout must be 360–1200 seconds")
    task_dir = materialize_dbt_task(Path(task_dir))
    spec = load_spec(task_dir)
    import modal

    receipt: dict[str, Any] = {
        "status": "rejected", "errors": [], "base_image": BASE_IMAGE,
        "spec_sha256": hashlib.sha256((task_dir / "task.json").read_bytes()).hexdigest(),
        "limitations": LIMITATIONS,
    }
    sandbox = None
    try:
        app = modal.App.lookup("superstate-graphs-environments", create_if_missing=True)
        sandbox = modal.Sandbox.create(
            "sleep", str(timeout_seconds), app=app,
            image=modal.Image.from_registry(BASE_IMAGE).entrypoint([]),
            cpu=2, memory=4096, timeout=timeout_seconds, block_network=True,
            env={"DB_TYPE": "duckdb", "DUCKDB_PATH": DATABASE,
                 "DBT_SEND_ANONYMOUS_USAGE_STATS": "false"},
        )
        receipt["sandbox_id"] = sandbox.object_id
        for path, content in (("/tmp/sg_spec.json", json.dumps(spec)),
                              ("/tmp/sg_validate.py", _shared_source() + _REMOTE_RUNNER)):
            with sandbox.open(path, "w") as remote:
                remote.write(content)
        process = sandbox.exec("python3", "/tmp/sg_validate.py", timeout=timeout_seconds - 30)
        process.wait()
        if process.returncode != 0:
            receipt["errors"].append(f"Sandbox validator exited {process.returncode}")
            receipt["stderr"] = process.stderr.read()[-8000:]
        else:
            with sandbox.open("/tmp/sg_receipt.json", "r") as remote:
                receipt.update(json.loads(remote.read()))
            if receipt["status"] == "validated":
                with sandbox.open("/tmp/sg_expected.json", "r") as remote:
                    (task_dir / "tests/expected.json").write_text(remote.read())
    except Exception as exc:
        receipt["status"] = "rejected"
        receipt["errors"].append(type(exc).__name__ + ": " + str(exc)[:2000])
    finally:
        if sandbox is not None:
            try:
                sandbox.terminate()
                receipt["sandbox_terminated"] = True
            except Exception as exc:
                receipt["sandbox_terminated"] = False
                receipt["termination_error"] = type(exc).__name__
        (task_dir / "runtime_validation.json").write_text(json.dumps(receipt, indent=2))
    return receipt
