import json
import io
import sys
from types import SimpleNamespace

import pytest

from superstate_graphs.dbt_runtime import (
    _REMOTE_RUNNER, _shared_source, compare_results, load_spec, materialize_dbt_task,
    relation_result, validate_dbt_task,
)


def _spec():
    return {"task_id": "sg_example", "project_name": "sg_project", "profile_name": "sg_project",
            "model_name": "revenue", "target_relation": "revenue", "target_schema": "sg_example",
            "instruction": "Build revenue in /app/sg_project.", "output_columns": ["n"],
            "reference_sql": "SELECT 3 AS n", "ordered": False,
            "starting_files": {"dbt_project.yml": "name: sg_project\n", "profiles.yml": "todo: true\n"},
            "oracle_files": {"models/revenue.sql": "SELECT 3 AS n"}}


def _write(tmp_path, spec):
    (tmp_path / "task.json").write_text(json.dumps(spec))
    return tmp_path


def test_package_keeps_oracle_and_expected_out_of_image(tmp_path):
    materialize_dbt_task(_write(tmp_path, _spec()))
    assert not (tmp_path / "environment/project/models/revenue.sql").exists()
    assert (tmp_path / "solution/oracle/models/revenue.sql").read_text() == "SELECT 3 AS n"
    dockerfile = (tmp_path / "environment/Dockerfile").read_text()
    assert "COPY project/" in dockerfile and "solution" not in dockerfile and "tests" not in dockerfile
    assert not (tmp_path / "tests/expected.json").exists()
    compile((tmp_path / "tests/check_task.py").read_text(), "check_task.py", "exec")
    compile(_shared_source() + _REMOTE_RUNNER, "remote.py", "exec")


@pytest.mark.parametrize("path", ["../outside.sql", "/tmp/model.sql", "models/../../bad.sql", "./x.sql", "models/x.sh"])
def test_project_paths_are_bounded(tmp_path, path):
    spec = _spec()
    spec["oracle_files"][path] = "SELECT 3"
    with pytest.raises(ValueError):
        load_spec(_write(tmp_path, spec))


def test_result_multiset_preserves_duplicates_and_column_order():
    expected = {"columns": ["x", "y"], "rows": [[1, 2], [1, 2], [3, 4]]}
    permuted = {"columns": ["x", "y"], "rows": [[3, 4], [1, 2], [1, 2]]}
    assert compare_results(permuted, expected)["matches"]
    assert not compare_results(permuted, expected, ordered=True)["matches"]
    assert not compare_results({**expected, "rows": [[1, 2], [3, 4]]}, expected)["matches"]
    assert not compare_results({**expected, "columns": ["y", "x"]}, expected)["matches"]


def test_relation_must_be_materialized_table_not_equivalent_view(tmp_path):
    import duckdb
    database = str(tmp_path / "test.duckdb")
    con = duckdb.connect(database)
    con.execute("CREATE TABLE correct_table AS SELECT 3 AS n")
    con.execute("CREATE VIEW equivalent_view AS SELECT 3 AS n")
    con.close()
    assert relation_result(database, "main", "correct_table") == {"columns": ["n"], "rows": [[3]]}
    with pytest.raises(ValueError, match="BASE TABLE"):
        relation_result(database, "main", "equivalent_view")


def test_reference_cannot_modify_database(tmp_path):
    spec = _spec()
    spec["reference_sql"] = "DELETE FROM foo"
    with pytest.raises(ValueError, match="exactly one SELECT"):
        load_spec(_write(tmp_path, spec))


def test_packaged_spec_parses_as_harbor_task(tmp_path):
    materialize_dbt_task(_write(tmp_path, _spec()))
    from harbor.models.task.config import TaskConfig
    import tomllib
    TaskConfig.model_validate(tomllib.loads((tmp_path / "task.toml").read_text()))


def test_runtime_failure_still_terminates_sandbox_without_model_secrets(tmp_path, monkeypatch):
    captured = {}
    class Sandbox:
        object_id = "sb-test"
        def open(self, *args):
            return io.StringIO()
        def exec(self, *args, **kwargs):
            raise RuntimeError("simulated runtime failure")
        def terminate(self):
            captured["terminated"] = True
    def create(*args, **kwargs):
        captured.update(kwargs)
        return Sandbox()
    fake_modal = SimpleNamespace(
        App=SimpleNamespace(lookup=lambda *args, **kwargs: "app"),
        Image=SimpleNamespace(from_registry=lambda *args: SimpleNamespace(entrypoint=lambda *args: "image")),
        Sandbox=SimpleNamespace(create=create),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    result = validate_dbt_task(_write(tmp_path, _spec()))
    assert result["status"] == "rejected" and result["sandbox_terminated"]
    assert captured["terminated"] and captured["timeout"] == 600 and captured["block_network"]
    assert "secrets" not in captured
    assert set(captured["env"]) == {"DB_TYPE", "DUCKDB_PATH", "DBT_SEND_ANONYMOUS_USAGE_STATS"}
    assert (tmp_path / "runtime_validation.json").exists()
