"""Offline split and leakage checks for the shared-world pilot manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superstate_graphs.prepare import build_manifest, database_receipt, write_preflight


def make_vendor(tmp_path: Path) -> Path:
    root = tmp_path / "vendor"
    definitions = {
        "easy-orders": ("easy", "analytics", ["aggregation"]),
        "dbt-fix-alpha": ("medium", "debugging", ["debugging"]),
        "dbt-fix-beta": ("medium", "debugging", ["debugging"]),
        "dbt-model-alpha": ("medium", "dbt-models", ["data-modeling"]),
        "dbt-model-beta": ("medium", "dbt-models", ["data-modeling"]),
        "dbt-analytics-alpha": ("medium", "analytics", ["analytics"]),
        "dbt-analytics-beta": ("medium", "analytics", ["analytics"]),
        "dbt-fulfillment-sla": ("medium", "analytics", ["logistics"]),
        "dbt-product-return-analysis": ("medium", "analytics", ["returns"]),
        "dbt-incremental-sales": ("hard", "data-engineering", ["incremental"]),
        "dbt-fix-timezone-sales": ("medium", "debugging", ["timezone"]),
    }
    for task_id, (difficulty, category, tags) in definitions.items():
        task = root / "tasks" / task_id
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            f'version = "1.0"\n[metadata]\ndifficulty = "{difficulty}"\n'
            f'category = "{category}"\ntags = {json.dumps(tags)}\n'
        )
        (task / "instruction.md").write_text(f"Public instruction for {task_id}.")
        (task / "tests").mkdir()
        (task / "tests/expected.txt").write_text("HIDDEN_EXPECTED_ANSWER_SENTINEL")
        (task / "solution").mkdir()
        (task / "solution/solve.sh").write_text("REFERENCE_SOLUTION_SENTINEL")
    (root / "configs").mkdir()
    (root / "configs/fast-30.txt").write_text("\n".join(definitions) + "\n")
    (root / "base-image/database").mkdir(parents=True)
    (root / "base-image/database/retail.duckdb").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 100000\n"
    )
    return root


def test_agent_payload_never_contains_reference_or_verifier(tmp_path: Path) -> None:
    manifest = build_manifest(make_vendor(tmp_path), source_size=7)
    encoded = json.dumps(manifest)
    assert "HIDDEN_EXPECTED_ANSWER_SENTINEL" not in encoded
    assert "REFERENCE_SOLUTION_SENTINEL" not in encoded
    heldout = set(manifest["selection"]["reserved_heldout_tasks"])
    for task in manifest["tasks"]:
        if task["task_id"] in heldout:
            assert "agent_payload" not in task
        else:
            assert set(task["agent_payload"]) == {"task_id", "instruction", "instruction_sha256", "db_type"}
    assert any("tests/expected.txt" in receipt["path"] for receipt in manifest["tasks"][0]["source_contract"]["files"])


def test_split_is_deterministic_disjoint_and_honest(tmp_path: Path) -> None:
    root = make_vendor(tmp_path)
    selection = build_manifest(root, source_size=7)["selection"]
    assert selection == build_manifest(root, source_size=7)["selection"]
    assert len(selection["source_tasks"]) == 7
    assert len(selection["competence_pilot"]) == 7
    assert all(len(ids) == 2 for ids in selection["competence_medium_by_bucket"].values())
    assert selection["task_disjoint"] and selection["coarse_family_disjoint"]
    assert not set(selection["all_exposed_task_ids"]) & set(selection["reserved_heldout_tasks"])
    assert "dbt-fix-timezone-sales" not in selection["all_exposed_task_ids"]
    assert selection["family_audit_required"]
    assert selection["validated_family_disjoint"] is False


def test_lfs_is_not_a_materialized_database(tmp_path: Path) -> None:
    root = make_vendor(tmp_path)
    receipt = database_receipt(root / "base-image/database/retail.duckdb", root)
    assert receipt["status"] == "git_lfs_pointer"
    assert receipt["materialized"] is False
    assert receipt["expected_bytes"] == 100000


def test_no_overwrite_without_explicit_flag(tmp_path: Path) -> None:
    manifest = build_manifest(make_vendor(tmp_path), source_size=7)
    output = tmp_path / "preflight"
    write_preflight(manifest, output)
    previous = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_preflight(manifest, output)
    assert (output / "manifest.json").read_bytes() == previous
    write_preflight(manifest, output, overwrite=True)
    assert (output / "manifest.json").read_bytes() == previous
