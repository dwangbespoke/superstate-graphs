"""Offline inventory and frozen task selection for the shared-world pilot.

This module never starts containers, queries a model, reads credentials, or puts
verifier/reference-solution contents into the agent payload. Python 3.11+ only;
all dependencies are in the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENDOR = PROJECT_ROOT / "vendor/data-eng-bench"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/preflight_v1"
SCHEMA_VERSION = "hvs-shared-world-preflight-v1"
RESERVED_FAMILIES = ("fulfillment", "late_arrivals", "returns")
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coarse_family(task_id: str, tags: list[str]) -> str:
    """Outcome-blind, deliberately coarse labels requiring later human review."""
    name = task_id.lower()
    if any(word in name for word in ("timezone", "time-zone")):
        return "timezone"
    if any(word in name for word in ("late-arriving", "incremental")):
        return "late_arrivals"
    if any(word in name for word in ("fulfillment", "shipping", "carrier")):
        return "fulfillment"
    if any(word in name for word in ("refund", "product-return", "fix-product-metrics")):
        return "returns"
    if any(word in name for word in ("attribution", "campaign", "marketing", "channel")):
        return "marketing"
    if any(word in name for word in ("cohort", "churn", "retention")):
        return "retention"
    if any(word in name for word in ("inventory", "warehouse", "stock", "fifo", "lifo")):
        return "inventory"
    if any(word in name for word in ("customer", "rfm", "tier", "loyalty")):
        return "customer"
    if any(word in name for word in ("payment", "revenue", "balance", "receivable")):
        return "finance"
    if any(word in name for word in ("product", "basket", "price")):
        return "product"
    # This fallback is only descriptive; it does not assert semantic equivalence.
    return "other"


def competence_bucket(task: dict[str, Any]) -> str:
    name = task["task_id"]
    category = task["metadata"].get("category", "")
    tags = set(task["metadata"].get("tags", []))
    if "debugging" in tags or "bug-fix" in tags or "fix-" in name or category == "debugging":
        return "debugging"
    if tags.intersection({"data-modeling", "fact-table", "dimensional-modeling", "snapshot"}) or category in {"dbt-models", "dbt-snapshots"}:
        return "modeling"
    return "analytics"


def file_receipt(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"refusing symlink in source inventory: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def database_receipt(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.relative_to(root).as_posix(), "status": "missing"}
    receipt = file_receipt(path, root)
    with path.open("rb") as stream:
        header = stream.read(512)
    if header.startswith(LFS_SIGNATURE):
        pointer = dict(line.split(" ", 1) for line in header.decode().splitlines()[1:] if " " in line)
        receipt.update({"status": "git_lfs_pointer", "materialized": False,
                        "lfs_oid": pointer.get("oid"),
                        "expected_bytes": int(pointer["size"]) if pointer.get("size", "").isdigit() else None})
    else:
        magic = header[8:12] == b"DUCK"
        receipt.update({"status": "duckdb_file" if magic else "unrecognized_binary",
                        "materialized": magic, "duckdb_header_recognized": magic,
                        "database_opened_or_validated": False})
    return receipt


def source_revision(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                    text=True, timeout=15, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    revision = git("rev-parse", "HEAD")
    # Never emit git remotes: a remote URL can contain an embedded credential.
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": revision, "tracked_worktree_clean": status == "" if status is not None else None,
            "revision_pinned": bool(revision), "tracked_changes_present": bool(status)}


def runtime_inventory() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in ("modal", "gepa", "openai", "litellm", "datasets", "pandas", "duckdb", "harbor", "torch"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {"importable": importlib.util.find_spec(name) is not None, "version": version}
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "commands": {name: shutil.which(name) for name in ("uv", "python3", "git", "git-lfs", "docker", "modal", "harbor")},
        "packages": packages,
        "credential_inspection_performed": False,
        "network_or_inference_calls_performed": False,
    }


def inventory_tasks(root: Path) -> list[dict[str, Any]]:
    result = []
    for config in sorted((root / "tasks").glob("*/task.toml")):
        data = tomllib.loads(config.read_text())
        metadata = data.get("metadata", {})
        task_dir = config.parent
        instruction = task_dir / "instruction.md"
        if not instruction.is_file():
            raise ValueError(f"task lacks instruction.md: {task_dir.name}")
        # Explicit allowlist: author details and arbitrary config fields are not forwarded.
        public_metadata = {key: metadata[key] for key in ("difficulty", "category", "tags", "db_name") if key in metadata}
        task = {
            "task_id": task_dir.name,
            "metadata": public_metadata,
            "coarse_family": coarse_family(task_dir.name, list(metadata.get("tags", []))),
            "source_contract": {
                "task_version": data.get("version"),
                "task_toml_sha256": sha256_file(config),
                "environment": data.get("environment", {}),
                "agent_timeout_sec": data.get("agent", {}).get("timeout_sec"),
                "verifier_timeout_sec": data.get("verifier", {}).get("timeout_sec"),
                "files": [file_receipt(path, root) for path in sorted(task_dir.rglob("*")) if path.is_file()],
            },
            "agent_payload": {
                "task_id": task_dir.name,
                "instruction": instruction.read_text(),
                "instruction_sha256": sha256_file(instruction),
                "db_type": "duckdb",
            },
        }
        task["competence_bucket"] = competence_bucket(task)
        result.append(task)
    if not result:
        raise ValueError(f"no task.toml files under {root / 'tasks'}")
    return result


def select_tasks(tasks: list[dict[str, Any]], fast_ids: list[str], source_size: int = 30) -> dict[str, Any]:
    by_id = {task["task_id"]: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate task IDs")
    missing = sorted(set(fast_ids) - set(by_id))
    if missing:
        raise ValueError(f"official fast subset references missing tasks: {missing}")
    if len(set(fast_ids)) != len(fast_ids):
        raise ValueError("official fast subset contains duplicates")
    excluded = sorted(task["task_id"] for task in tasks if task["coarse_family"] == "timezone")
    # Reserve whole explicitly defined coarse families before inspecting outcomes.
    heldout = sorted(task["task_id"] for task in tasks if task["coarse_family"] in RESERVED_FAMILIES)
    reserved = set(excluded + heldout)
    eligible = [task for task in tasks if task["task_id"] not in reserved]
    easy = sorted(task["task_id"] for task in eligible if task["metadata"].get("difficulty") == "easy")
    selected_medium: dict[str, list[str]] = {}
    fast_set = set(fast_ids)
    for bucket in ("debugging", "modeling", "analytics"):
        pool = [task["task_id"] for task in eligible
                if task["metadata"].get("difficulty") == "medium" and task["competence_bucket"] == bucket]
        selected_medium[bucket] = sorted(pool, key=lambda name: (name not in fast_set, name))[:2]
    pilot = easy + [task_id for bucket in selected_medium.values() for task_id in bucket]
    source = [task_id for task_id in fast_ids if task_id not in reserved][:source_size]
    # Prefer pilot tasks when replacing excluded/reserved official-fast entries.
    candidates = pilot + sorted(task["task_id"] for task in eligible)
    for task_id in candidates:
        if len(source) >= source_size:
            break
        if task_id not in source:
            source.append(task_id)
    if len(source) != source_size:
        raise ValueError(f"requested {source_size} source tasks; only {len(source)} eligible tasks")
    exposed = sorted(set(pilot + source))
    return {
        "selection_uses_outcomes": False,
        "official_fast30": fast_ids,
        "competence_pilot": pilot,
        "competence_easy": easy,
        "competence_medium_by_bucket": selected_medium,
        "competence_pilot_complete": len(easy) > 0 and all(len(ids) == 2 for ids in selected_medium.values()),
        "source_tasks": source,
        "source_replacements": [task_id for task_id in source if task_id not in fast_set],
        "source_removed_from_fast30": [task_id for task_id in fast_ids if task_id not in source],
        "reserved_heldout_tasks": heldout,
        "reserved_coarse_families": list(RESERVED_FAMILIES),
        "excluded_timezone_tasks": excluded,
        "pilot_tasks_outside_source": sorted(set(pilot) - set(source)),
        "all_exposed_task_ids": exposed,
        "task_disjoint": not set(exposed).intersection(heldout),
        "coarse_family_disjoint": not {by_id[name]["coarse_family"] for name in exposed}.intersection(
            {by_id[name]["coarse_family"] for name in heldout}),
        "validated_family_disjoint": False,
        "family_audit_required": True,
        "split_scope": "Frozen task-level split; family names are an unvalidated heuristic, not a leakage guarantee.",
    }


def build_manifest(root: Path, source_size: int = 30) -> dict[str, Any]:
    root = root.resolve()
    fast_path = root / "configs/fast-30.txt"
    fast_ids = [line.strip() for line in fast_path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    tasks = inventory_tasks(root)
    selection = select_tasks(tasks, fast_ids, source_size)
    reference_paths = ("README.md", ".gitattributes", "configs/fast-30.txt", "base-image/Dockerfile",
                       "base-image/fix_data.py", "base-image/dbt_models_duckdb/dbt_project.yml",
                       "base-image/dbt_models_duckdb/profiles.yml")
    # Held-out tasks have metadata/hashes only; never distribute their instructions as rollout inputs.
    heldout = set(selection["reserved_heldout_tasks"])
    for task in tasks:
        if task["task_id"] in heldout:
            task.pop("agent_payload")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "snowflake-labs/data-eng-bench",
        "backend": "duckdb",
        "vendor_path": str(root),
        "source_revision": source_revision(root),
        "shared_source_files": [file_receipt(root / name, root) for name in reference_paths if (root / name).is_file()],
        "database": database_receipt(root / "base-image/database/retail.duckdb", root),
        "selection": selection,
        "tasks": tasks,
        "runtime": runtime_inventory(),
        "agent_payload_contract": {
            "allowed_keys": ["task_id", "instruction", "instruction_sha256", "db_type"],
            "includes_verifier_or_solution_bodies": False,
            "source_file_receipts_are_not_agent_input": True,
            "mount_only_task_environment_during_rollout": True,
            "verifier_and_solution_must_be_unavailable_until_evaluation": True,
        },
    }


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    selection = manifest["selection"]
    blockers = []
    if not manifest["source_revision"]["revision_pinned"]:
        blockers.append("Source revision has not been pinned to a Git commit.")
    if not manifest["database"].get("materialized"):
        blockers.append("DuckDB source is missing, a Git LFS pointer, or unrecognized; materialize and validate it before rollout.")
    if not manifest["runtime"]["commands"].get("docker"):
        blockers.append("Local Docker CLI is unavailable; use an explicitly configured remote sandbox or install a local runtime.")
    return {
        "schema_version": SCHEMA_VERSION,
        "commit": manifest["source_revision"]["commit"],
        "task_count": len(manifest["tasks"]),
        "difficulty_counts": dict(sorted(Counter(task["metadata"].get("difficulty", "unknown") for task in manifest["tasks"]).items())),
        "pilot_count": len(selection["competence_pilot"]),
        "pilot_ids": selection["competence_pilot"],
        "source_count": len(selection["source_tasks"]),
        "heldout_count": len(selection["reserved_heldout_tasks"]),
        "heldout_ids": selection["reserved_heldout_tasks"],
        "excluded_timezone_ids": selection["excluded_timezone_tasks"],
        "task_disjoint": selection["task_disjoint"],
        "coarse_family_disjoint": selection["coarse_family_disjoint"],
        "validated_family_disjoint": False,
        "family_audit_required": True,
        "database_status": manifest["database"]["status"],
        "launch_blockers": blockers,
        "jobs_launched": 0,
        "inference_calls": 0,
        "readiness_is_not_launch_authorization": True,
    }


def write_preflight(manifest: dict[str, Any], output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    names = ("manifest.json", "summary.json", "notes.md")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(manifest)
    notes = "\n".join([
        "# Shared-world pilot: offline preflight",
        "",
        "No model calls, paid jobs, network requests, environment execution, or credential reads were performed by this script.",
        "",
        f"Source commit: `{summary['commit']}`. {summary['task_count']} tasks inventoried; "
        f"{summary['pilot_count']} competence tasks; {summary['source_count']} source tasks; {summary['heldout_count']} reserved tasks.",
        "",
        "The pilot uses all eligible easy tasks plus two medium tasks in each of debugging, modeling, and analytics. "
        "Official fast-subset membership, then lexical order, break ties. Buckets are metadata/name heuristics.",
        "",
        "The source selection preserves official fast-30 order, removes reserved and timezone tasks, and fills vacancies "
        "from pilot tasks, then remaining task IDs in lexical order. Pilot tasks outside the source selection remain "
        "development exposure and must never be reclassified as held out.",
        "",
        "Fulfillment, late-arrival, and returns families were reserved before any outcomes. The task-ID split is "
        "disjoint and the implementation's coarse labels are disjoint. These labels are NOT a validated semantic "
        "family split: related SQL skills, source tables, shared project files, and task variants can overlap. "
        "Review family and solution/template similarity before claiming transfer to unseen task families.",
        "",
        "Agent payloads contain only the task instruction, ID, instruction hash, and backend selector. "
        "Hidden tests, expected outputs, reference solutions, and their file receipts are not agent inputs. "
        "A future runner must mount only the task environment and must not expose the vendor tree, tests, or solution "
        "directory during collection. Reserved tasks omit agent payloads entirely.",
        "",
        "The source database receipt distinguishes an LFS pointer from a DuckDB header. Header recognition is not "
        "a database integrity or query smoke test. A materialized DB must still be opened, checked, and hashed.",
        "",
        "Blockers / remaining setup:",
        *[f"- {item}" for item in summary["launch_blockers"]],
        "- Freeze model, sampling configuration, sandbox limits, and an explicit runtime/token budget before collection.",
        "- Task family audit is required; no learning, GEPA optimization, or training result is claimed by this preflight.",
        "",
    ])
    payloads = {"manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                "summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n", "notes.md": notes}
    for name in names:
        (output / name).write_text(payloads[name])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-size", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source_size < 1:
        parser.error("--source-size must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error(f"refusing to overwrite nonempty output directory: {args.output}")
    manifest = build_manifest(args.vendor, args.source_size)
    print(json.dumps(write_preflight(manifest, args.output, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
