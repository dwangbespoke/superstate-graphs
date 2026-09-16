"""Validate a generated task in one bounded Modal sandbox, without model credentials."""
from pathlib import Path
import argparse
import json

from superstate_graphs.dbt_runtime import materialize_dbt_task, validate_dbt_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    if args.package_only:
        materialize_dbt_task(args.task_dir)
        print(json.dumps({"status": "awaiting_runtime_validation", "task_dir": str(args.task_dir)}))
        return
    receipt = validate_dbt_task(args.task_dir, timeout_seconds=args.timeout)
    print(json.dumps({key: receipt.get(key) for key in (
        "status", "errors", "reference", "sandbox_terminated", "spec_sha256")}))
    raise SystemExit(0 if receipt["status"] == "validated" else 1)


if __name__ == "__main__":
    main()
