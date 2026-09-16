"""Prepare isolated Harbor task copies and bounded experiment configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE_IMAGE = "ghcr.io/snowflake-labs/data-eng-bench-base@sha256:ef92b6ef197a89ff1d8b371aaf5de19343003abc462991ddeedb1b1005e2e04b"


def stage_tasks(ids: list[str]) -> Path:
    source = ROOT / "vendor/data-eng-bench/tasks"
    dest = ROOT / "data/tasks"
    dest.mkdir(parents=True, exist_ok=True)
    receipts = []
    for task_id in ids:
        target = dest / task_id
        if target.exists():
            continue
        shutil.copytree(source / task_id, target)
        compose = target / "environment/docker-compose.yaml"
        if compose.exists():
            parsed = yaml.safe_load(compose.read_text())
            if set(parsed["services"]) != {"main"}:
                raise ValueError(f"Task needs multiple services: {task_id}")
            compose.unlink()
        dockerfile = target / "environment/Dockerfile"
        dockerfile.write_text(dockerfile.read_text().replace("ghcr.io/snowflake-labs/data-eng-bench-base:1.0.0", BASE_IMAGE))
        # All task-specific COPY/RUN instructions, task instructions, oracles,
        # and verifiers are unchanged. Modal runs the single main container.
        toml = target / "task.toml"
        text = toml.read_text().replace("[environment]\n", '[environment]\nallow_internet = false\n')
        toml.write_text(text)
        receipts.append({"task_id": task_id, "adaptations": ["single-container Modal instead of redundant main-only compose", "pin public base image by digest", "disable sandbox internet"],
                         "instruction_sha256": hashlib.sha256((target / "instruction.md").read_bytes()).hexdigest()})
    (dest.parent / "adaptations.json").write_text(json.dumps(receipts, indent=2))
    return dest


def config(ids: list[str], name: str, attempts: int, oracle: bool, endpoint: dict | None = None) -> dict:
    endpoint = endpoint or {"api_base": "https://pending.invalid/v1", "model": "Qwen/Qwen3.5-9B"}
    agent = {"name": "oracle", "override_timeout_sec": 900} if oracle else {
        "import_path": "superstate_graphs.recorder:RecordingTerminus2",
        "model_name": "hosted_vllm/" + endpoint["model"],
        "override_timeout_sec": 1200,
        "kwargs": {"max_turns": 30, "temperature": 0.6, "api_base": endpoint["api_base"],
                   "enable_summarize": False, "collect_rollout_details": True,
                   "store_all_messages": True, "trajectory_config": {"raw_content": True, "linear_history": True},
                   "model_info": {"max_input_tokens": 45056, "max_output_tokens": 4096, "max_tokens": 49152,
                                  "input_cost_per_token": 0.0, "output_cost_per_token": 0.0, "litellm_provider": "hosted_vllm", "mode": "chat"},
                   "llm_call_kwargs": {"max_tokens": 4096, "top_p": 0.95,
                                       "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20}}}}
    return {"job_name": name, "jobs_dir": str(ROOT / "results/rollouts"), "n_attempts": attempts,
            "orchestrator": {"type": "local", "n_concurrent_trials": 2},
            "environment": {"type": "modal", "delete": True,
                            "env": {"DB_TYPE": "duckdb", "DUCKDB_PATH": "/app/database/retail.duckdb"},
                            "kwargs": {"app_name": "superstate-graphs-environments", "sandbox_timeout_secs": 2400, "sandbox_idle_timeout_secs": 300}},
            "verifier": {"override_timeout_sec": 600},
            "agents": [agent], "datasets": [{"path": str(ROOT / "data/tasks"), "task_names": ids}]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "results/preflight_v1/manifest.json")
    p.add_argument("--endpoint", type=Path)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selection = manifest["selection"]
    pilot = selection["competence_pilot"]
    source = selection["source_tasks"]
    stage_tasks(sorted(set(pilot + source)))
    endpoint = json.loads(args.endpoint.read_text()) if args.endpoint else None
    out = ROOT / "configs/generated"
    out.mkdir(parents=True, exist_ok=True)
    for name, ids, attempts, oracle in [("oracle-smoke", pilot[:1], 1, True), ("pilot", pilot[:6], 2, False), ("source-rollouts", source[:18], 3, False)]:
        cfg = config(ids, name, attempts, oracle, endpoint)
        from harbor.models.job.config import JobConfig
        JobConfig.model_validate(cfg)
        (out / f"{name}.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({"configs": str(out), "pilot_tasks": pilot[:6], "source_tasks": source[:18]}))


if __name__ == "__main__":
    main()
