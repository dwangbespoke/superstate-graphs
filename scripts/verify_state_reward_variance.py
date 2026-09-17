#!/usr/bin/env python3
"""Independently verify recorded state reward statistics without model calls.

Recompute all outcome weightings, manual-grade sensitivity, Bessel corrections,
task decomposition, and task-cluster bootstrap intervals. This checks the
recorded analysis; it never invokes the production outcome-analysis function.
Only counts, hashes, and verification metadata enter the output receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WEIGHTINGS = ("history_weighted", "trajectory_deduplicated", "task_balanced")
BASE_SEED = 20260917


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])


def exporter_module():
    """Reuse independent publication checks, not the production statistics generator."""
    path = Path(__file__).with_name("export_full_graph_artifacts.py")
    spec = importlib.util.spec_from_file_location("variance_verification_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_variance(run: Path, corpus: Path) -> dict:
    source_hashes = {}

    def read(name: str, *, from_corpus=False, jsonl=False):
        raw = ((corpus if from_corpus else run) / name).read_bytes()
        source_hashes[("corpus/" if from_corpus else "run/") + name] = sha(raw)
        return [json.loads(line) for line in raw.splitlines()] if jsonl else json.loads(raw)

    manifest = read("manifest.json", from_corpus=True)
    history_rows = read("histories.jsonl", from_corpus=True, jsonl=True)
    require(
        source_hashes["corpus/histories.jsonl"] == manifest["artifact_sha256"]["histories.jsonl"],
        "Immutable history index hash mismatch",
    )
    histories = {row["history_id"]: row for row in history_rows}
    require(
        len(histories) == len(history_rows) == manifest["statistics"]["histories"],
        "History census is duplicated or incomplete",
    )
    assignments = read("assignments_all.json")
    require(set(assignments) == set(histories), "Assignment/history census mismatch")
    for hid, row in assignments.items():
        require(
            isinstance(row.get("state_id"), str) and bool(row["state_id"]),
            "Unclassified history in final variance input",
        )
        require(
            all(
                row.get(key) == histories[hid][key]
                for key in ("history_id", "task_id", "rollout_id", "split", "step")
            ),
            "Assignment identity does not match immutable history index",
        )
    full = read("state_reward_variance.json")
    by_split = read("state_reward_variance_by_split.json")
    require(
        set(by_split) == {h["split"] for h in histories.values()}, "Split variance census mismatch"
    )
    analyses = {"all": full, **by_split}
    exporter = exporter_module()
    report = exporter.report_builder()
    checks, max_error, computed_intervals, skipped_intervals = 0, 0.0, 0, 0

    def close(actual: Any, expected: float | None, label: str):
        nonlocal checks, max_error
        checks += 1
        if expected is None:
            require(actual is None, "Expected unavailable statistic: " + label)
            return
        require(
            type(actual) in (float, int) and math.isfinite(actual),
            "Invalid numeric statistic: " + label,
        )
        error = abs(actual - expected)
        max_error = max(max_error, error)
        require(error < 1e-10, "Numeric statistic mismatch: " + label)

    scope_receipts = {}
    for scope, analysis in analyses.items():
        subset = {hid: h for hid, h in histories.items() if scope == "all" or h["split"] == scope}
        subset_assignments = {hid: assignments[hid] for hid in subset}
        require(
            report.variance_census_verified(analysis, subset_assignments),
            "Occupied-state membership census mismatch: " + scope,
        )
        exporter.validate_variance_moments(analysis, subset, subset_assignments)
        members = defaultdict(list)
        for hid, history in subset.items():
            members[assignments[hid]["state_id"]].append(history)
        require(
            {s["state_id"] for s in analysis["states"]} == set(members),
            "Occupied-state variance rows mismatch",
        )
        for state in analysis["states"]:
            sid = state["state_id"]
            rows = members[sid]
            for exclude, record in (
                (False, state),
                (True, state["sensitivity_excluding_manual_rewards"]),
            ):
                visits = Counter(h["rollout_id"] for h in rows)
                require(
                    record["member_histories"] == len(rows)
                    and record["distinct_visiting_rollouts"] == len(visits)
                    and record["distinct_visiting_tasks"] == len({h["task_id"] for h in rows}),
                    "Variance or sensitivity membership metadata mismatch",
                )
                by_rollout = {}
                for history in rows:
                    rid = history["rollout_id"]
                    entry = (
                        history["task_id"],
                        history["reward"],
                        history["reward_provenance"].get("kind", ""),
                    )
                    require(
                        rid not in by_rollout or by_rollout[rid] == entry,
                        "Inconsistent terminal outcome within one rollout",
                    )
                    by_rollout[rid] = entry
                by_task = defaultdict(list)
                manual = []
                missing = []
                for rid, (task, reward, kind) in by_rollout.items():
                    if str(kind).startswith("manual"):
                        manual.append(rid)
                        if exclude:
                            continue
                    if reward is None:
                        missing.append(rid)
                        continue
                    require(
                        type(reward) in (float, int) and math.isfinite(reward),
                        "Invalid terminal outcome",
                    )
                    by_task[task].append((reward, visits[rid]))
                tasks = sorted(by_task)
                n = sum(len(v) for v in by_task.values())
                require(
                    record["binary_rewards_only"]
                    is (
                        bool(by_task)
                        and all(
                            reward in (0, 1) for values in by_task.values() for reward, _ in values
                        )
                    ),
                    "Binary-reward metadata mismatch",
                )
                require(
                    record["rewarded_rollouts"] == n and record["rewarded_tasks"] == len(tasks),
                    "Rewarded rollout/task count mismatch",
                )
                require(
                    record["manual_reward_rollouts"] == sorted(manual),
                    "Manual-grade provenance census mismatch",
                )
                require(
                    record["excluded_manual_reward_rollouts"]
                    == (sorted(manual) if exclude else []),
                    "Manual-grade exclusion census mismatch",
                )
                require(
                    record["missing_reward_rollouts"] == sorted(missing),
                    "Missing-outcome census mismatch",
                )
                variance = record["trajectory_deduplicated"]["population_variance"]
                close(
                    record["trajectory_deduplicated"]["bessel_corrected_descriptive_variance"],
                    variance * n / (n - 1) if n > 1 else None,
                    "Bessel correction",
                )
                task_means = [math.fsum(r for r, _ in by_task[t]) / len(by_task[t]) for t in tasks]
                task_second = [
                    math.fsum(r * r for r, _ in by_task[t]) / len(by_task[t]) for t in tasks
                ]
                within = (
                    math.fsum(
                        max(0.0, second - average * average)
                        for average, second in zip(task_means, task_second)
                    )
                    / len(tasks)
                    if tasks
                    else None
                )
                task_mean = math.fsum(task_means) / len(tasks) if tasks else None
                between = (
                    math.fsum((p - task_mean) ** 2 for p in task_means) / len(tasks)
                    if tasks
                    else None
                )
                close(
                    record["task_balanced_decomposition"]["mean_within_task_population_variance"],
                    within,
                    "Within-task decomposition",
                )
                close(
                    record["task_balanced_decomposition"]["between_task_mean_variance"],
                    between,
                    "Between-task decomposition",
                )
                for offset, weighting in enumerate(WEIGHTINGS):
                    groups = []
                    for task in tasks:
                        values = by_task[task]
                        if weighting == "history_weighted":
                            groups.append(
                                (
                                    sum(count for _, count in values),
                                    math.fsum(reward * count for reward, count in values),
                                    math.fsum(reward * reward * count for reward, count in values),
                                )
                            )
                        elif weighting == "trajectory_deduplicated":
                            groups.append(
                                (
                                    len(values),
                                    math.fsum(reward for reward, _ in values),
                                    math.fsum(reward * reward for reward, _ in values),
                                )
                            )
                        else:
                            groups.append(
                                (
                                    1,
                                    math.fsum(reward for reward, _ in values) / len(values),
                                    math.fsum(reward * reward for reward, _ in values)
                                    / len(values),
                                )
                            )
                    interval = record[weighting]["task_cluster_bootstrap"]
                    draws = 1000 if scope == "all" else 500
                    require(
                        interval["method"]
                        == "percentile bootstrap of task families, retaining within-task outcomes",
                        "Unknown interval method",
                    )
                    require(
                        interval["requested_draws"] == draws
                        and interval["task_clusters"] == len(groups)
                        and interval["confidence"] == 0.95,
                        "Bootstrap parameters differ from recorded run protocol",
                    )
                    if len(groups) < 2:
                        require(
                            interval["status"] == "insufficient_task_clusters"
                            and interval["mean_interval"] is None
                            and interval["population_variance_interval"] is None,
                            "Insufficient task clusters must not have bootstrap intervals",
                        )
                        skipped_intervals += 1
                        continue
                    require(
                        interval["status"] == "computed" and interval["completed_draws"] == draws,
                        "Bootstrap completion metadata mismatch",
                    )
                    state_seed = int(
                        sha(
                            json.dumps(
                                sid, sort_keys=True, ensure_ascii=False, allow_nan=False
                            ).encode()
                        )[:8],
                        16,
                    )
                    rng = random.Random(BASE_SEED + state_seed + offset)
                    means, variances = [], []
                    for _ in range(draws):
                        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
                        weight = math.fsum(g[0] for g in sampled)
                        average = math.fsum(g[1] for g in sampled) / weight
                        second = math.fsum(g[2] for g in sampled) / weight
                        means.append(average)
                        variances.append(max(0, second - average * average))
                    for key, values in (
                        ("mean_interval", means),
                        ("population_variance_interval", variances),
                    ):
                        require(
                            isinstance(interval[key], list) and len(interval[key]) == 2,
                            "Invalid interval endpoints",
                        )
                        for actual, probability in zip(
                            interval[key], ((1 - 0.95) / 2, 1 - (1 - 0.95) / 2)
                        ):
                            close(actual, percentile(values, probability), weighting + "/" + key)
                    computed_intervals += 1
        scope_receipts[scope] = {
            "histories": len(subset),
            "occupied_states": len(members),
            "moment_and_membership_validation": True,
        }
    for name, fingerprint in source_hashes.items():
        root, filename = name.split("/", 1)
        require(
            sha(((corpus if root == "corpus" else run) / filename).read_bytes()) == fingerprint,
            "Artifact changed during verification",
        )
    return {
        "format": "independent-state-variance-verification-v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scopes": scope_receipts,
        "history_trajectory_and_equal_task_moments_recomputed": True,
        "manual_excluded_sensitivity_recomputed": True,
        "bessel_correction_and_task_decomposition_recomputed": True,
        "task_cluster_intervals_recomputed": computed_intervals,
        "intervals_without_sufficient_task_clusters": skipped_intervals,
        "additional_numeric_checks": checks,
        "max_absolute_error": max_error,
        "source_artifact_sha256": source_hashes,
        "verification_script_sha256": sha(Path(__file__).read_bytes()),
        "verification_dependency_sha256": {
            name: sha(Path(__file__).with_name(name).read_bytes())
            for name in ("export_full_graph_artifacts.py", "build_full_graph_report.py")
        },
        "limitations": [
            "Descriptive terminal outcomes of visiting trajectories; not identical-history continuation variance or causal difficulty.",
            "Task-cluster bootstrap conditions on learned assignments and recorded outcomes; graph selection and policy uncertainty are excluded.",
        ],
        "model_calls": 0,
        "formation_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", "--dataset", dest="corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = verify_variance(args.run, args.corpus)
    except ValueError as error:
        parser.exit(1, f"Verification failed: {error}. No new receipt written.\n")
    except (KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Verification failed ({type(error).__name__}); no new receipt written.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, args.output)
    print(
        f"Verified {receipt['scopes']['all']['occupied_states']} occupied states and {receipt['task_cluster_intervals_recomputed']} task-cluster interval pairs. Receipt: {args.output}"
    )


if __name__ == "__main__":
    main()
