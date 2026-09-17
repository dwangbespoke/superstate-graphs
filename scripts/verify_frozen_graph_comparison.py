#!/usr/bin/env python3
"""Verify frozen graph comparison artifacts offline, without model or endpoint access.

Recompute immutable test identities, per-rollout scores, paired means, and the
recorded original-task cluster bootstrap. The receipt contains hashes and
aggregate evidence only. Source histories and judge feedback are never emitted.
The installed frozen runtime rules must match the evaluated run; mismatches fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from superstate_graphs.graph_evolution import GraphRuntime, validate_spec


COMPONENTS = (
    "score",
    "history_coverage",
    "transition_coverage",
    "coherence",
    "outgoing_applicability",
    "complexity",
)
SOURCE_FILES = (
    "heldout_comparison.json",
    "heldout_test.json",
    "baseline_heldout.json",
    "seed_candidate.json",
    "optimized_candidate.json",
    "gepa_result.json",
    "run_contract.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mean(values) -> float:
    values = list(values)
    require(bool(values), "A required measurement group is empty")
    return math.fsum(values) / len(values)


def public_models(contract: dict) -> tuple[dict, dict]:
    """Use only public recorded identities, never local serving configuration."""
    models = []
    for prefix in ("", "teacher_"):
        public = {key: contract.get(prefix + key) for key in ("model", "revision")}
        require(
            all(isinstance(value, str) and value for value in public.values()),
            "Run contract lacks a public model name or pinned revision",
        )
        models.append(public)
    return tuple(models)


def verify_comparison(run_dir: Path, corpus_dir: Path) -> dict:
    """Return a modest successful-verification receipt, or raise on any mismatch."""
    sources = {name: read_json(run_dir / name) for name in SOURCE_FILES}
    source_hashes = {name: file_digest(run_dir / name) for name in SOURCE_FILES}
    comparison = sources["heldout_comparison.json"]
    seed, selected = sources["seed_candidate.json"], sources["optimized_candidate.json"]
    result = sources["gepa_result.json"]
    candidates, scores = result["candidates"], result["val_aggregate_scores"]
    require(bool(candidates) and len(candidates) == len(scores), "Candidate/score census mismatch")
    require(
        all(type(value) in (int, float) and math.isfinite(value) for value in scores),
        "GEPA aggregate score is not finite",
    )
    best = result["best_idx"]
    require(type(best) is int and 0 <= best < len(candidates), "Invalid selected candidate index")
    require(
        best == max(range(len(scores)), key=scores.__getitem__),
        "Selected index is not first maximum",
    )
    require(
        seed == candidates[0] and selected == candidates[best],
        "Candidate specifications mismatch GEPA indices",
    )

    manifest = read_json(corpus_dir / "manifest.json")
    corpus_hashes = {}
    for name in ("rollouts.jsonl", "splits.json"):
        corpus_hashes[name] = file_digest(corpus_dir / name)
        require(
            corpus_hashes[name] == manifest.get("artifact_sha256", {}).get(name),
            "Immutable corpus artifact hash mismatch: " + name,
        )
    splits = read_json(corpus_dir / "splits.json")
    expected_order = splits["test"]["rollout_ids"]
    expected_ids = set(expected_order)
    require(
        bool(expected_ids) and len(expected_ids) == len(expected_order),
        "Invalid test rollout census",
    )
    original = {}
    with (corpus_dir / "rollouts.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in expected_ids:
                require(row["id"] not in original, "Duplicate test rollout in corpus")
                original[row["id"]] = row
    require(set(original) == expected_ids, "Corpus does not contain the exact test split")
    by_task = defaultdict(list)
    for rid, rollout in original.items():
        require(rollout["split"] == "test", "Test rollout split is incorrect")
        require(
            rollout["history_count"] == len(rollout["steps"]) + 1, "History/step census mismatch"
        )
        by_task[rollout["task_id"]].append(rid)
    tasks = sorted(by_task)
    require(set(tasks) == set(splits["test"]["task_ids"]), "Test task census mismatch")
    require(
        comparison["common_rollout_ids"] == sorted(expected_ids), "Paired rollout census mismatch"
    )
    require(comparison["rollouts"] == len(original), "Comparison rollout count mismatch")
    require(comparison["original_tasks"] == len(tasks), "Comparison task count mismatch")
    require(
        comparison["selected_equals_seed"] is (selected == seed), "Candidate equality flag mismatch"
    )
    require(
        comparison["optimization_uses_test_feedback"] is False,
        "Test-feedback exclusion is not declared",
    )
    require(
        comparison["before_transductive_completion"] is True,
        "Pre-completion evaluation is not declared",
    )
    router, teacher = public_models(sources["run_contract.json"])
    # These namespaces have no client, URL, key, or callable inference method.
    runtime = GraphRuntime(
        SimpleNamespace(runtime=router), run_dir, teacher_llm=SimpleNamespace(runtime=teacher)
    )
    numeric_checks, max_error = 0, 0.0

    def close(actual, expected, label):
        nonlocal numeric_checks, max_error
        require(
            type(actual) in (int, float) and math.isfinite(actual),
            "Nonfinite measurement: " + label,
        )
        error = abs(actual - expected)
        require(error < 1e-12, "Numeric mismatch: " + label)
        numeric_checks += 1
        max_error = max(max_error, error)

    evaluations = {}
    for label, filename, candidate in (
        ("seed", "baseline_heldout.json", seed),
        ("selected", "heldout_test.json", selected),
    ):
        data = sources[filename]
        rows = data["rollouts"]
        require(len(rows) == len(original), "Evaluation rollout count mismatch")
        indexed = {row["rollout_id"]: row for row in rows}
        require(
            set(indexed) == expected_ids and len(indexed) == len(rows),
            "Evaluation rollout census mismatch",
        )
        require(
            data["candidate_hash"] == digest(candidate) == comparison[label + "_candidate_hash"],
            "Candidate hash mismatch",
        )
        graph = validate_spec(candidate)
        provenances = []
        for rid, rollout in original.items():
            row = indexed[rid]
            provenance = runtime._cache_provenance(candidate, rollout, judge=True)
            runtime._validate_evaluation(row, graph, rollout, candidate, provenance)
            provenances.append(provenance)
            n = rollout["history_count"]
            metrics = row["metrics"]
            for name in COMPONENTS[1:]:
                value = metrics[name]
                require(
                    type(value) in (int, float) and math.isfinite(value),
                    "Invalid component measurement",
                )
                require(
                    value >= 0 and (name == "complexity" or value <= 1),
                    "Component outside its valid range",
                )
            close(
                metrics["history_coverage"],
                sum(row["membership_supported"]) / n,
                "history coverage",
            )
            close(
                metrics["transition_coverage"],
                sum(t["supported"] for t in row["transitions"]) / max(n - 1, 1),
                "transition coverage",
            )
            close(metrics["complexity"], len(canonical(graph)) / 100000, "specification complexity")
            expected_score = (
                0.25 * metrics["history_coverage"]
                + 0.50 * metrics["transition_coverage"]
                + 0.15 * metrics["coherence"]
                + 0.10 * metrics["outgoing_applicability"]
                - 0.03 * metrics["complexity"]
            )
            close(row["score"], expected_score, "weighted rollout score")
        identity = digest({"candidate": candidate, "inputs": provenances})
        require(
            data["evaluation_identity"] == identity == comparison[label + "_evaluation_identity"],
            "Ordered evaluation identity mismatch",
        )
        close(
            data["summary"]["mean_score"],
            mean(row["score"] for row in rows),
            "evaluation summary score",
        )
        require(
            data["summary"]["rollouts"] == len(original)
            and data["summary"]["original_tasks"] == len(tasks),
            "Evaluation summary census mismatch",
        )
        evaluations[label] = indexed

    state_changed, edge_changed = (
        seed[key] != selected[key] for key in ("state_spec", "edge_spec")
    )
    assignments_identical = all(
        evaluations["seed"][rid]["assignments"] == evaluations["selected"][rid]["assignments"]
        for rid in original
    )
    if not state_changed:
        require(
            assignments_identical,
            "Identical state specification has differing cached routing assignments",
        )
    change_scope = (
        "both"
        if state_changed and edge_changed
        else "state_only"
        if state_changed
        else "edge_only"
        if edge_changed
        else "unchanged"
    )
    require(set(comparison["per_task"]) == set(tasks), "Paired per-task census mismatch")
    require(set(comparison["components"]) == set(COMPONENTS), "Paired component census mismatch")
    differences = {name: {} for name in COMPONENTS}

    def value(label, rid, name):
        row = evaluations[label][rid]
        return row["score"] if name == "score" else row["metrics"][name]

    for name in COMPONENTS:
        aggregate = comparison["components"][name]
        averages = {
            label: mean(value(label, rid, name) for rid in original) for label in evaluations
        }
        for label in evaluations:
            close(aggregate[label + "_mean"], averages[label], name + " aggregate mean")
            if name not in ("score", "complexity"):
                filename = "baseline_heldout.json" if label == "seed" else "heldout_test.json"
                close(
                    sources[filename]["summary"]["metrics"][name],
                    averages[label],
                    name + " summary mean",
                )
        close(
            aggregate["mean_difference"],
            averages["selected"] - averages["seed"],
            name + " mean difference",
        )
        task_means = {label: [] for label in evaluations}
        for task, ids in by_task.items():
            row = comparison["per_task"][task]
            require(row["rollouts"] == len(ids), "Per-task rollout count mismatch")
            per = {label: mean(value(label, rid, name) for rid in ids) for label in evaluations}
            for label in evaluations:
                close(row["components"][name][label + "_mean"], per[label], name + " task mean")
                task_means[label].append(per[label])
            difference = per["selected"] - per["seed"]
            differences[name][task] = difference
            close(row["components"][name]["difference"], difference, name + " task difference")
        for label in evaluations:
            close(
                aggregate[label + "_task_balanced_mean"],
                mean(task_means[label]),
                name + " task-balanced mean",
            )
        close(
            aggregate["task_balanced_difference"],
            mean(task_means["selected"]) - mean(task_means["seed"]),
            name + " task-balanced difference",
        )
        require(
            aggregate["favorable_direction"] == ("lower" if name == "complexity" else "higher"),
            "Component favorable direction mismatch",
        )

    bootstrap = comparison["bootstrap"]
    require(
        bootstrap["method"] == "paired original-task cluster percentile bootstrap",
        "Unknown bootstrap method",
    )
    require(
        bootstrap["estimand"]
        == "equal-task mean of selected-minus-seed paired rollout differences",
        "Unexpected bootstrap estimand",
    )
    require(
        bootstrap["confidence"] == 0.95 and bootstrap["resampling_units"] == len(tasks),
        "Bootstrap confidence or unit mismatch",
    )
    require(
        type(bootstrap["seed"]) is int and type(bootstrap["draws"]) is int,
        "Invalid bootstrap parameters",
    )
    require(
        bootstrap["equal_rollout_counts_per_task"]
        is (len({len(ids) for ids in by_task.values()}) == 1),
        "Bootstrap weighting flag mismatch",
    )
    require(
        bootstrap["draws"] > 0 if len(tasks) >= 2 else bootstrap["draws"] == 0,
        "Invalid bootstrap draw count",
    )
    rng = random.Random(bootstrap["seed"])
    replicates = {name: [] for name in COMPONENTS}
    for _ in range(bootstrap["draws"]):
        chosen = rng.choices(tasks, k=len(tasks))
        for name in COMPONENTS:
            replicates[name].append(mean(differences[name][task] for task in chosen))
    for name in COMPONENTS:
        interval = comparison["components"][name]["paired_task_bootstrap_95ci"]
        ordered = sorted(replicates[name])
        if not ordered:
            require(interval is None, "Single-task comparison must not report a bootstrap interval")
            continue
        require(isinstance(interval, list) and len(interval) == 2, "Invalid bootstrap interval")
        for quantile, actual in zip((0.025, 0.975), interval):
            position = quantile * (len(ordered) - 1)
            lower, fraction = int(position), position % 1
            expected = ordered[lower] + fraction * (
                ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
            )
            close(actual, expected, name + " bootstrap interval")

    # Refuse to sign a receipt if an input changed during this read-only check.
    require(
        all(file_digest(run_dir / name) == value for name, value in source_hashes.items()),
        "Evaluation artifact changed during verification",
    )
    return {
        "format": "independent-frozen-comparison-verification-v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "rollouts": len(original),
        "original_task_clusters": len(tasks),
        "histories": sum(row["history_count"] for row in original.values()),
        "numeric_comparisons": numeric_checks,
        "maximum_absolute_error": max_error,
        "checks": {
            "immutable_corpus_hashes_and_exact_test_split": True,
            "candidate_indices_and_hashes": True,
            "cache_provenance_against_actual_corpus_and_installed_frozen_rules": True,
            "ordered_evaluation_identity": True,
            "row_score_and_coverage_recomputed": True,
            "aggregate_and_task_means_recomputed": True,
            "paired_task_bootstrap_all_components_recomputed": True,
        },
        "seed_selected_change_scope": change_scope,
        "routing_assignment_records_identical": assignments_identical,
        "model_provenance": {"router": router, "teacher": teacher},
        "score": {
            key: comparison["components"]["score"][key]
            for key in (
                "seed_mean",
                "selected_mean",
                "task_balanced_difference",
                "paired_task_bootstrap_95ci",
            )
        },
        "bootstrap": {
            key: bootstrap[key]
            for key in ("method", "estimand", "confidence", "draws", "seed", "resampling_units")
        },
        "source_sha256": source_hashes,
        "corpus_sha256": corpus_hashes
        | {"manifest.json": file_digest(corpus_dir / "manifest.json")},
        "verification_script_sha256": file_digest(Path(__file__)),
        "limitations": [
            "LLM semantic proxies, not population accuracy or learner lift.",
            "Task-cluster intervals condition on fixed graphs and their single cached evaluations; model-generation and optimizer-selection uncertainty are excluded.",
            "Semantic overlap between original task clusters can remain.",
            "Identical routing can have different membership judgments because the evaluator sees each full graph, including its edges.",
            "The no-test-feedback and pre-completion flags are recorded declarations; this artifact check does not independently establish historical process order.",
        ],
        "formation_modified": False,
        "model_calls": 0,
        "final_export_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="Run containing frozen evaluation artifacts"
    )
    parser.add_argument(
        "--corpus",
        "--dataset",
        dest="corpus",
        type=Path,
        required=True,
        help="Immutable full_corpus output directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Successful verification receipt JSON"
    )
    args = parser.parse_args()
    try:
        receipt = verify_comparison(args.run, args.corpus)
    except ValueError as exc:
        parser.exit(1, f"Verification failed: {exc}. No new receipt written.\n")
    except (KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Verification failed ({type(exc).__name__}); no new receipt written.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, args.output)
    print(
        f"Verified {receipt['rollouts']} rollouts across {receipt['original_task_clusters']} task clusters; {receipt['numeric_comparisons']} numerical checks. Receipt: {args.output}"
    )


if __name__ == "__main__":
    main()
