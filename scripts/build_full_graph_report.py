#!/usr/bin/env python3
"""Build a shareable progress/final report from actual full-graph artifacts.

Only allowlisted aggregate metrics, graph definitions, and bounded witness IDs
are published. Runtime configuration, raw histories, assignment evidence, judge
feedback, task instructions, and exception text are never copied. The report is
safe to regenerate while a run is in progress and does not run any inference.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIMITATIONS = [
    "The corpus contains archived Horizon/Sonnet 4.5 trajectories mapped to Data Eng Bench task "
    "families; these are not executions of the current public benchmark.",
    "Task IDs are disjoint across training, Pareto, and test splits. Semantic family disjointness "
    "has not been established. Pareto scores are adaptively selected validation scores.",
    "Frozen held-out evaluation precedes all-corpus graph completion. Final completion and graph "
    "audits are transductive and are not a second held-out generalization measurement.",
    "Semantic scores and source applicability audits are LLM proxy judgments. The independent "
    "audit uses a separate frozen procedure, not a different model family. Sampled support does "
    "not establish universal applicability or executable graph paths.",
    "Terminal reward variance is descriptive. Repeated histories from one rollout share its "
    "outcome; trajectory-deduplicated and task-balanced estimates are reported separately. "
    "No claim of causal difficulty, identical-history variance, or post-training lift is made.",
    "The dataset includes one zero assigned by manual transcript review; the variance artifact "
    "contains a sensitivity analysis excluding manual rewards.",
    "Generated task specifications have LLM feasibility reviews only unless an explicit execution "
    "receipt is present. No benchmark execution or training benefit is inferred from a draft.",
]


def safe_text(value: Any) -> str:
    """Redact recognizable credential syntax in otherwise publishable definitions."""
    text = str(value) if value is not None else ""
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{10,}|AKIA[A-Z0-9]{16})\b", "[REDACTED]", text)
    return re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )


def numeric(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def number_map(value: Any) -> dict:
    return {safe_text(k): numeric(v) for k, v in value.items()} if isinstance(value, dict) else {}


def fmt(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return safe_text(value)


def md(value: Any) -> str:
    escaped = html.escape(fmt(value)).replace("\n", " ")
    return re.sub(r"([\\|\[\]()*_`])", r"\\\1", escaped)


def public_heldout_comparison(value: dict) -> dict:
    """Allowlist paired evaluation summaries; never copy inference rows or evidence."""
    metrics = (
        "score",
        "history_coverage",
        "transition_coverage",
        "coherence",
        "outgoing_applicability",
        "complexity",
    )
    components = {}
    for metric in metrics:
        row = value.get("components", {}).get(metric)
        if not isinstance(row, dict):
            continue
        interval = row.get("paired_task_bootstrap_95ci")
        components[metric] = {
            **{
                key: numeric(row.get(key))
                for key in (
                    "seed_mean",
                    "selected_mean",
                    "mean_difference",
                    "seed_task_balanced_mean",
                    "selected_task_balanced_mean",
                    "task_balanced_difference",
                )
            },
            "paired_task_bootstrap_95ci": [numeric(item) for item in interval]
            if isinstance(interval, list) and len(interval) == 2
            else None,
            "favorable_direction": safe_text(row.get("favorable_direction")),
        }
    bootstrap = value.get("bootstrap", {})
    return {
        **{
            key: safe_text(value.get(key))
            for key in (
                "format",
                "seed_candidate_hash",
                "selected_candidate_hash",
                "seed_evaluation_identity",
                "selected_evaluation_identity",
                "evidence_type",
            )
        },
        **{
            key: value.get(key) is True
            for key in (
                "selected_equals_seed",
                "before_transductive_completion",
                "optimization_uses_test_feedback",
            )
        },
        "rollouts": numeric(value.get("rollouts")),
        "original_tasks": numeric(value.get("original_tasks")),
        "common_rollout_ids": [safe_text(item) for item in value.get("common_rollout_ids", [])],
        "components": components,
        "per_task": {
            safe_text(task): {
                "rollouts": numeric(row.get("rollouts")),
                "components": {
                    key: {
                        name: numeric(measurement.get(name))
                        for name in ("seed_mean", "selected_mean", "difference")
                    }
                    for key, measurement in row.get("components", {}).items()
                    if key in metrics
                },
            }
            for task, row in value.get("per_task", {}).items()
        },
        "bootstrap": {
            **{key: safe_text(bootstrap.get(key)) for key in ("method", "estimand")},
            **{
                key: numeric(bootstrap.get(key))
                for key in ("confidence", "draws", "seed", "resampling_units")
            },
            "equal_rollout_counts_per_task": bootstrap.get("equal_rollout_counts_per_task") is True,
        },
        "limitations": [safe_text(item) for item in value.get("limitations", [])],
    }


def variance_census_verified(value: dict | None, assignments: dict | None) -> bool:
    """Require a statistics row for every occupied state and its exact membership counts."""
    if value is None or assignments is None:
        return False
    rows = value.get("states", [])
    groups = defaultdict(list)
    for assignment in assignments.values():
        if assignment.get("state_id") is not None:
            groups[assignment["state_id"]].append(assignment)
    ids = [row.get("state_id") for row in rows]
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(groups)
        or value.get("histories") != len(assignments)
        or value.get("assigned_histories") != sum(map(len, groups.values()))
        or value.get("missing_assignment_count") != 0
        or value.get("unassigned_count") != 0
    ):
        return False
    for row in rows:
        members = groups[row["state_id"]]
        if (
            row.get("member_histories") != len(members)
            or row.get("distinct_visiting_rollouts")
            != len({member.get("rollout_id") for member in members})
            or row.get("distinct_visiting_tasks")
            != len({member.get("task_id") for member in members})
        ):
            return False
        for key in ("history_weighted", "trajectory_deduplicated", "task_balanced"):
            moment = row.get(key, {})
            weight = numeric(moment.get("weight"))
            if weight is None or weight < 0:
                return False
            if weight > 0 and (
                numeric(moment.get("mean")) is None
                or numeric(moment.get("population_variance")) is None
                or moment["population_variance"] < 0
            ):
                return False
    return True


def collect_report(run_dir: Path, corpus_dir: Path) -> dict:
    warnings: list[str] = []
    receipts: dict[str, str] = {}

    def read(name: str, *, corpus: bool = False) -> Any:
        path = (corpus_dir if corpus else run_dir) / name
        if not path.exists():
            return None
        raw = path.read_bytes()
        receipts[("corpus/" if corpus else "run/") + name] = hashlib.sha256(raw).hexdigest()
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            warnings.append(f"Could not parse {name}; artifact is unavailable in this snapshot.")
            return None

    corpus_manifest = read("manifest.json", corpus=True) or {}
    corpus_stats = corpus_manifest.get("statistics", {})
    contract = read("run_contract.json") or {}
    completion = read("completion.json")
    result = read("gepa_result.json")
    heldout = read("heldout_test.json")
    baseline = read("baseline_heldout.json")
    comparison = read("heldout_comparison.json")
    variance = read("state_reward_variance.json")
    audit = read("independent_audit_summary.json")
    tasks = read("task_draft_summary.json")
    executable = read("executable_task_summary.json")
    graph = read("graph.json")
    graph_source = "graph.json" if graph else None
    if not graph:
        for name in ("optimized_candidate.json", "seed_candidate.json"):
            candidate = read(name)
            if candidate:
                try:
                    decoded = {
                        **json.loads(candidate["state_spec"]),
                        **json.loads(candidate["edge_spec"]),
                    }
                    graph = {"nodes": decoded["states"], "edges": decoded["edges"]}
                    graph_source = name
                except (KeyError, TypeError, json.JSONDecodeError):
                    warnings.append(f"Could not decode graph definitions from {name}.")
                break
    graph = graph or {"nodes": [], "edges": []}
    assignments = read("assignments_all.json")
    assignment_source = "assignments_all.json" if assignments is not None else None
    if assignments is None:
        assignments = read("optimized_assignments_all.json")
        assignment_source = "optimized_assignments_all.json" if assignments is not None else None

    expected_ids = None
    history_index = corpus_dir / "histories.jsonl"
    if history_index.exists():
        expected_ids = set()
        duplicates = 0
        digest = hashlib.sha256()
        try:
            with history_index.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    hid = json.loads(line)["history_id"]
                    duplicates += hid in expected_ids
                    expected_ids.add(hid)
            receipts["corpus/histories.jsonl"] = digest.hexdigest()
            if duplicates:
                warnings.append(f"Corpus history index contains {duplicates} duplicate IDs.")
                expected_ids = None
        except (KeyError, json.JSONDecodeError):
            warnings.append("History index could not be verified; exact census remains unverified.")
            expected_ids = None
    target_histories = numeric(corpus_stats.get("histories", contract.get("histories")))
    if expected_ids is not None and target_histories != len(expected_ids):
        warnings.append("Corpus manifest history count disagrees with the explicit history index.")
    actual_ids = set(assignments) if isinstance(assignments, dict) else set()
    missing = len(expected_ids - actual_ids) if expected_ids is not None else None
    extra = len(actual_ids - expected_ids) if expected_ids is not None else None
    nulls = (
        sum(a.get("state_id") is None for a in assignments.values())
        if assignments is not None
        else None
    )
    invalid_rows = (
        sum(a.get("history_id") != hid for hid, a in assignments.items())
        if assignments is not None
        else None
    )
    exact = (
        expected_ids is not None
        and actual_ids == expected_ids
        and len(expected_ids) == target_histories
        and not invalid_rows
    )
    if invalid_rows:
        warnings.append(f"Assignment keys disagree with {invalid_rows} history identifiers.")

    events = []
    if (run_dir / "events.jsonl").exists():
        for line in (run_dir / "events.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # An append may be in flight; never expose the incomplete line.
                continue
    proposed = [event for event in events if event.get("event") == "proposal"]
    attempts = max((int(e.get("attempt", 0)) for e in proposed), default=0)
    attempts = max(attempts, len(list((run_dir / "proposals").glob("*.json"))))
    coverage_passes = len(
        {e.get("candidate_hash") for e in events if e.get("event") == "coverage_accepted"}
    )
    scores = [numeric(v) for v in (result or {}).get("val_aggregate_scores", [])]
    scores = scores if all(v is not None for v in scores) else []
    best_idx = max(range(len(scores)), key=scores.__getitem__) if scores else None
    expected_candidate_hash = None
    if best_idx is not None and best_idx < len((result or {}).get("candidates", [])):
        candidate = result["candidates"][best_idx]
        encoded = json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        expected_candidate_hash = hashlib.sha256(encoded.encode()).hexdigest()
    heldout_matches = (
        heldout is not None
        and expected_candidate_hash is not None
        and heldout.get("candidate_hash") == expected_candidate_hash
    )
    if heldout is not None and expected_candidate_hash is not None and not heldout_matches:
        warnings.append("Frozen held-out report does not match the selected GEPA candidate.")
    comparison_validated = False
    if comparison is not None:
        seed = ((result or {}).get("candidates") or [None])[0]
        seed_hash = hashlib.sha256(
            json.dumps(seed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        common_ids = comparison.get("common_rollout_ids", [])
        selected_ids = [row.get("rollout_id") for row in (heldout or {}).get("rollouts", [])]
        baseline_ids = [row.get("rollout_id") for row in (baseline or {}).get("rollouts", [])]
        comparison_validated = bool(
            heldout_matches
            and baseline is not None
            and seed is not None
            and comparison.get("seed_candidate_hash") == baseline.get("candidate_hash") == seed_hash
            and comparison.get("selected_candidate_hash") == expected_candidate_hash
            and comparison.get("seed_evaluation_identity") == baseline.get("evaluation_identity")
            and comparison.get("selected_evaluation_identity")
            == (heldout or {}).get("evaluation_identity")
            and isinstance(comparison.get("seed_evaluation_identity"), str)
            and isinstance(comparison.get("selected_evaluation_identity"), str)
            and len(common_ids) == len(set(common_ids)) == comparison.get("rollouts")
            and set(common_ids) == set(selected_ids) == set(baseline_ids)
            and len(common_ids) == len(selected_ids) == len(baseline_ids)
            and comparison.get("before_transductive_completion") is True
            and comparison.get("optimization_uses_test_feedback") is False
        )
        if not comparison_validated:
            warnings.append(
                "Paired held-out comparison identities or common rollout census are unverified."
            )

    members: dict[str, list] = defaultdict(list)
    if assignments is not None:
        for assignment in assignments.values():
            if assignment.get("state_id") is not None:
                members[assignment["state_id"]].append(assignment)
    rewards = {state["state_id"]: state for state in (variance or {}).get("states", [])}
    nodes = []
    for node in graph.get("nodes", []):
        sid = node["id"]
        rows = members.get(sid, [])
        outcome = rewards.get(sid, {})
        nodes.append(
            {
                **{key: safe_text(node.get(key)) for key in ("id", "name", "description")},
                "exclusions": [safe_text(item) for item in node.get("exclusions", [])],
                "member_histories": len(rows) if assignments is not None else None,
                "distinct_rollouts": len({r.get("rollout_id") for r in rows})
                if assignments is not None
                else None,
                "distinct_tasks": len({r.get("task_id") for r in rows})
                if assignments is not None
                else None,
                "reward_mean": numeric(outcome.get("trajectory_deduplicated", {}).get("mean")),
                "reward_variance": numeric(
                    outcome.get("trajectory_deduplicated", {}).get("population_variance")
                ),
                "history_weighted_mean": numeric(outcome.get("history_weighted", {}).get("mean")),
                "history_weighted_variance": numeric(
                    outcome.get("history_weighted", {}).get("population_variance")
                ),
                "task_balanced_mean": numeric(outcome.get("task_balanced", {}).get("mean")),
                "task_balanced_variance": numeric(
                    outcome.get("task_balanced", {}).get("population_variance")
                ),
                "manual_excluded_variance": numeric(
                    outcome.get("sensitivity_excluding_manual_rewards", {})
                    .get("trajectory_deduplicated", {})
                    .get("population_variance")
                ),
            }
        )
    edges = []
    for edge in graph.get("edges", []):
        witnesses = edge.get("witnesses", [])
        edges.append(
            {
                **{
                    key: safe_text(edge.get(key))
                    for key in ("id", "source", "target", "operation", "effect", "bindings")
                },
                "witness_count": len(witnesses) if "witnesses" in edge else None,
                "distinct_task_count": numeric(edge.get("distinct_task_count")),
                "sampled_traversable": edge.get("traversable") is True,
                "evidence_status": safe_text(
                    edge.get("evidence_status", "Optimized specification; not yet audited")
                ),
                "source_audit_verdicts": number_map(edge.get("audit", {}).get("verdicts", {})),
                "universal_contract_certified": False,
                "witness_reference_subset": [
                    {
                        key: safe_text(w[key])
                        for key in (
                            "transition_id",
                            "rollout_id",
                            "task_id",
                            "source_id",
                            "target_id",
                        )
                        if key in w
                    }
                    for w in witnesses[:3]
                ],
                "witness_reference_subset_limit": 3,
            }
        )
    node_ids = {node["id"] for node in nodes}
    unknown_states = len(
        [
            row
            for row in (assignments or {}).values()
            if row.get("state_id") is not None and row["state_id"] not in node_ids
        ]
    )
    if unknown_states:
        warnings.append(
            f"{unknown_states} assignments reference states absent from the reported graph snapshot."
        )
    observed = sum(edge["witness_count"] or 0 for edge in edges)
    unassigned_transitions = len(graph.get("unassigned_transitions", []))
    expected_transitions = numeric(corpus_stats.get("transitions"))
    transition_ids = [
        w.get("transition_id") for edge in graph.get("edges", []) for w in edge.get("witnesses", [])
    ]
    transition_ids.extend(w.get("transition_id") for w in graph.get("unassigned_transitions", []))
    expected_transition_ids = set()
    for hid in expected_ids or []:
        try:
            rid, index = hid.rsplit(":h", 1)
            if f"{rid}:h{int(index) + 1:04d}" in expected_ids:
                expected_transition_ids.add(f"{rid}:t{int(index):04d}")
        except (ValueError, TypeError):
            warnings.append(
                "Unexpected history identifier format prevented transition census verification."
            )
            break
    transition_census = (
        graph_source == "graph.json"
        and len(transition_ids) == len(set(transition_ids)) == expected_transitions
        and set(transition_ids) == expected_transition_ids
    )
    required = {
        "gepa_result": result is not None,
        "frozen_test": heldout_matches,
        "final_graph": graph_source == "graph.json",
        "final_assignments": assignment_source == "assignments_all.json",
        "reward_variance": variance is not None,
        "reward_variance_census": variance_census_verified(variance, assignments),
        "independent_audits": audit is not None,
        "task_drafts": tasks is not None,
        "completion_record": completion is not None and completion.get("status") == "complete",
        "exact_history_census": exact and nulls == 0 and not unknown_states,
        "transition_census": transition_census,
        "all_transitions_have_edges": graph_source == "graph.json" and unassigned_transitions == 0,
    }
    if variance is not None and not required["reward_variance_census"]:
        warnings.append(
            "Reward variance does not cover the exact occupied-state membership census with finite moments."
        )
    if (
        executable is not None
        or (completion or {}).get("executable_task_target_reached") is not None
    ):
        required["executable_stage_finalized"] = executable is not None and executable.get(
            "status"
        ) in {"target_reached", "supported_paths_exhausted"}
    if baseline is not None or comparison is not None:
        required["paired_heldout_comparison"] = comparison_validated
    selected = []
    for draft in (tasks or {}).get("selected_examples", []):
        selected.append(
            {
                **{
                    key: safe_text(draft.get(key))
                    for key in (
                        "path_id",
                        "title",
                        "status",
                        "review_verdict",
                        "selection_evidence",
                    )
                },
                "executed": draft.get("executed") is True,
                "benchmark_validated": draft.get("benchmark_validated") is True,
            }
        )
    independence = (audit or {}).get("independence", {})
    executed_examples = []
    for example in (executable or {}).get("selected_examples", []):
        validation = example.get("local_validation", {})
        verified = (
            example.get("status") == "locally_executed_consistent"
            and validation.get("reference_query_executed") is True
            and validation.get("independent_query_executed") is True
            and validation.get("queries_agree") is True
        )
        executed_examples.append(
            {
                "path_id": safe_text(example.get("path_id")),
                "title": safe_text(example.get("title", example.get("path_id"))),
                "status": safe_text(example.get("status")),
                "reference_query_executed": validation.get("reference_query_executed") is True,
                "independent_query_executed": validation.get("independent_query_executed") is True,
                "queries_agree": validation.get("queries_agree") is True,
                "execution_receipt_supported": verified,
                "learner_evaluated": example.get("learner_evaluated"),
                "original_benchmark_reproduced": example.get("original_benchmark_reproduced"),
            }
        )
    executed_count = sum(example["execution_receipt_supported"] for example in executed_examples)
    if executable is not None and executed_count != executable.get(
        "locally_executed_consistent_count"
    ):
        warnings.append(
            "Executable task count disagrees with selected examples' explicit local execution receipts."
        )
        required["executable_receipts_consistent"] = False
    if executable is not None and executable.get("status") == "supported_paths_exhausted":
        warnings.append(
            "Executable-task construction exhausted eligible paths before reaching its target."
        )
    if executable is not None and executable.get("status") in {
        "target_reached",
        "supported_paths_exhausted",
    }:
        target = numeric(executable.get("requested_examples"))
        reached = executable.get("target_reached")
        required["executable_target_consistent"] = (
            target is not None
            and target >= 1
            and reached is (executed_count >= target)
            and (executable["status"] == "target_reached") is reached
        )
        if not required["executable_target_consistent"]:
            warnings.append(
                "Executable target status disagrees with requested count or local execution receipts."
            )
    complete = all(required.values())
    router_model = {
        "name": safe_text(contract.get("model")),
        "revision": safe_text(contract.get("revision")),
    }
    teacher_model = {
        "name": safe_text(contract.get("teacher_model")),
        "revision": safe_text(contract.get("teacher_revision")),
        "reasoning": contract.get("teacher_reasoning"),
    }
    audit_model = {
        "name": safe_text(independence.get("public_model", {}).get("model")),
        "revision": safe_text(independence.get("public_model", {}).get("revision")),
    }
    return {
        "report_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "partial",
        "status_meaning": "Artifact and census completion; not a statement that all semantic checks passed",
        "completion_checks": required,
        "warnings": warnings,
        "corpus": {
            key: numeric(corpus_stats.get(key, contract.get(key)))
            for key in ("tasks", "rollouts", "messages", "histories", "transitions")
        }
        | {
            "task_splits": number_map(corpus_stats.get("task_split_counts", {})),
            "rollout_splits": number_map(corpus_stats.get("rollout_split_counts", {})),
            "reward_provenance": number_map(corpus_stats.get("reward_provenance_counts", {})),
        },
        "model": router_model,
        "model_provenance": {
            "router": router_model,
            "teacher": teacher_model,
            "independent_audit": audit_model,
        },
        "optimization": {
            "proposals_observed": attempts,
            "proposal_budget": numeric(contract.get("max_proposals")),
            "crossover_proposals_observed": sum(e.get("kind") == "crossover" for e in proposed),
            "coverage_check_passes_observed": coverage_passes,
            "accepted_revisions": max(0, len((result or {}).get("candidates", [])) - 1)
            if result is not None
            else None,
            "candidate_count": len((result or {}).get("candidates", []))
            if result is not None
            else None,
            "initial_pareto_score": scores[0] if scores else None,
            "best_pareto_score": scores[best_idx] if best_idx is not None else None,
            "best_candidate_index": best_idx,
            "minibatch_rollouts": numeric(contract.get("minibatch")),
            "train_rollouts": numeric(contract.get("train_rollouts")),
            "pareto_rollouts": numeric(contract.get("pareto_rollouts")),
        },
        "frozen_test": {
            "available": heldout is not None,
            "matches_selected_candidate": heldout_matches,
            "candidate_hash": safe_text((heldout or {}).get("candidate_hash")),
            "rollouts": numeric((heldout or {}).get("summary", {}).get("rollouts")),
            "mean_score": numeric((heldout or {}).get("summary", {}).get("mean_score")),
            "metrics": number_map((heldout or {}).get("summary", {}).get("metrics", {})),
        },
        "heldout_comparison": {
            "available": comparison is not None,
            "identities_and_census_verified": comparison_validated,
            **public_heldout_comparison(comparison or {}),
        },
        "assignments": {
            "artifact": assignment_source,
            "expected": target_histories,
            "recorded": len(assignments) if assignments is not None else None,
            "missing_ids": missing,
            "extra_ids": extra,
            "unassigned": nulls,
            "invalid_identity_rows": invalid_rows,
            "unknown_state_rows": unknown_states,
            "exact_ids_verified": exact,
        },
        "graph": {
            "artifact": graph_source,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "sampled_traversable_edges": sum(e["sampled_traversable"] for e in edges),
            "observed_transitions": observed if graph_source == "graph.json" else None,
            "unassigned_transitions": unassigned_transitions
            if graph_source == "graph.json"
            else None,
            "witness_export_policy": "At most three references per edge, explicitly a subset; full local witness ledger remains unchanged",
        },
        "independent_audits": {
            "planned": numeric((audit or {}).get("planned_checks")),
            "completed": numeric((audit or {}).get("completed_checks")),
            "failed_requests": numeric((audit or {}).get("failed_requests")),
            "by_kind": {
                safe_text(k): number_map(v) for k, v in (audit or {}).get("by_kind", {}).items()
            },
            "different_model_from_optimizer": independence.get("different_model_from_optimizer"),
        },
        "task_drafts": {
            **{
                key: numeric((tasks or {}).get(key))
                for key in (
                    "requested_examples",
                    "attempted_paths",
                    "draft_count",
                    "executed_task_count",
                    "benchmark_validated_count",
                )
            },
            "review_verdict_counts": number_map((tasks or {}).get("review_verdict_counts", {})),
            "selected_examples": selected,
            "generation_error_count": sum(
                item.get("status") == "error" for item in (tasks or {}).get("attempts", [])
            )
            if tasks is not None
            else None,
        },
        "executable_tasks": {
            "available": executable is not None,
            "status": safe_text((executable or {}).get("status", "not_started_or_unavailable")),
            "requested_examples": numeric((executable or {}).get("requested_examples")),
            "eligible_supported_paths": numeric((executable or {}).get("eligible_supported_paths")),
            "attempted_paths": len((executable or {}).get("attempts", []))
            if executable is not None
            else None,
            "reported_consistent_count": numeric(
                (executable or {}).get("locally_executed_consistent_count")
            ),
            "receipt_supported_consistent_count": executed_count
            if executable is not None
            else None,
            "target_reached": (executable or {}).get("target_reached"),
            "attempt_status_counts": number_map(
                (executable or {}).get("attempt_status_counts", {})
            ),
            "construction_error_count": sum(
                item.get("status") == "construction_error"
                for item in (executable or {}).get("attempts", [])
            )
            if executable is not None
            else None,
            "failed_requests": numeric((executable or {}).get("failed_requests")),
            "learner_evaluated": (executable or {}).get("learner_evaluated"),
            "official_benchmark_verifier": (executable or {}).get("official_benchmark_verifier"),
            "selected_examples": executed_examples,
        },
        "limitations": LIMITATIONS,
        "source_artifact_sha256": receipts,
    }


def markdown_report(report: dict) -> str:
    c, o, a, g = (report[key] for key in ("corpus", "optimization", "assignments", "graph"))
    lines = [
        "# Full-corpus superstate graph",
        "",
        f"**Status: {report['status'].upper()}.** {report['status_meaning']}.",
        "",
        f"Snapshot: {report['generated_at_utc']}. Router: {md(report['model_provenance']['router']['name'])}. "
        f"Teacher: {md(report['model_provenance']['teacher']['name'] or None)}.",
        "",
        "| Measurement | Observed value |",
        "|---|---|",
        *[
            f"| {label} | {md(value)} |"
            for label, value in [
                ("Original tasks", c["tasks"]),
                ("Rollouts", c["rollouts"]),
                ("Complete history prefixes", c["histories"]),
                ("Recorded transitions", c["transitions"]),
                ("GEPA proposed revisions", o["proposals_observed"]),
                ("GEPA accepted revisions", o["accepted_revisions"]),
                ("Initial Pareto mean score", o["initial_pareto_score"]),
                ("Best Pareto mean score", o["best_pareto_score"]),
                ("Frozen held-out mean score", report["frozen_test"]["mean_score"]),
                ("Assignment records", a["recorded"]),
                ("Missing history IDs", a["missing_ids"]),
                ("Explicitly unassigned histories", a["unassigned"]),
                ("Superstates", g["node_count"]),
                ("Edges in available graph", g["edge_count"]),
                (
                    "Edges eligible under sampled applicability checks",
                    g["sampled_traversable_edges"],
                ),
                ("Independent checks completed", report["independent_audits"]["completed"]),
                (
                    "Independent audit failed requests",
                    report["independent_audits"]["failed_requests"],
                ),
                (
                    "Locally executed consistent synthetic tasks",
                    report["executable_tasks"]["receipt_supported_consistent_count"],
                ),
                (
                    "Executable task target reached",
                    "Yes"
                    if report["executable_tasks"]["target_reached"] is True
                    else "No"
                    if report["executable_tasks"]["target_reached"] is False
                    else None,
                ),
                (
                    "Executable construction errors",
                    report["executable_tasks"]["construction_error_count"],
                ),
                (
                    "Executable stage failed requests (when recorded)",
                    report["executable_tasks"]["failed_requests"],
                ),
            ]
        ],
        "",
    ]
    if report["status"] != "complete":
        lines += [
            "Pending or unverified: "
            + ", ".join(key for key, value in report["completion_checks"].items() if not value)
            + ".",
            "",
        ]
    comparison = report["heldout_comparison"]
    lines += ["## Frozen seed versus selected graph", ""]
    if comparison["identities_and_census_verified"]:
        lines += [
            f"Both fixed graphs were evaluated on the same {md(comparison['rollouts'])} rollouts "
            f"from {md(comparison['original_tasks'])} held-out tasks, before all-corpus completion. "
            "These are LLM semantic proxy scores; they are not learner performance.",
            "",
            "| Metric | Seed mean | Selected mean | Paired task mean change | 95% task-cluster interval |",
            "|---|---:|---:|---:|---|",
        ]
        for metric, values in comparison["components"].items():
            interval = values["paired_task_bootstrap_95ci"]
            lines.append(
                "| "
                + " | ".join(
                    md(value)
                    for value in (
                        metric,
                        values["seed_mean"],
                        values["selected_mean"],
                        values["task_balanced_difference"],
                        " to ".join(fmt(v) for v in interval) if interval is not None else None,
                    )
                )
                + " |"
            )
        lines += [
            "",
            f"Paired original-task cluster bootstrap: {md(comparison['bootstrap']['draws'])} draws. "
            "Positive change favors the selected graph except complexity, where lower is better. "
            "Intervals condition on fixed graphs and cached model judgments; they exclude optimizer "
            "selection and model-generation uncertainty.",
            "",
        ]
    else:
        lines += [
            "No identity-verified paired held-out comparison is available in this snapshot.",
            "",
        ]
    lines += [
        f"Available graph source: `{g['artifact'] or 'none'}`. Assignment source: `{a['artifact'] or 'none'}`.",
        "",
        "## States with the most assigned histories",
        "",
        "Reward mean and variance below count each visiting trajectory once. Variance is a population descriptive estimate.",
        "",
        "| State | Histories | Rollouts | Tasks | Mean reward | Reward variance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for node in sorted(g["nodes"], key=lambda n: (-(n["member_histories"] or 0), n["id"]))[:25]:
        lines.append(
            "| "
            + " | ".join(
                md(value)
                for value in [
                    f"{node['id']}: {node['name']}",
                    node["member_histories"],
                    node["distinct_rollouts"],
                    node["distinct_tasks"],
                    node["reward_mean"],
                    node["reward_variance"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Independent audits",
        "",
        "| Check type | Supported | Contradicted | Unknown | Not run |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, counts in report["independent_audits"]["by_kind"].items():
        lines.append(
            "| "
            + " | ".join(
                [md(kind)]
                + [
                    md(counts.get(k, 0))
                    for k in ("supported", "contradicted", "unknown", "not_run")
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Selected task specifications",
        "",
        "| Draft | Feasibility review | Executed | Benchmark validated |",
        "|---|---|---|---|",
    ]
    for draft in report["task_drafts"]["selected_examples"]:
        lines.append(
            "| "
            + " | ".join(
                md(v)
                for v in [
                    draft["title"],
                    draft["review_verdict"],
                    "Yes" if draft["executed"] else "No",
                    "Yes" if draft["benchmark_validated"] else "No",
                ]
            )
            + " |"
        )
    if not report["task_drafts"]["selected_examples"]:
        lines += ["", "No selected task drafts are available in this snapshot."]
    lines += [
        "",
        "## Locally executed synthetic tasks",
        "",
        "These are separate from the reviewed specification drafts above. Execution means "
        "two independently prompted reference SQL queries ran on a synthetic fixture and agreed; "
        "it does not mean a learner passed the task or the source benchmark was reproduced.",
        "",
        f"Stage status: {md(report['executable_tasks']['status'])}. "
        f"Verified local receipts: {md(report['executable_tasks']['receipt_supported_consistent_count'])}. "
        f"Requested: {md(report['executable_tasks']['requested_examples'])}.",
        "",
        "| Task | Reference executed | Independent query executed | Results agree |",
        "|---|---|---|---|",
    ]
    for task in report["executable_tasks"]["selected_examples"]:
        lines.append(
            "| "
            + " | ".join(
                [md(task["title"])]
                + [
                    "Yes" if task[key] else "No"
                    for key in (
                        "reference_query_executed",
                        "independent_query_executed",
                        "queries_agree",
                    )
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        *["- " + text for text in report["limitations"]],
        "",
        "The searchable [HTML report](report.html) contains every available node and edge, plus "
        "an adjacency matrix with exact witness counts and a sampled-traversable filter. "
        "[report.json](report.json) contains allowlisted report data and artifact hashes. "
        "Edge witness references are limited to three labeled examples per edge; the full local graph remains unchanged.",
        "",
    ]
    if report["warnings"]:
        lines += [
            "## Snapshot warnings",
            "",
            *["- " + safe_text(w) for w in report["warnings"]],
            "",
        ]
    return "\n".join(lines)


def adjacency_matrix(graph: dict) -> str:
    """Draw a sparse SVG over the complete Cartesian grid of source/target states."""
    node_ids = [node["id"] for node in graph["nodes"]]
    if not node_ids:
        return '<p class="muted">No state definitions are available yet.</p>'
    indices = {sid: i for i, sid in enumerate(node_ids)}
    pairs: dict[tuple[int, int], dict] = defaultdict(lambda: {"all": 0, "eligible": 0, "edges": 0})
    for edge in graph["edges"]:
        if edge["source"] not in indices or edge["target"] not in indices:
            continue
        row = pairs[indices[edge["source"]], indices[edge["target"]]]
        count = edge["witness_count"] or 0
        row["all"] += count
        row["eligible"] += count if edge["sampled_traversable"] else 0
        row["edges"] += 1
    pad, cell = 160, 14
    extent, side = len(node_ids) * cell, len(node_ids) * cell + pad + 20
    parts = [
        f'<svg id="adjacency" role="group" aria-label="Directed state adjacency matrix" '
        f'viewBox="0 0 {side} {side}" data-pad="{pad}" data-cell="{cell}" data-n="{len(node_ids)}" '
        f'data-maximum="{max((v["all"] for v in pairs.values()), default=0)}" xmlns="http://www.w3.org/2000/svg">',
        f'<defs><pattern id="cell-grid" x="{pad}" y="{pad}" width="{cell}" height="{cell}" patternUnits="userSpaceOnUse">'
        f'<rect width="{cell}" height="{cell}" fill="#f5f7fa"/><path d="M {cell} 0 L 0 0 0 {cell}" fill="none" stroke="#dce3eb" stroke-width=".6"/></pattern></defs>',
        f'<rect class="matrix-background" x="{pad}" y="{pad}" width="{extent}" height="{extent}" fill="url(#cell-grid)"/>',
        f'<text x="{pad}" y="18" font-size="12">Destination state →</text>',
        f'<text x="10" y="{pad - 12}" font-size="12">Source state ↓</text>',
    ]
    maximum = max((value["all"] for value in pairs.values()), default=0)
    for (i, j), value in sorted(pairs.items()):
        intensity = math.log1p(value["all"]) / math.log1p(maximum) if maximum else 0
        color = f"rgb({round(221 - intensity * 201)},{round(239 - intensity * 145)},{round(244 - intensity * 127)})"
        hidden = ' style="display:none"' if not value["all"] else ""
        title = html.escape(
            f"{node_ids[i]} → {node_ids[j]}: {value['all']} observed witnesses; "
            f"{value['eligible']} on sampled-traversable edges",
            quote=True,
        )
        parts.append(
            f'<rect class="matrix-cell" x="{pad + j * cell + 0.5}" y="{pad + i * cell + 0.5}" '
            f'width="{cell - 1}" height="{cell - 1}" fill="{color}"{hidden} data-i="{i}" data-j="{j}" '
            f'data-all="{value["all"]}" data-eligible="{value["eligible"]}" data-edges="{value["edges"]}">'
            f"<title>{title}</title></rect>"
        )
    for i, sid in enumerate(node_ids):
        text = html.escape(sid, quote=True)
        full = html.escape(graph["nodes"][i].get("name", sid), quote=True)
        x, y = pad + i * cell + cell / 2, pad + i * cell + cell / 2 + 3
        common = (
            f'class="matrix-node" data-node="{text}" data-index="{i}" tabindex="0" role="button"'
        )
        parts.append(
            f'<text {common} data-axis="source" x="{pad - 7}" y="{y}" text-anchor="end" font-size="9">'
            f"{text}<title>{full}; click to focus this state</title></text>"
        )
        parts.append(
            f'<text {common} data-axis="target" transform="translate({x},{pad - 7}) rotate(-65)" '
            f'text-anchor="start" font-size="9">{text}<title>{full}; click to focus this state</title></text>'
        )
    parts.append("</svg>")
    return "".join(parts)


REPORT_JS = r"""
(() => {
  'use strict';
  const searches = [...document.querySelectorAll('input[data-table]')];
  const mode = document.getElementById('matrix-mode');
  const focusLabel = document.getElementById('matrix-focus');
  const detail = document.getElementById('matrix-detail');
  const svg = document.getElementById('adjacency');
  let focusNode = null, focusPair = null;
  function filterTables() {
    for (const input of searches) {
      const table = input.dataset.table;
      const query = input.value.toLocaleLowerCase();
      const rows = [...document.querySelectorAll('#' + table + ' tbody tr')];
      let visible = 0;
      for (const row of rows) {
        let show = row.textContent.toLocaleLowerCase().includes(query);
        if (table === 'states') {
          if (focusNode !== null) show = show && row.dataset.node === focusNode;
          if (focusPair !== null) show = show && focusPair.includes(row.dataset.node);
        } else {
          if (focusNode !== null) show = show &&
            (row.dataset.source === focusNode || row.dataset.target === focusNode);
          if (focusPair !== null) show = show &&
            row.dataset.source === focusPair[0] && row.dataset.target === focusPair[1];
          if (mode.value === 'eligible') show = show && row.dataset.sampled === '1';
        }
        row.hidden = !show;
        visible += Number(show);
      }
      document.getElementById(table + '-count').textContent = visible + ' of ' + rows.length + ' rows shown';
    }
  }
  function setFocus(node, pair) {
    focusNode = node; focusPair = pair;
    for (const input of searches) input.value = '';
    focusLabel.textContent = node !== null ? 'Focus: state ' + node :
      pair !== null ? 'Focus: ' + pair[0] + ' → ' + pair[1] : 'Focus: all states and pairs';
    for (const label of document.querySelectorAll('.matrix-node')) {
      label.classList.toggle('selected', node === label.dataset.node ||
        (pair !== null && pair.includes(label.dataset.node)));
    }
    filterTables();
  }
  for (const input of searches) input.addEventListener('input', filterTables);
  document.getElementById('matrix-clear').addEventListener('click', () => {
    setFocus(null, null); detail.textContent = 'No cell selected.';
  });
  if (svg) {
    const labels = [...svg.querySelectorAll('.matrix-node[data-axis="source"]')];
    const nodeIds = labels.map(label => label.dataset.node);
    const cells = [...svg.querySelectorAll('.matrix-cell')];
    const byPair = new Map(cells.map(cell => [cell.dataset.i + ':' + cell.dataset.j, cell.dataset]));
    const maximum = Number(svg.dataset.maximum);
    function updateColors() {
      for (const cell of cells) {
        const count = Number(cell.dataset[mode.value === 'eligible' ? 'eligible' : 'all']);
        cell.style.display = count > 0 ? '' : 'none';
        const intensity = maximum > 0 ? Math.log1p(count) / Math.log1p(maximum) : 0;
        cell.setAttribute('fill', 'rgb(' + Math.round(221 - intensity * 201) + ',' +
          Math.round(239 - intensity * 145) + ',' + Math.round(244 - intensity * 127) + ')');
      }
      filterTables();
    }
    function pairAt(event) {
      const transform = svg.getScreenCTM();
      if (!transform) return null;
      const point = svg.createSVGPoint();
      point.x = event.clientX; point.y = event.clientY;
      const local = point.matrixTransform(transform.inverse());
      const pad = Number(svg.dataset.pad), size = Number(svg.dataset.cell);
      const i = Math.floor((local.y - pad) / size), j = Math.floor((local.x - pad) / size);
      return i >= 0 && j >= 0 && i < nodeIds.length && j < nodeIds.length ? [i, j] : null;
    }
    function describe(pair) {
      const values = byPair.get(pair[0] + ':' + pair[1]) || {all: 0, eligible: 0, edges: 0};
      detail.textContent = nodeIds[pair[0]] + ' → ' + nodeIds[pair[1]] + ': ' +
        values.all + ' observed witnesses; ' + values.eligible +
        ' on sampled-traversable edges; ' + values.edges + ' operation contracts.';
    }
    svg.addEventListener('pointermove', event => { const pair = pairAt(event); if (pair) describe(pair); });
    svg.addEventListener('click', event => {
      const label = event.target.closest('.matrix-node');
      if (label) { setFocus(label.dataset.node, null); return; }
      const pair = pairAt(event);
      if (pair) { describe(pair); setFocus(null, pair.map(index => nodeIds[index])); }
    });
    for (const label of svg.querySelectorAll('.matrix-node')) {
      label.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault(); setFocus(label.dataset.node, null);
        }
      });
    }
    const zoom = document.getElementById('matrix-zoom');
    function resize() {
      const fit = Math.min(svg.viewBox.baseVal.width, svg.parentElement.clientWidth || svg.viewBox.baseVal.width);
      svg.style.width = fit * Number(zoom.value) + 'px';
    }
    zoom.addEventListener('input', resize);
    window.addEventListener('resize', resize);
    mode.addEventListener('change', updateColors);
    resize(); updateColors();
  } else mode.addEventListener('change', filterTables);
  filterTables();
})();
"""


def html_report(report: dict) -> str:
    def esc(value: Any) -> str:
        return html.escape(fmt(value), quote=True)

    graph = report["graph"]
    cards = [
        ("Task families", report["corpus"]["tasks"]),
        ("Rollouts", report["corpus"]["rollouts"]),
        ("History prefixes", report["corpus"]["histories"]),
        ("Superstates", graph["node_count"]),
        ("Observed/specification edges", graph["edge_count"]),
        ("Sampled-eligible edges", graph["sampled_traversable_edges"]),
    ]
    state_rows = []
    for node in graph["nodes"]:
        state_rows.append(
            f'<tr data-node="{esc(node["id"])}">'
            + f"<td><b>{esc(node['id'])}</b><br>{esc(node['name'])}</td>"
            + f"<td>{esc(node['description'])}<small>Exclusions: {esc('; '.join(node['exclusions']))}</small></td>"
            + "".join(
                f"<td>{esc(node[key])}</td>"
                for key in (
                    "member_histories",
                    "distinct_rollouts",
                    "distinct_tasks",
                    "reward_mean",
                    "reward_variance",
                )
            )
            + "</tr>"
        )
    edge_rows = []
    for edge in graph["edges"]:
        refs = "; ".join(w.get("transition_id", "") for w in edge["witness_reference_subset"])
        edge_rows.append(
            f'<tr data-source="{esc(edge["source"])}" data-target="{esc(edge["target"])}" data-sampled="{1 if edge["sampled_traversable"] else 0}">'
            + f"<td>{esc(edge['id'])}<br><b>{esc(edge['source'])} → {esc(edge['target'])}</b></td>"
            + f"<td>{esc(edge['operation'])}<small>Effect: {esc(edge['effect'])}</small><small>Bindings: {esc(edge['bindings'])}</small></td>"
            + f"<td>{esc(edge['witness_count'])}<small>Tasks: {esc(edge['distinct_task_count'])}</small></td>"
            + f"<td>{'Sampled support' if edge['sampled_traversable'] else 'Not eligible'}<small>{esc(edge['evidence_status'])}</small></td>"
            + f"<td>{esc(refs)}<small>Reference subset: at most 3. Not the full witness ledger.</small></td></tr>"
        )
    pending = ", ".join(k for k, v in report["completion_checks"].items() if not v) or "None"
    optimization = report["optimization"]
    draft_rows = "".join(
        f"<tr><td>{esc(d['title'])}</td><td>{esc(d['review_verdict'])}</td>"
        f"<td>{'Yes' if d['executed'] else 'No'}</td><td>{'Yes' if d['benchmark_validated'] else 'No'}</td></tr>"
        for d in report["task_drafts"]["selected_examples"]
    )
    audits = "".join(
        f"<li>{esc(kind)}: {esc(', '.join(f'{k}: {v}' for k, v in counts.items()))}</li>"
        for kind, counts in report["independent_audits"]["by_kind"].items()
    )
    execution_rows = "".join(
        f"<tr><td>{esc(task['title'])}</td>"
        + "".join(
            f"<td>{'Yes' if task[key] else 'No'}</td>"
            for key in ("reference_query_executed", "independent_query_executed", "queries_agree")
        )
        + "</tr>"
        for task in report["executable_tasks"]["selected_examples"]
    )
    models = report["model_provenance"]
    execution = report["executable_tasks"]
    reached = (
        "Yes"
        if execution["target_reached"] is True
        else "No"
        if execution["target_reached"] is False
        else None
    )
    comparison = report["heldout_comparison"]
    comparison_html = "<h3>Frozen seed versus selected graph</h3>"
    if comparison["identities_and_census_verified"]:
        comparison_html += (
            f"<p>The same {esc(comparison['rollouts'])} rollouts from {esc(comparison['original_tasks'])} "
            "held-out tasks were evaluated before all-corpus completion. These are semantic proxy scores, "
            "not learner performance.</p><table><thead><tr><th>Metric</th><th>Seed mean</th>"
            "<th>Selected mean</th><th>Paired task mean change</th><th>95% task-cluster interval</th>"
            "</tr></thead><tbody>"
        )
        for metric, values in comparison["components"].items():
            interval = values["paired_task_bootstrap_95ci"]
            comparison_html += (
                "<tr>"
                + "".join(
                    "<td>" + esc(value) + "</td>"
                    for value in (
                        metric,
                        values["seed_mean"],
                        values["selected_mean"],
                        values["task_balanced_difference"],
                        " to ".join(fmt(v) for v in interval) if interval is not None else None,
                    )
                )
                + "</tr>"
            )
        comparison_html += (
            f"</tbody></table><p>{esc(comparison['bootstrap']['draws'])} paired task-cluster bootstrap draws. "
            "Positive change favors selection except complexity, where lower is better. Intervals condition "
            "on fixed graphs and cached model judgments; they exclude selection and generation uncertainty.</p>"
        )
    else:
        comparison_html += "<p>No identity-verified paired comparison is available yet.</p>"
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Full-corpus superstate graph</title>
<style>body{font:15px/1.55 system-ui,sans-serif;color:#203047;background:#f3f6fa;margin:0}main{max-width:1500px;margin:auto;padding:36px}h1{font-size:36px;line-height:1.15;margin:8px 0}h2{margin-top:40px}a{color:#12677b}header,section{background:white;border:1px solid #dce4ee;border-radius:12px;padding:24px;margin-bottom:20px}.status{display:inline-block;background:#fff0ca;padding:5px 12px;border-radius:20px;font-weight:700}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.card{padding:16px;background:#edf5f7;border-radius:9px}.card b{display:block;font-size:28px;color:#145568}.muted,small{color:#5a6b7f}small{display:block;font-size:12px;margin-top:6px}input[type=search]{box-sizing:border-box;width:100%;padding:12px;border:1px solid #acbbcd;border-radius:6px;margin:10px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:12px;vertical-align:top;border-bottom:1px solid #e3e9f0;overflow-wrap:anywhere}th{background:#edf2f7;position:sticky;top:0}td:first-child{min-width:95px}.scroll{overflow:auto;max-height:680px}details{margin-top:18px}summary{cursor:pointer;font-weight:600}.notice{border-left:4px solid #dbab43;padding-left:16px}footer{color:#607086;font-size:12px}.matrix-tools{display:flex;gap:16px;flex-wrap:wrap;align-items:center}.matrix-tools select,.matrix-tools button{padding:8px;border:1px solid #acbbcd;border-radius:5px;background:white}.matrix-viewport{max-height:640px;overflow:auto;border:1px solid #dce4ee;margin-top:16px}#adjacency{display:block;max-width:none;height:auto}.matrix-node,.matrix-cell{cursor:pointer}.matrix-node:hover,.matrix-node.selected{fill:#087c99;font-weight:bold}.matrix-legend{display:inline-block;width:150px;height:10px;background:linear-gradient(to right,#ddeff4,#145e75);margin-right:8px}#matrix-detail{min-height:26px}@media(max-width:700px){main{padding:12px}header,section{padding:16px}h1{font-size:28px}}</style></head><body><main>
"""
        + f"""<header><span class="status">{esc(report["status"].upper())}</span><h1>Full-corpus superstate graph</h1>
<p class="muted">Snapshot {esc(report["generated_at_utc"])} · Router: {esc(models["router"]["name"] or None)} · Teacher: {esc(models["teacher"]["name"] or None)}</p>
<p class="notice">{esc(report["status_meaning"])}. Pending or unverified: {esc(pending)}.</p>
<div class="cards">{"".join(f'<div class="card">{esc(label)}<b>{esc(value)}</b></div>' for label, value in cards)}</div></header>
<section><h2>Optimization and frozen evaluation</h2><p>Observed proposals: <b>{esc(optimization["proposals_observed"])}</b>.
Accepted revisions: <b>{esc(optimization["accepted_revisions"])}</b>. Initial Pareto mean: <b>{esc(optimization["initial_pareto_score"])}</b>.
Best Pareto mean: <b>{esc(optimization["best_pareto_score"])}</b>. Frozen test mean: <b>{esc(report["frozen_test"]["mean_score"])}</b>.</p>{comparison_html}
<p>Assignment records: {esc(report["assignments"]["recorded"])}; missing history IDs: {esc(report["assignments"]["missing_ids"])}; explicitly unassigned: {esc(report["assignments"]["unassigned"])}.
Available graph: {esc(graph["artifact"])}.</p><p>Observed transitions: {esc(graph["observed_transitions"])}; unassigned transitions: {esc(graph["unassigned_transitions"])}.</p></section>
<section><h2>Directed adjacency matrix</h2><p>All source/destination state pairs are represented. Blank cells mean zero recorded witnesses; color uses a logarithmic witness-count scale. A sampled-eligible edge is not universally certified.</p>
<div class="matrix-tools"><label>Edges <select id="matrix-mode"><option value="all">All observed edges</option><option value="eligible">Sampled-traversable only</option></select></label><label>Zoom <input id="matrix-zoom" type="range" min="1" max="4" value="1" step="0.25"></label><button id="matrix-clear" type="button">Clear state/pair focus</button></div>
<p><span class="matrix-legend"></span>More observed witnesses → darker. Hover any cell for exact counts; click an axis state or cell to filter the tables.</p><p id="matrix-focus" class="muted">Focus: all states and pairs</p><p id="matrix-detail" aria-live="polite">No cell selected.</p><div class="matrix-viewport">{adjacency_matrix(graph)}</div></section>
<section><h2>Superstates</h2><p>Every available definition is shown. Reward mean and population variance use one outcome per visiting trajectory.</p>
<input type="search" data-table="states" aria-label="Search states" placeholder="Search state ID, name, or definition"><p id="states-count" class="muted"></p>
<div class="scroll"><table id="states"><thead><tr><th>State</th><th>Definition</th><th>Histories</th><th>Rollouts</th><th>Tasks</th><th>Mean reward</th><th>Variance</th></tr></thead><tbody>{"".join(state_rows)}</tbody></table></div></section>
<section><h2>Directed edges</h2><p>Observed witnesses and sampled applicability are separate evidence. No edge is universally certified.</p>
<input type="search" data-table="edges" aria-label="Search edges" placeholder="Search endpoints, operation, effect, or evidence status"><p id="edges-count" class="muted"></p>
<div class="scroll"><table id="edges"><thead><tr><th>Edge</th><th>Operation contract</th><th>Witnesses</th><th>Evidence</th><th>Witness reference subset</th></tr></thead><tbody>{"".join(edge_rows)}</tbody></table></div></section>
<section><h2>Independent checks</h2><p>Planned: {esc(report["independent_audits"]["planned"])}; completed: {esc(report["independent_audits"]["completed"])}; failed requests: {esc(report["independent_audits"]["failed_requests"])}.</p><ul>{audits}</ul>
<p>Audit model: {esc(models["independent_audit"]["name"] or None)}. Router revision: {esc(models["router"]["revision"] or None)}. Teacher revision: {esc(models["teacher"]["revision"] or None)}.</p>
<h2>Selected task specifications</h2><table><thead><tr><th>Draft</th><th>Feasibility review</th><th>Executed</th><th>Benchmark validated</th></tr></thead><tbody>{draft_rows}</tbody></table>
<p>These reviewed specification drafts are separate from the executed synthetic fixtures below. Draft generation errors: {esc(report["task_drafts"]["generation_error_count"])}.</p>
<h2>Locally executed synthetic tasks</h2><p>Stage: {esc(execution["status"])}. Reference-query agreement receipts: <b>{esc(execution["receipt_supported_consistent_count"])}</b> of {esc(execution["requested_examples"])} requested. Target reached: <b>{esc(reached)}</b>. Construction errors: {esc(execution["construction_error_count"])}. Failed requests, when recorded: {esc(execution["failed_requests"])}.</p>
<table><thead><tr><th>Task</th><th>Reference executed</th><th>Independent query executed</th><th>Results agree</th></tr></thead><tbody>{execution_rows}</tbody></table>
<p>Execution here means two reference queries ran on a new synthetic DuckDB fixture and agreed. It does not establish learner success or reproduce official benchmark verification.</p></section>
<section><h2>Interpretation and limits</h2><ul>{"".join("<li>" + esc(text) + "</li>" for text in report["limitations"])}</ul>
<details><summary>Snapshot warnings</summary><ul>{"".join("<li>" + esc(text) + "</li>" for text in report["warnings"]) or "<li>None.</li>"}</ul></details></section>
<footer>Self-contained report. No raw histories, task transcripts, assignment evidence, or runtime credentials are embedded. Graph witness references are explicitly limited to three per edge; the full local graph is unchanged.</footer>
"""
        + "</main><script>"
        + REPORT_JS
        + "</script></body></html>"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".report-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_report(run_dir: Path, corpus_dir: Path, output_dir: Path) -> dict:
    report = collect_report(run_dir, corpus_dir)
    _write(
        output_dir / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _write(output_dir / "README.md", markdown_report(report))
    _write(output_dir / "report.html", html_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/full_graph/run_v1"))
    parser.add_argument("--corpus", type=Path, default=Path("results/full_graph/corpus"))
    parser.add_argument("--output", type=Path, default=Path("reports/full-corpus-2026-09-17"))
    args = parser.parse_args()
    report = build_report(args.run, args.corpus, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.output),
                "histories": report["corpus"]["histories"],
                "proposals": report["optimization"]["proposals_observed"],
                "states": report["graph"]["node_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
