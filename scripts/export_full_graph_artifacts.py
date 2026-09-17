#!/usr/bin/env python3
"""Publish the complete graph's definitions, IDs, memberships, and outcome analysis.

This final-only exporter fails closed on incomplete or inconsistent results. It
does not run inference, fabricate missing artifacts, or export raw transcripts.
The default corpus contract is the entire 103-task/1,030-rollout collection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FULL_COUNTS = {"tasks": 103, "rollouts": 1030, "histories": 37532, "transitions": 36502}
FORMAT = "complete-superstate-research-export-v1"
SECRET_PATTERN = re.compile(
    rb"(?:\bsk-[A-Za-z0-9_-]{10,}\b|\bAKIA[A-Z0-9]{16}\b|"
    rb"(?i:api[_ -]?key|authorization|password)\s*[:=]\s*[^\s,;]{8,}|"
    rb"(?i:bearer)\s+[A-Za-z0-9._-]{16,})"
)
TRANSCRIPT_MARKERS = (
    b"=== USER MESSAGE ===",
    b"=== ASSISTANT MESSAGE ===",
    b"<full_source_history>",
    b"<actual_next_action_and_observation>",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def select(record: dict, keys: tuple[str, ...]) -> dict:
    return {key: record[key] for key in keys if key in record}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def assert_publishable(data: bytes, name: str) -> None:
    """Refuse recognizable secrets/transcript framing instead of changing data."""
    if SECRET_PATTERN.search(data):
        raise ValueError(f"Possible credential in allowlisted artifact {name}; publication refused")
    if any(marker in data for marker in TRANSCRIPT_MARKERS):
        raise ValueError(f"Raw transcript framing found in {name}; publication refused")


def report_builder():
    path = Path(__file__).with_name("build_full_graph_report.py")
    spec = importlib.util.spec_from_file_location("full_graph_export_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(state: dict) -> dict:
    return select(state, ("id", "name", "description", "exclusions"))


def _contract(edge: dict) -> dict:
    return select(edge, ("id", "source", "target", "operation", "effect", "bindings"))


def sampled_edge_eligibility(
    graph: dict, assignments: dict, jobs: list, results: list, summary: dict
) -> dict:
    """Recompute finite-sample eligibility; never trust a stored traversable flag alone."""
    from superstate_graphs.graph_analysis import aggregate_independent_audits

    if not isinstance(jobs, list) or not isinstance(results, list):
        raise ValueError("Sampled graph requires recorded audit jobs and results")
    job_index = {job["audit_id"]: job for job in jobs}
    if len(job_index) != len(jobs) or any(not isinstance(key, str) or not key for key in job_index):
        raise ValueError("Audit jobs have duplicate or invalid identifiers")
    edge_index = {edge["id"]: edge for edge in graph["edges"]}
    members = defaultdict(set)
    for hid, assignment in assignments.items():
        members[assignment["state_id"]].add(hid)
    observed_sources = defaultdict(set)
    for edge in graph["edges"]:
        observed_sources[edge["source"], edge["target"]].update(
            witness.get("source_id", witness.get("source_history_id"))
            for witness in edge.get("witnesses", [])
        )
    for job in jobs:
        if job["kind"] != "edge_source_applicability":
            continue
        edge = edge_index.get(job.get("edge_id"))
        if edge is None or job.get("state_ids") != [edge["source"], edge["target"]]:
            raise ValueError("Edge audit references an absent or mismatched operation")
        source_ids, target_ids = job.get("history_ids"), job.get("target_history_ids")
        if (
            not isinstance(source_ids, list)
            or len(source_ids) != 1
            or not isinstance(target_ids, list)
            or not target_ids
            or not set(source_ids) <= members[edge["source"]]
            or not set(target_ids) <= members[edge["target"]]
            or job.get("source_member_count") != len(members[edge["source"]])
            or job.get("source_observed_to_target")
            is not (source_ids[0] in observed_sources[edge["source"], edge["target"]])
        ):
            raise ValueError("Edge audit history membership or sampling metadata is inconsistent")
    for result in results:
        job = job_index.get(result.get("audit_id"))
        if job is None or result.get("kind") != job["kind"]:
            raise ValueError("Audit result does not match its planned job")
        if result.get("execution_status") == "error" and result.get("verdict") != "unknown":
            raise ValueError("Failed audit request cannot support or contradict an edge")
        if job["kind"] == "edge_source_applicability" and result.get("verdict") in {
            "supported",
            "contradicted",
        }:
            evidence = result.get("verified_evidence", [])
            if (
                result.get("schema_and_citations_checked") is not True
                or result.get("citation_errors") != []
                or not isinstance(evidence, list)
                or any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("quote"), str)
                    or not item["quote"].strip()
                    for item in evidence
                )
                or not set(job["history_ids"]) <= {item.get("history_id") for item in evidence}
                or any(
                    item.get("history_id")
                    not in set(job["history_ids"] + job["target_history_ids"])
                    for item in evidence
                )
            ):
                raise ValueError(
                    "Definitive edge audit lacks its validated source-citation receipt"
                )
    recomputed = aggregate_independent_audits(jobs, results, graph)
    for field in ("planned_checks", "completed_checks", "by_kind"):
        if summary.get(field) != recomputed[field]:
            raise ValueError("Independent audit summary disagrees with jobs/results: " + field)
    if summary.get("failed_requests") != sum(
        row.get("execution_status") == "error" for row in results
    ):
        raise ValueError("Independent audit failed-request count is inconsistent")
    saved_reports = {edge["edge_id"]: edge for edge in summary.get("edges", [])}
    if len(saved_reports) != len(summary.get("edges", [])) or set(saved_reports) != set(edge_index):
        raise ValueError("Independent edge audit summary census is incomplete or duplicated")
    comparison_fields = (
        "edge_id",
        "observed_witness_count",
        "independent_sampled_source_ids",
        "untraversed_sources_tested",
        "verdicts",
        "has_independent_counterexample",
    )
    eligible, excluded = set(), Counter()
    for report in recomputed["edges"]:
        edge = edge_index[report["edge_id"]]
        # reported_traversable is intentionally the PRE-audit flag. It is not
        # compared with the post-audit flag on graph.edges here.
        for saved in (saved_reports[edge["id"]], edge.get("audit", {})):
            if any(saved.get(field) != report[field] for field in comparison_fields):
                raise ValueError("Stored edge audit disagrees with recorded jobs/results")
        if edge["audit"].get("source_member_count") != len(members[edge["source"]]):
            raise ValueError("Edge audit source member count disagrees with assignments")
        plausible = edge.get("proposal", {}).get("universal_source_plausible")
        if type(plausible) is not bool:
            raise ValueError("Edge proposal lacks its recorded reusability verdict")
        verdicts = report["verdicts"]
        reasons = {
            "no_supported_source_check": not verdicts.get("supported", 0),
            "contradicted_source_check": bool(verdicts.get("contradicted", 0)),
            "unknown_source_check": bool(verdicts.get("unknown", 0)),
            "planned_source_check_not_run": bool(verdicts.get("not_run", 0)),
            "proposer_rejected": plausible is False,
        }
        expected = not any(reasons.values())
        if edge.get("traversable") is not expected:
            raise ValueError(
                "Stored traversable flag disagrees with recomputed sampled eligibility"
            )
        if expected:
            eligible.add(edge["id"])
        else:
            excluded.update(reason for reason, applies in reasons.items() if applies)
    return {
        "eligible_ids": eligible,
        "excluded_reason_counts_nonexclusive": dict(sorted(excluded.items())),
    }


def _candidate(candidate: dict) -> dict:
    """Preserve the selected routing specification exactly after a strict shape check."""
    if set(candidate) != {"state_spec", "edge_spec"}:
        raise ValueError("Unexpected candidate components in final export")
    state, edge = (json.loads(candidate[key]) for key in ("state_spec", "edge_spec"))
    if set(state) != {"router_instructions", "states"} or set(edge) != {"edges"}:
        raise ValueError("Unexpected fields inside selected candidate")
    if any(set(node) - {"id", "name", "description", "exclusions"} for node in state["states"]):
        raise ValueError("Unexpected state evidence fields inside selected candidate")
    if any(
        set(item) - {"id", "source", "target", "operation", "effect", "bindings"}
        for item in edge["edges"]
    ):
        raise ValueError("Unexpected edge evidence fields inside selected candidate")
    assert_publishable(encoded(candidate), "selected_candidate.json")
    return candidate


def accepted_candidate_archive(result: dict, seed: dict, selected: dict) -> dict:
    """Publish only GEPA's recorded candidate lineage and aggregate scores (pinned schema v2)."""
    required = (
        "candidates",
        "parents",
        "val_aggregate_scores",
        "best_idx",
        "validation_schema_version",
    )
    if not isinstance(result, dict) or any(key not in result for key in required):
        raise ValueError(
            "Accepted candidate archive requires recorded candidates, parents, scores, best_idx, and schema version"
        )
    if (
        type(result["validation_schema_version"]) is not int
        or result["validation_schema_version"] != 2
    ):
        raise ValueError("Accepted candidate archive requires pinned GEPA result schema version 2")
    candidates, parents, scores = (
        result[key] for key in ("candidates", "parents", "val_aggregate_scores")
    )
    if (
        not isinstance(candidates, list)
        or not candidates
        or not isinstance(parents, list)
        or not isinstance(scores, list)
        or len(candidates) != len(parents)
        or len(candidates) != len(scores)
    ):
        raise ValueError("Accepted candidate/parent/score cardinalities must match and be nonempty")
    validated = [_candidate(candidate) for candidate in candidates]
    if validated[0] != _candidate(seed):
        raise ValueError("Recorded candidate index 0 differs from seed_candidate.json")
    if parents[0] != [None]:
        raise ValueError("Recorded seed parent row must be [null] under pinned GEPA schema")
    for index, row in enumerate(parents[1:], start=1):
        if (
            not isinstance(row, list)
            or not row
            or any(
                not isinstance(parent, int) or isinstance(parent, bool) or not 0 <= parent < index
                for parent in row
            )
            or len(row) != len(set(row))
        ):
            raise ValueError(
                "Each accepted candidate must reference distinct earlier parent indices"
            )
    if any(
        isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        for score in scores
    ):
        raise ValueError("Accepted candidate aggregate Pareto scores must be finite numbers")
    best = result["best_idx"]
    if (
        not isinstance(best, int)
        or isinstance(best, bool)
        or not 0 <= best < len(candidates)
        or best != max(range(len(scores)), key=scores.__getitem__)
    ):
        raise ValueError(
            "Recorded best_idx disagrees with pinned GEPA first-maximum score selection"
        )
    if validated[best] != _candidate(selected):
        raise ValueError("Recorded best candidate differs from selected_candidate.json")
    return {
        "format": "gepa-accepted-graph-candidate-archive-v1",
        "validation_schema_version": result["validation_schema_version"],
        "best_idx": best,
        "candidates": validated,
        "parents": [list(row) for row in parents],
        "val_aggregate_scores": list(scores),
        "candidate_sha256": [digest_bytes(encoded(candidate)) for candidate in validated],
        "scope": "Seed and candidates admitted to the recorded GEPA state; excludes rejected proposals and does not imply every archived candidate remains on the Pareto frontier",
        "index_scope": "Zero-based recorded candidate indices, not proposal-attempt numbers; continuation receipt boundaries cannot be mapped to candidate indices without separate evidence",
        "score_scope": "Aggregate Pareto-validation scores used adaptively during optimization, not held-out test scores",
        "omitted_fields": [
            "per-rollout scores and outputs",
            "reflection datasets and feedback",
            "run-directory paths",
        ],
    }


def _reward_provenance(record: dict) -> dict:
    return select(
        record,
        (
            "kind",
            "source",
            "source_sha256",
            "field",
            "assignment_record",
            "assignment_sha256",
            "original_verifier_executed",
        ),
    )


def _moments(record: dict) -> dict:
    result = select(
        record, ("weight", "mean", "population_variance", "bessel_corrected_descriptive_variance")
    )
    if "task_cluster_bootstrap" in record:
        result["task_cluster_bootstrap"] = select(
            record["task_cluster_bootstrap"],
            (
                "method",
                "confidence",
                "requested_draws",
                "task_clusters",
                "mean_interval",
                "population_variance_interval",
                "status",
                "completed_draws",
            ),
        )
    return result


def _variance_state(record: dict, *, sensitivity: bool = True) -> dict:
    result = select(
        record,
        (
            "state_id",
            "member_histories",
            "distinct_visiting_rollouts",
            "distinct_visiting_tasks",
            "rewarded_rollouts",
            "rewarded_tasks",
            "missing_reward_rollouts",
            "manual_reward_rollouts",
            "excluded_manual_reward_rollouts",
            "binary_rewards_only",
        ),
    )
    for key in ("history_weighted", "trajectory_deduplicated", "task_balanced"):
        if key in record:
            result[key] = _moments(record[key])
    if "task_balanced_decomposition" in record:
        result["task_balanced_decomposition"] = select(
            record["task_balanced_decomposition"],
            (
                "mean_within_task_population_variance",
                "between_task_mean_variance",
                "interpretation",
            ),
        )
    result["per_task"] = [
        select(row, ("task_id", "rewarded_rollouts", "mean", "population_variance"))
        for row in record.get("per_task", [])
    ]
    if sensitivity and "sensitivity_excluding_manual_rewards" in record:
        result["sensitivity_excluding_manual_rewards"] = _variance_state(
            record["sensitivity_excluding_manual_rewards"], sensitivity=False
        )
    return result


def _variance(value: dict) -> dict:
    result = select(
        value,
        (
            "format_version",
            "histories",
            "assigned_histories",
            "missing_assignment_count",
            "unassigned_count",
            "missing_assignment_ids",
            "unassigned_ids",
            "limitation",
            "formation_uses_outcomes",
            "graph_scope",
        ),
    )
    result["states"] = [_variance_state(row) for row in value["states"]]
    result["weighting"] = select(
        value.get("weighting", {}),
        (
            "history_weighted",
            "trajectory_deduplicated",
            "task_balanced",
        ),
    )
    return result


def validate_variance_moments(value: dict, histories: dict, assignments: dict) -> None:
    """Recompute descriptive moments from immutable outcomes; no model calls or bootstrap reruns."""
    groups = defaultdict(list)
    for hid, history in histories.items():
        groups[assignments[hid]["state_id"]].append(history)

    def check(actual: Any, expected: float | None, label: str) -> None:
        if expected is None:
            valid = actual is None
        else:
            valid = (
                not isinstance(actual, bool)
                and isinstance(actual, (int, float))
                and math.isfinite(actual)
                and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10)
            )
        if not valid:
            raise ValueError(f"Reward variance moment mismatch: {label}")

    def verify(row: dict, members: list[dict], *, exclude_manual: bool) -> None:
        by_rollout = {}
        for history in members:
            rid = history["rollout_id"]
            reward = history["reward"]
            identity = (history["task_id"], reward, history["reward_provenance"].get("kind", ""))
            if rid in by_rollout and by_rollout[rid] != identity:
                raise ValueError("One rollout has inconsistent terminal reward metadata")
            by_rollout[rid] = identity
        valid = {}
        for rid, (task, reward, kind) in by_rollout.items():
            if exclude_manual and str(kind).startswith("manual"):
                continue
            if reward is None:
                continue
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                raise ValueError("Corpus contains an invalid terminal reward")
            valid[rid] = (task, reward)
        by_task = defaultdict(list)
        for task, reward in valid.values():
            by_task[task].append(reward)
        values = {
            "history_weighted": [
                (valid[h["rollout_id"]][1], 1.0) for h in members if h["rollout_id"] in valid
            ],
            "trajectory_deduplicated": [(reward, 1.0) for _, reward in valid.values()],
            "task_balanced": [
                (reward, 1.0 / len(rewards)) for rewards in by_task.values() for reward in rewards
            ],
        }
        for key, weighted in values.items():
            weight = math.fsum(w for _, w in weighted)
            mean = math.fsum(reward * w for reward, w in weighted) / weight if weight else None
            variance = (
                max(
                    0.0,
                    math.fsum(w * reward * reward for reward, w in weighted) / weight - mean * mean,
                )
                if weight
                else None
            )
            actual = row.get(key, {})
            label = f"{row.get('state_id', 'manual-excluded')}/{key}"
            for field, expected in (
                ("weight", weight),
                ("mean", mean),
                ("population_variance", variance),
            ):
                check(actual.get(field), expected, f"{label}/{field}")
        task_ids = [task.get("task_id") for task in row.get("per_task", [])]
        if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(by_task):
            raise ValueError("Variance per-task rows do not cover all rewarded visiting tasks")
        for task in row.get("per_task", []):
            rewards = by_task.get(task.get("task_id"), [])
            if not rewards:
                raise ValueError("Variance per-task row has no rewarded visiting rollouts")
            mean = math.fsum(rewards) / len(rewards)
            check(task.get("rewarded_rollouts"), len(rewards), "per_task/rewarded_rollouts")
            check(task.get("mean"), mean, "per_task/mean")
            check(
                task.get("population_variance"),
                max(0.0, math.fsum(v * v for v in rewards) / len(rewards) - mean * mean),
                "per_task/population_variance",
            )

    for row in value["states"]:
        members = groups[row["state_id"]]
        verify(row, members, exclude_manual=False)
        sensitivity = row.get("sensitivity_excluding_manual_rewards")
        if not isinstance(sensitivity, dict):
            raise ValueError("Reward variance is missing manual-grade sensitivity analysis")
        verify(sensitivity, members, exclude_manual=True)


def export_artifacts(
    run_dir: Path,
    corpus_dir: Path,
    output_dir: Path,
    *,
    include_tasks: bool = False,
    expected_counts: dict | None = None,
) -> dict:
    """Export once, atomically. ``expected_counts`` supports small offline test corpora."""
    run_dir, corpus_dir, output_dir = map(Path, (run_dir, corpus_dir, output_dir))
    expected_counts = dict(FULL_COUNTS if expected_counts is None else expected_counts)
    report = report_builder().collect_report(run_dir, corpus_dir)
    if report["status"] != "complete":
        missing = sorted(
            key for key, complete in report["completion_checks"].items() if not complete
        )
        raise ValueError("Final export requires complete verified artifacts: " + ", ".join(missing))
    for key, count in expected_counts.items():
        if report["corpus"].get(key) != count:
            raise ValueError(f"Full-corpus export expected {count} {key}; count does not match")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace an existing final research export: {output_dir}"
        )

    source_hashes = dict(report["source_artifact_sha256"])

    def load(name: str, *, corpus: bool = False) -> Any:
        raw = ((corpus_dir if corpus else run_dir) / name).read_bytes()
        source_hashes[("corpus/" if corpus else "run/") + name] = digest_bytes(raw)
        return json.loads(raw)

    graph = load("graph.json")
    assignments = load("assignments_all.json")
    contract = load("run_contract.json")
    corpus_manifest = load("manifest.json", corpus=True)
    for name in ("histories.jsonl", "transitions.jsonl", "rollouts.jsonl", "splits.json"):
        with (corpus_dir / name).open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        expected_hash = corpus_manifest.get("artifact_sha256", {}).get(name)
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"Corpus artifact hash mismatch: {name}")
        source_hashes["corpus/" + name] = actual_hash
    selected = _candidate(load("optimized_candidate.json"))
    if digest_bytes(encoded(selected)) != report["frozen_test"]["candidate_hash"]:
        raise ValueError("Selected specification differs from the frozen-test candidate")
    if not (run_dir / "seed_candidate.json").is_file():
        raise ValueError("Final candidate archive requires the recorded seed_candidate.json")
    seed = _candidate(load("seed_candidate.json"))
    archive = accepted_candidate_archive(load("gepa_result.json"), seed, selected)
    archive["source_gepa_result_sha256"] = source_hashes["run/gepa_result.json"]
    stages = [_candidate(stage) for stage in graph.get("routing_stages", [selected])]
    if not stages or stages[0] != selected:
        raise ValueError("Routing-stage chain does not begin with the selected candidate")
    splits = load("splits.json", corpus=True)
    split_export = {"seed": splits.get("seed")}
    for split in ("train", "pareto", "test"):
        split_export[split] = select(splits[split], ("task_ids", "rollout_ids"))
    task_membership = [
        task for split in ("train", "pareto", "test") for task in splits[split]["task_ids"]
    ]
    rollout_membership = [
        rid for split in ("train", "pareto", "test") for rid in splits[split]["rollout_ids"]
    ]
    if len(set(task_membership)) != len(task_membership) or len(set(rollout_membership)) != len(
        rollout_membership
    ):
        raise ValueError("Exported task/rollout splits overlap")
    if (
        len(task_membership) != expected_counts["tasks"]
        or len(rollout_membership) != expected_counts["rollouts"]
    ):
        raise ValueError("Split membership does not cover the entire corpus")
    rollout_to_split = {
        rid: split for split in ("train", "pareto", "test") for rid in splits[split]["rollout_ids"]
    }
    history_index = {}
    history_file = corpus_dir / "histories.jsonl"
    with history_file.open() as stream:
        for line in stream:
            history = json.loads(line)
            if history["history_id"] in history_index:
                raise ValueError("Duplicate history ID in source corpus")
            history_index[history["history_id"]] = history
    if set(history_index) != set(assignments):
        raise ValueError("Assignment/history census changed during export")
    if {row["task_id"] for row in history_index.values()} != set(task_membership):
        raise ValueError("History task identities differ from complete split membership")
    states = {_state(node)["id"]: _state(node) for node in graph["nodes"]}
    if len(states) != len(graph["nodes"]) or len({edge["id"] for edge in graph["edges"]}) != len(
        graph["edges"]
    ):
        raise ValueError("Final graph must have unique state and edge identifiers")
    classified, members = [], defaultdict(list)
    for hid, history in sorted(history_index.items()):
        assignment = assignments[hid]
        for key in ("history_id", "rollout_id", "task_id", "split", "step"):
            if assignment.get(key) != history.get(key):
                raise ValueError(f"Assignment and source history metadata disagree: {hid}, {key}")
        if history["split"] != rollout_to_split.get(history["rollout_id"]):
            raise ValueError("History and rollout split assignments disagree")
        if history["task_id"] not in splits[history["split"]]["task_ids"]:
            raise ValueError("History task crosses the declared original-task split")
        sid = assignment.get("state_id")
        if sid not in states:
            raise ValueError("Every exported history must have an existing non-null superstate")
        row = select(
            history,
            (
                "history_id",
                "rollout_id",
                "task_id",
                "split",
                "step",
                "prefix_char_end",
                "prefix_sha256",
                "last_observed_history",
            ),
        )
        row.update(
            {
                "state_id": sid,
                "routing_stage": assignment.get("routing_stage", 0),
                "terminal_reward": history["reward"],
                "reward_provenance": _reward_provenance(history["reward_provenance"]),
            }
        )
        classified.append(row)
        members[sid].append(hid)
    nodes = [
        {
            **states[sid],
            "member_history_ids": members.get(sid, []),
            "member_history_count": len(members.get(sid, [])),
        }
        for sid in sorted(states)
    ]
    variance = load("state_reward_variance.json")
    validate_variance_moments(variance, history_index, assignments)

    expected_transitions = {}
    with (corpus_dir / "transitions.jsonl").open() as stream:
        for line in stream:
            transition = json.loads(line)
            if transition["transition_id"] in expected_transitions:
                raise ValueError("Duplicate transition ID in source corpus")
            expected_transitions[transition["transition_id"]] = transition
    witnesses, edges = [], []
    public_edges = {edge["id"]: edge for edge in report["graph"]["edges"]}
    for edge in graph["edges"]:
        if edge["source"] not in states or edge["target"] not in states:
            raise ValueError("Export contains a dangling edge")
        edge_witnesses = []
        for witness in edge.get("witnesses", []):
            tid = witness["transition_id"]
            expected = expected_transitions.get(tid)
            if expected is None:
                raise ValueError("Witness is absent from the corpus transition index")
            source = witness.get("source_id", witness.get("source_history_id"))
            target = witness.get("target_id", witness.get("target_history_id"))
            if (
                source != expected["source_history_id"]
                or target != expected["target_history_id"]
                or assignments[source]["state_id"] != edge["source"]
                or assignments[target]["state_id"] != edge["target"]
                or witness["rollout_id"] != expected["rollout_id"]
                or witness["task_id"] != expected["task_id"]
                or witness["step"] != expected["step"]
            ):
                raise ValueError(
                    "Witness boundary, membership, or rollout identity disagrees with corpus"
                )
            witnesses.append(
                {
                    "transition_id": tid,
                    "edge_id": edge["id"],
                    "source_history_id": source,
                    "target_history_id": target,
                    "source_state_id": edge["source"],
                    "target_state_id": edge["target"],
                    "rollout_id": witness["rollout_id"],
                    "task_id": witness["task_id"],
                    "split": expected["split"],
                    "step": witness["step"],
                }
            )
            edge_witnesses.append(tid)
        audit = select(
            edge.get("audit", {}),
            (
                "method",
                "source_member_count",
                "verdicts",
                "independent_sampled_source_ids",
                "untraversed_sources_tested",
                "has_independent_counterexample",
            ),
        )
        audit["universal_contract_certified"] = False
        audit["request_errors"] = public_edges[edge["id"]]["source_audit_request_errors"]
        audit["non_error_unknown"] = public_edges[edge["id"]]["source_audit_non_error_unknown"]
        audit["sampling_interpretation"] = (
            "Sampled source IDs and untraversed_sources_tested include result records for failed requests; verdicts and request_errors distinguish support, semantic uncertainty, and failures."
        )
        edges.append(
            {
                **_contract(edge),
                "witness_ids": edge_witnesses,
                "witness_count": len(edge_witnesses),
                "distinct_task_count": len({w["task_id"] for w in edge.get("witnesses", [])}),
                "traversable": edge.get("traversable") is True,
                "evidence_status": edge.get("evidence_status"),
                "audit": audit,
            }
        )
    witness_ids = [w["transition_id"] for w in witnesses]
    if (
        len(witness_ids) != len(set(witness_ids))
        or set(witness_ids) != set(expected_transitions)
        or len(witness_ids) != expected_counts["transitions"]
    ):
        raise ValueError("Export must contain each recorded transition exactly once")
    eligibility = sampled_edge_eligibility(
        graph,
        assignments,
        load("independent_audit_jobs.json"),
        load("independent_audit_results.json"),
        load("independent_audit_summary.json"),
    )
    sampled_graph = {
        "format": "sampled-supported-superstate-graph-v1",
        "directed": True,
        "nodes": nodes,
        "edges": sorted(
            (edge for edge in edges if edge["id"] in eligibility["eligible_ids"]),
            key=lambda edge: edge["id"],
        ),
        "node_count": len(nodes),
        "edge_count": len(eligibility["eligible_ids"]),
        "observed_graph_edge_count": len(edges),
        "excluded_edge_count": len(edges) - len(eligibility["eligible_ids"]),
        "excluded_reason_counts_nonexclusive": eligibility["excluded_reason_counts_nonexclusive"],
        "eligibility_rule": "At least one supported source audit; every planned source audit supported; no contradicted, unknown, not-run, or proposer-rejected contract",
        "semantics": "Finite sampled LLM support for proposing paths. Every retained edge has observed witnesses; the sample does not establish applicability to every source history or execution of a composed path.",
        "universal_applicability_certified": False,
        "executable_path_certified": False,
        "all_nodes_retained": True,
        "isolated_nodes_and_empty_edge_sets_are_valid": True,
        "evidence_validation": "Recomputed from recorded audit jobs/results and proposer verdicts, checked against audit summaries, membership identities, and final traversable flags. Source-citation validation receipts are checked; source quotations are not published or rejudged.",
        "source_artifact_sha256": {
            name: source_hashes["run/" + name]
            for name in (
                "graph.json",
                "assignments_all.json",
                "independent_audit_jobs.json",
                "independent_audit_results.json",
                "independent_audit_summary.json",
            )
        },
    }

    rollout_rows = []
    with (corpus_dir / "rollouts.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            public = select(
                row,
                (
                    "id",
                    "rollout_id",
                    "task_id",
                    "horizon_task_id",
                    "task_version_id",
                    "cohort_id",
                    "split",
                    "reward",
                    "source_messages_sha256",
                    "transcript_sha256",
                    "history_count",
                    "message_count",
                ),
            )
            public["reward_provenance"] = _reward_provenance(row["reward_provenance"])
            rollout_rows.append(public)
    if {r["id"] for r in rollout_rows} != set(rollout_membership) or len(
        rollout_rows
    ) != expected_counts["rollouts"]:
        raise ValueError("Export rollout census differs from split membership")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".complete-graph-export-", dir=output_dir.parent))
    files: dict[str, dict] = {}

    def save_bytes(name: str, data: bytes, *, rows: int | None = None) -> None:
        assert_publishable(data, name)
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = {"sha256": digest_bytes(data), "bytes": len(data)}
        if rows is not None:
            files[name]["records"] = rows

    def save_json(name: str, value: Any) -> None:
        save_bytes(name, encoded(value) + b"\n")

    def save_jsonl(name: str, records: list[dict]) -> None:
        data = b"".join(encoded(row) + b"\n" for row in records)
        assert_publishable(data, name)
        compressed = gzip.compress(data, mtime=0)
        path = staging / name
        path.write_bytes(compressed)
        files[name] = {
            "sha256": digest_bytes(compressed),
            "bytes": len(compressed),
            "records": len(records),
            "uncompressed_sha256": digest_bytes(data),
        }

    try:
        save_jsonl("nodes.jsonl.gz", nodes)
        save_jsonl("edges.jsonl.gz", sorted(edges, key=lambda edge: edge["id"]))
        save_json("sampled_supported_graph.json", sampled_graph)
        save_jsonl("assignments.jsonl.gz", classified)
        save_jsonl("witnesses.jsonl.gz", sorted(witnesses, key=lambda w: w["transition_id"]))
        save_jsonl("rollouts.jsonl.gz", sorted(rollout_rows, key=lambda row: row["id"]))
        save_json("selected_candidate.json", selected)
        save_json("seed_candidate.json", seed)
        save_json("accepted_candidate_archive.json", archive)
        save_json("routing_stages.json", stages)
        save_json("splits.json", split_export)
        save_json("state_reward_variance.json", _variance(variance))
        if (run_dir / "state_reward_variance_by_split.json").exists():
            split_variance = load("state_reward_variance_by_split.json")
            if set(split_variance) != {history["split"] for history in history_index.values()}:
                raise ValueError("Split variance does not cover the corpus splits")
            for split, value in split_variance.items():
                subset = {
                    hid: history
                    for hid, history in history_index.items()
                    if history["split"] == split
                }
                subset_assignments = {hid: assignments[hid] for hid in subset}
                if not report_builder().variance_census_verified(value, subset_assignments):
                    raise ValueError("Split variance membership census is incomplete")
                validate_variance_moments(value, subset, subset_assignments)
            save_json(
                "state_reward_variance_by_split.json",
                {
                    split: _variance(value)
                    for split, value in split_variance.items()
                    if split in ("train", "pareto", "test")
                },
            )
        save_json(
            "evaluation_summary.json",
            {
                "optimization": report["optimization"],
                "frozen_test": report["frozen_test"],
                "independent_audits": report["independent_audits"],
                "task_drafts": report["task_drafts"],
                "executable_tasks": report["executable_tasks"],
            },
        )
        if report["heldout_comparison"]["identities_and_census_verified"]:
            save_json("heldout_comparison.json", report["heldout_comparison"])
        save_json("usage_accounting.json", report["usage"])
        if report["optimizer_continuations"]["declared"]:
            save_json("optimizer_continuations.json", report["optimizer_continuations"]["records"])
        task_receipts = []
        if include_tasks and (run_dir / "executable_task_summary.json").exists():
            summary = load("executable_task_summary.json")
            for item in summary.get("selected_examples", []):
                if item.get("status") != "locally_executed_consistent":
                    continue
                path_id = item["path_id"]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", path_id):
                    raise ValueError("Invalid executable-task path identifier")
                source = run_dir / "executable_tasks" / path_id
                receipt = read_json(source / "result.json")
                validation = read_json(source / "local_validation.json")
                if (
                    receipt.get("status") != "locally_executed_consistent"
                    or validation.get("status") != "locally_executed_consistent"
                    or not validation.get("reference_query_executed")
                    or not validation.get("independent_query_executed")
                    or not validation.get("queries_agree")
                ):
                    raise ValueError(
                        "Executable-task receipt does not establish local query agreement"
                    )
                base = f"executable_tasks/{path_id}/"
                source_names = (
                    "instruction.md",
                    "fixture.duckdb",
                    "oracle.sql",
                    "independent.sql",
                    "expected_result.json",
                    "local_validation.json",
                    "task.json",
                    "source_path.json",
                )
                raw_files = {}
                for name in source_names:
                    raw = (source / name).read_bytes()
                    expected = receipt.get("artifact_sha256", {}).get(name)
                    if not expected or digest_bytes(raw) != expected:
                        raise ValueError(
                            f"Executable-task artifact hash mismatch: {path_id}/{name}"
                        )
                    source_hashes[f"run/executable_tasks/{path_id}/{name}"] = expected
                    raw_files[name] = raw
                for name in (
                    "instruction.md",
                    "fixture.duckdb",
                    "oracle.sql",
                    "independent.sql",
                    "expected_result.json",
                ):
                    save_bytes(base + name, raw_files[name])
                task_spec = json.loads(raw_files["task.json"])
                source_path = json.loads(raw_files["source_path.json"])
                path_steps = source_path.get("transitions", [])
                path_edge_ids = [step.get("transition_id") for step in path_steps]
                edge_lookup = {edge["id"]: edge for edge in edges}
                if (
                    not path_steps
                    or task_spec.get("path_id") != path_id
                    or source_path.get("path_id") != path_id
                    or task_spec.get("path_transition_ids") != path_edge_ids
                    or any(eid not in edge_lookup for eid in path_edge_ids)
                ):
                    raise ValueError(
                        "Executable task path does not match final graph edge identifiers"
                    )
                path_state_ids = [edge_lookup[path_edge_ids[0]]["source"]]
                path_witnesses = []
                witness_lookup = {
                    (w["edge_id"], w["source_history_id"], w["target_history_id"]): w
                    for w in witnesses
                }
                for edge_id, step in zip(path_edge_ids, path_steps):
                    edge = edge_lookup[edge_id]
                    if edge["source"] != path_state_ids[-1] or not edge["traversable"]:
                        raise ValueError(
                            "Executable task uses a disconnected or ineligible graph path"
                        )
                    path_state_ids.append(edge["target"])
                    key = edge_id, step.get("source_history_id"), step.get("target_history_id")
                    witness = witness_lookup.get(key)
                    if witness is None:
                        raise ValueError(
                            "Executable task path witness is absent from the full graph ledger"
                        )
                    path_witnesses.append(
                        select(
                            witness,
                            (
                                "edge_id",
                                "transition_id",
                                "source_history_id",
                                "target_history_id",
                                "rollout_id",
                                "task_id",
                            ),
                        )
                    )
                if source_path.get("state_ids") != path_state_ids:
                    raise ValueError(
                        "Executable task state path disagrees with graph edge endpoints"
                    )
                save_json(
                    base + "source_path.json",
                    {
                        "path_id": path_id,
                        "state_ids": path_state_ids,
                        "graph_edge_ids": path_edge_ids,
                        "witnesses": path_witnesses,
                        "cross_task": len({w["task_id"] for w in path_witnesses}) > 1,
                        "evidence_type": "Recorded transition witnesses; composed source histories were not executed",
                        "universal_contract_certified": False,
                    },
                )
                save_json(
                    base + "task.json",
                    select(
                        task_spec,
                        (
                            "format",
                            "status",
                            "title",
                            "path_id",
                            "ordered_output",
                            "synthetic_fixture",
                            "output_columns",
                            "path_transition_ids",
                            "preserved_challenges",
                            "adaptations",
                        ),
                    ),
                )
                save_json(
                    base + "local_validation.json",
                    select(
                        validation,
                        (
                            "format",
                            "status",
                            "reference_query_executed",
                            "independent_query_executed",
                            "reference_result",
                            "independent_result",
                            "nonempty_result",
                            "output_schema_matches",
                            "queries_agree",
                            "elapsed_seconds",
                            "synthetic_fixture",
                            "learner_evaluated",
                            "original_benchmark_reproduced",
                            "official_benchmark_verifier",
                            "universal_graph_contract_certified",
                            "limitations",
                        ),
                    ),
                )
                task_receipts.append(
                    {
                        "path_id": path_id,
                        "status": "locally_executed_consistent",
                        "graph_edge_ids": path_edge_ids,
                        "learner_evaluated": False,
                        "original_benchmark_reproduced": False,
                    }
                )
                save_bytes(
                    base + "README.md",
                    (
                        "# Executed synthetic SQL task\n\nGive the learner only `instruction.md` and "
                        "`fixture.duckdb`. Reference SQL and expected output are evaluation-only.\n\n"
                        "Verify a submitted query from the repository environment:\n\n"
                        "```sh\npython -m superstate_graphs.graph_task_examples verify "
                        "--task-dir . --submission answer.sql\n```\n\n"
                        "Two independently prompted reference queries ran locally and agreed. This "
                        "does not demonstrate learner success, official-benchmark reproduction, or "
                        "universal graph applicability.\n"
                    ).encode(),
                )
        provenance = {
            "format": FORMAT,
            "corpus_counts": expected_counts,
            "model": report["model"],
            "model_provenance": report["model_provenance"],
            "run_configuration": select(
                contract,
                (
                    "rollouts",
                    "histories",
                    "train_rollouts",
                    "pareto_rollouts",
                    "test_rollouts",
                    "max_proposals",
                    "minibatch",
                    "optimization_hours",
                    "model",
                    "revision",
                    "teacher_model",
                    "teacher_revision",
                    "teacher_reasoning",
                    "formation_uses_rewards",
                    "full_prefixes",
                    "retention_scope",
                    "test_then_completion",
                ),
            ),
            "corpus": report["corpus"],
            "source_artifact_sha256": source_hashes,
            "formation_uses_terminal_rewards": False,
            "graph_scope": "All-corpus transductive completion after frozen held-out evaluation",
            "graph_stages": report["graph_stages"],
            "seed_selected_specification_comparison": report["optimization"][
                "seed_selected_specification_comparison"
            ],
            "optimizer_continuations": report["optimizer_continuations"],
            "edge_construction_scope": "Final observed endpoint-pair contracts are reconstructed after frozen evaluation; selected_candidate.json preserves the edge specification evaluated by GEPA",
            "retention_scope": "Previously evaluated training histories per candidate",
            "universal_contract_certified": False,
            "exported_executable_tasks": task_receipts,
            "omitted_fields": [
                "raw query and trajectory content",
                "assignment evidence text",
                "judge prompts and feedback",
                "runtime connection details and credentials",
            ],
        }
        save_json("provenance.json", provenance)
        save_bytes(
            "README.md",
            (
                "# Complete superstate graph research artifacts\n\n"
                f"This export contains **{expected_counts['tasks']:,} original task IDs, "
                f"{expected_counts['rollouts']:,} rollouts, {expected_counts['histories']:,} complete history "
                f"assignments, and {expected_counts['transitions']:,} recorded transition witnesses**. "
                "Every membership and witness ID is retained; no witness subsampling is used here. "
                "Original transcript contents are not redistributed.\n\n"
                "| File | Content |\n|---|---|\n"
                "| nodes.jsonl.gz | All state definitions and complete member-history ID lists |\n"
                "| edges.jsonl.gz | All directed operation contracts and complete witness-ID lists |\n"
                "| sampled_supported_graph.json | Ready-to-use directed graph with all nodes and only edges whose finite sampled eligibility was independently recomputed from audit records |\n"
                "| assignments.jsonl.gz | Every history-to-state mapping, prefix hashes, split, and terminal outcome metadata |\n"
                "| witnesses.jsonl.gz | Every transition, with exact history/state endpoints and rollout identity |\n"
                "| rollouts.jsonl.gz | Complete rollout metadata and reward provenance, excluding messages |\n"
                "| selected_candidate.json; routing_stages.json | Exact selected specification and ordered completion stages |\n"
                "| seed_candidate.json; accepted_candidate_archive.json | Validated baseline and every recorded GEPA candidate specification, parent indices, and aggregate Pareto scores; no reflection evidence |\n"
                "| splits.json | Full original-task and rollout split membership |\n"
                "| state_reward_variance.json | Full outcome moments, weightings, intervals, and manual-grade sensitivity |\n"
                "| evaluation_summary.json; provenance.json | Observed evaluation summaries and source fingerprints |\n"
                "| usage_accounting.json | Deduplicated successful cached-request usage and separate overlapping process snapshots; not billing |\n"
                "| optimizer_continuations.json (when declared) | Allowlisted method-repair receipts, boundaries, source/checkpoint hashes, and declared invariants |\n"
                "| heldout_comparison.json (when available) | Paired seed-versus-selected held-out aggregates, task-cluster intervals, and common rollout IDs; no inference evidence |\n"
                "| manifest.json | Exported-file SHA-256 hashes and record counts |\n\n"
                "JSONL compression uses gzip with a fixed timestamp. Read with Python's `gzip.open(path, 'rt')`; "
                "each line is one JSON object. The `history_id` and `transition_id` columns provide lossless joins. "
                "The first routing stage is the optimized candidate; later stages classify only histories "
                "left unassigned by earlier stages. Final graph completion is transductive.\n\n"
                "For path proposals, load `sampled_supported_graph.json` and traverse its `edges` by "
                "`source` and `target`. It retains all nodes, including isolated nodes, and can have zero "
                "eligible edges. Every included edge has at least one supported source check and no "
                "contradicted, unknown, missing planned check, or proposer rejection. This is finite "
                "sampled LLM support, not universal source applicability or proof that a composed path "
                "executes. The complete observed graph and witness ledger remain in the unfiltered "
                "JSONL files.\n\n"
                "The accepted-candidate archive preserves the final GEPA result's original indices, "
                "parent rows, and aggregate Pareto-validation scores. Its candidate zero equals the "
                "published seed, and best_idx equals the published selected candidate. It does not "
                "include rejected proposals or imply that all archived versions remain Pareto-optimal. "
                "Candidate indices are not proposal-attempt numbers; continuation boundaries do not "
                "directly index the candidate archive. "
                "Any declared method continuation remains disclosed separately in optimizer_continuations.json.\n\n"
                "The source is the audited historical Horizon/Sonnet 4.5 counterpart corpus, not a rerun "
                "of the current public benchmark. One terminal zero comes from documented manual trace review. "
                "Formation excludes rewards; descriptive reward statistics are attached afterward. "
                "Trajectory-deduplicated and task-balanced statistics address repeated visits and uneven task representation.\n\n"
                "Held-out scores precede all-corpus completion. Independent audits use a separate frozen "
                "LLM procedure, not an independent model family. Traversable flags mean sampled support "
                "for drafting, not universal applicability or guaranteed executable compositions. "
                "No post-training lift is claimed.\n\n"
                f"Optional locally executed synthetic tasks included: **{len(task_receipts)}**. "
                "Their SQL agreement checks are not learner evaluations or official benchmark verification. "
                "See per-task receipts and instructions when present.\n"
            ).encode(),
        )
        manifest = {
            "format": FORMAT,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_counts": expected_counts,
            "nodes": len(nodes),
            "edges": len(edges),
            "assignment_records": len(classified),
            "witness_records": len(witnesses),
            "executable_tasks": len(task_receipts),
            "files": files,
        }
        (staging / "manifest.json").write_bytes(encoded(manifest) + b"\n")
        if output_dir.exists():
            raise FileExistsError(
                "Another process published this export while it was being prepared"
            )
        staging.rename(output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/full_graph/run_v1"))
    parser.add_argument("--corpus", type=Path, default=Path("results/full_graph/corpus"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/full-corpus-2026-09-17/artifacts")
    )
    parser.add_argument("--include-tasks", action="store_true")
    args = parser.parse_args()
    manifest = export_artifacts(
        args.run, args.corpus, args.output, include_tasks=args.include_tasks
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "corpus_counts",
                    "nodes",
                    "edges",
                    "assignment_records",
                    "witness_records",
                    "executable_tasks",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
