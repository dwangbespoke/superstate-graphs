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
    variance = read("state_reward_variance.json")
    audit = read("independent_audit_summary.json")
    tasks = read("task_draft_summary.json")
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
        "independent_audits": audit is not None,
        "task_drafts": tasks is not None,
        "completion_record": completion is not None and completion.get("status") == "complete",
        "exact_history_census": exact and nulls == 0 and not unknown_states,
        "transition_census": transition_census,
    }
    complete = all(required.values())
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
        "model": {
            "name": safe_text(contract.get("model")),
            "revision": safe_text(contract.get("revision")),
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
        f"Snapshot: {report['generated_at_utc']}. Model: {md(report['model']['name'])}.",
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
        "## Interpretation and limits",
        "",
        *["- " + text for text in report["limitations"]],
        "",
        "The searchable [HTML report](report.html) contains every available node and edge. "
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
            "<tr>"
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
            "<tr>"
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
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Full-corpus superstate graph</title>
<style>body{font:15px/1.55 system-ui,sans-serif;color:#203047;background:#f3f6fa;margin:0}main{max-width:1500px;margin:auto;padding:36px}h1{font-size:36px;line-height:1.15;margin:8px 0}h2{margin-top:40px}a{color:#12677b}header,section{background:white;border:1px solid #dce4ee;border-radius:12px;padding:24px;margin-bottom:20px}.status{display:inline-block;background:#fff0ca;padding:5px 12px;border-radius:20px;font-weight:700}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.card{padding:16px;background:#edf5f7;border-radius:9px}.card b{display:block;font-size:28px;color:#145568}.muted,small{color:#5a6b7f}small{display:block;font-size:12px;margin-top:6px}input{box-sizing:border-box;width:100%;padding:12px;border:1px solid #acbbcd;border-radius:6px;margin:10px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:12px;vertical-align:top;border-bottom:1px solid #e3e9f0;overflow-wrap:anywhere}th{background:#edf2f7;position:sticky;top:0}td:first-child{min-width:95px}.scroll{overflow:auto;max-height:680px}details{margin-top:18px}summary{cursor:pointer;font-weight:600}.notice{border-left:4px solid #dbab43;padding-left:16px}footer{color:#607086;font-size:12px}@media(max-width:700px){main{padding:12px}header,section{padding:16px}h1{font-size:28px}}</style></head><body><main>
"""
        + f"""<header><span class="status">{esc(report["status"].upper())}</span><h1>Full-corpus superstate graph</h1>
<p class="muted">Snapshot {esc(report["generated_at_utc"])} · {esc(report["model"]["name"])}</p>
<p class="notice">{esc(report["status_meaning"])}. Pending or unverified: {esc(pending)}.</p>
<div class="cards">{"".join(f'<div class="card">{esc(label)}<b>{esc(value)}</b></div>' for label, value in cards)}</div></header>
<section><h2>Optimization and frozen evaluation</h2><p>Observed proposals: <b>{esc(optimization["proposals_observed"])}</b>.
Accepted revisions: <b>{esc(optimization["accepted_revisions"])}</b>. Initial Pareto mean: <b>{esc(optimization["initial_pareto_score"])}</b>.
Best Pareto mean: <b>{esc(optimization["best_pareto_score"])}</b>. Frozen test mean: <b>{esc(report["frozen_test"]["mean_score"])}</b>.</p>
<p>Assignment records: {esc(report["assignments"]["recorded"])}; missing history IDs: {esc(report["assignments"]["missing_ids"])}; explicitly unassigned: {esc(report["assignments"]["unassigned"])}.
Available graph: {esc(graph["artifact"])}.</p><p>Observed transitions: {esc(graph["observed_transitions"])}; unassigned transitions: {esc(graph["unassigned_transitions"])}.</p></section>
<section><h2>Superstates</h2><p>Every available definition is shown. Reward mean and population variance use one outcome per visiting trajectory.</p>
<input type="search" data-table="states" aria-label="Search states" placeholder="Search state ID, name, or definition"><p id="states-count" class="muted"></p>
<div class="scroll"><table id="states"><thead><tr><th>State</th><th>Definition</th><th>Histories</th><th>Rollouts</th><th>Tasks</th><th>Mean reward</th><th>Variance</th></tr></thead><tbody>{"".join(state_rows)}</tbody></table></div></section>
<section><h2>Directed edges</h2><p>Observed witnesses and sampled applicability are separate evidence. No edge is universally certified.</p>
<input type="search" data-table="edges" aria-label="Search edges" placeholder="Search endpoints, operation, effect, or evidence status"><p id="edges-count" class="muted"></p>
<div class="scroll"><table id="edges"><thead><tr><th>Edge</th><th>Operation contract</th><th>Witnesses</th><th>Evidence</th><th>Witness reference subset</th></tr></thead><tbody>{"".join(edge_rows)}</tbody></table></div></section>
<section><h2>Independent checks</h2><p>Planned: {esc(report["independent_audits"]["planned"])}; completed: {esc(report["independent_audits"]["completed"])}; failed requests: {esc(report["independent_audits"]["failed_requests"])}.</p><ul>{audits}</ul>
<h2>Selected task specifications</h2><table><thead><tr><th>Draft</th><th>Feasibility review</th><th>Executed</th><th>Benchmark validated</th></tr></thead><tbody>{draft_rows}</tbody></table>
<p>Execution count: {esc(report["task_drafts"]["executed_task_count"])}. Benchmark validation count: {esc(report["task_drafts"]["benchmark_validated_count"])}.</p></section>
<section><h2>Interpretation and limits</h2><ul>{"".join("<li>" + esc(text) + "</li>" for text in report["limitations"])}</ul>
<details><summary>Snapshot warnings</summary><ul>{"".join("<li>" + esc(text) + "</li>" for text in report["warnings"]) or "<li>None.</li>"}</ul></details></section>
<footer>Self-contained report. No raw histories, task transcripts, assignment evidence, or runtime credentials are embedded. Graph witness references are explicitly limited to three per edge; the full local graph is unchanged.</footer>
"""
        + """</main><script>for(const input of document.querySelectorAll('input[data-table]')){const id=input.dataset.table;const rows=[...document.querySelectorAll('#'+id+' tbody tr')];const filter=()=>{const q=input.value.toLocaleLowerCase();let shown=0;for(const row of rows){const visible=row.textContent.toLocaleLowerCase().includes(q);row.hidden=!visible;if(visible)shown++;}document.getElementById(id+'-count').textContent=shown+' of '+rows.length+' rows shown';};input.addEventListener('input',filter);filter();}</script></body></html>"""
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
