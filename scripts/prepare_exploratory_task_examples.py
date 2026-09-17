#!/usr/bin/env python3
"""Select, or explicitly execute, a separate witness-specific task experiment.

Default mode is read-only and never opens a model runtime. Execution requires a
completed original run and writes to a separate directory. Unknown applicability
is permitted here; it never becomes primary sampled-supported eligibility.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from superstate_graphs.full_corpus import render_history
from superstate_graphs.graph_analysis import _edge_view, _node_view, select_grounded_paths
from superstate_graphs.graph_task_examples import construct_executable_example
from superstate_graphs import graph_task_examples


LABEL = "witness_specific_exploratory_not_sampled_supported"
SOURCE_FILES = (
    "completion.json",
    "graph.json",
    "assignments_all.json",
    "task_path_proposals.json",
    "task_draft_summary.json",
    "executable_task_summary.json",
    "independent_audit_jobs.json",
    "independent_audit_results.json",
    "independent_audit_summary.json",
)
ARTIFACT_NAMES = {
    "source_path.json",
    "generator_response.json",
    "task.json",
    "independent_solution_review.json",
    "local_validation.json",
    "instruction.md",
    "oracle.sql",
    "independent.sql",
    "fixture.duckdb",
    "expected_result.json",
    "README.md",
}
LIMITATIONS = [
    "Witness-specific exploratory tasks, not sampled-supported graph paths.",
    "Unknown applicability, including citation-invalidated judgments, remains unresolved; raw and validated verdicts are retained separately.",
    "No validated or raw proposed counterexample is admitted. Particular observed witnesses may support a concrete task even when reusable source applicability is unestablished.",
    "A successful receipt means a strict independent faithfulness review and two separately prompted SQL solutions executed and agreed on a synthetic fixture.",
    "It does not prove reusable or universal edges, execution of arbitrary graph paths, learner success, or post-training lift.",
    "Original graph, audit, completion, and executable-example results remain unchanged.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def export_validator():
    spec = importlib.util.spec_from_file_location(
        "exploratory_export_checks", Path(__file__).with_name("export_full_graph_artifacts.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_paths(run: Path, corpus: Path, *, max_paths=24, resample_filtered_paths=False) -> dict:
    require(
        type(max_paths) is int and 1 <= max_paths <= 24,
        "Path attempt limit must be between 1 and 24",
    )
    completion_path = run / "completion.json"
    require(completion_path.is_file(), "Original run has not completed")
    completion = json.loads(completion_path.read_text())
    require(completion.get("status") == "complete", "Original run is not complete")
    raw_sources = {name: (run / name).read_bytes() for name in SOURCE_FILES}
    sources = {name: json.loads(raw) for name, raw in raw_sources.items()}
    hashes = {"run/" + name: hashlib.sha256(raw).hexdigest() for name, raw in raw_sources.items()}
    primary = sources["executable_task_summary.json"]
    require(
        primary.get("status") in {"target_reached", "supported_paths_exhausted"},
        "Original executable task stage has not finalized",
    )
    manifest = json.loads((corpus / "manifest.json").read_text())
    hashes["corpus/manifest.json"] = fingerprint(corpus / "manifest.json")
    for name in ("histories.jsonl", "transitions.jsonl", "rollouts.jsonl"):
        hashes["corpus/" + name] = fingerprint(corpus / name)
        require(
            hashes["corpus/" + name] == manifest.get("artifact_sha256", {}).get(name),
            "Immutable corpus artifact hash mismatch",
        )
    histories = {
        row["history_id"]: row
        for row in (
            json.loads(line) for line in (corpus / "histories.jsonl").read_text().splitlines()
        )
    }
    transitions = {
        row["transition_id"]: row
        for row in (
            json.loads(line) for line in (corpus / "transitions.jsonl").read_text().splitlines()
        )
    }
    graph, assignments = sources["graph.json"], sources["assignments_all.json"]
    require(
        set(assignments) == set(histories)
        and all(row.get("state_id") is not None for row in assignments.values()),
        "Original full assignment census is incomplete",
    )
    jobs, results = (
        sources["independent_audit_jobs.json"],
        sources["independent_audit_results.json"],
    )
    # Validate primary rules and flags; this function never changes them.
    export_validator().sampled_edge_eligibility(
        graph, assignments, jobs, results, sources["independent_audit_summary.json"]
    )
    results_by_id = {row["audit_id"]: row for row in results}
    jobs_by_edge = defaultdict(list)
    for job in jobs:
        if job["kind"] == "edge_source_applicability":
            jobs_by_edge[job["edge_id"]].append(job)
    edges, nodes = {e["id"]: e for e in graph["edges"]}, {n["id"]: n for n in graph["nodes"]}
    require(
        len(edges) == len(graph["edges"]) and len(nodes) == len(graph["nodes"]),
        "Duplicate graph identifiers",
    )
    allowed, edge_evidence, rejected = set(), {}, Counter()
    for eid, edge in edges.items():
        if edge.get("proposal", {}).get("universal_source_plausible") is not True:
            rejected["proposer_not_plausible"] += 1
            continue
        planned = jobs_by_edge[eid]
        if not planned:
            rejected["missing_audit_evidence"] += 1
            continue
        evidence = []
        disallowed = None
        for job in planned:
            result = results_by_id.get(job["audit_id"])
            if (
                result is None
                or result.get("execution_status") == "error"
                or result.get("schema_and_citations_checked") is not True
            ):
                disallowed = "missing_or_failed_audit_evidence"
                break
            verdict, original = result.get("verdict"), result.get("original_verdict")
            if verdict == "contradicted" or original == "contradicted":
                disallowed = "validated_or_raw_counterexample"
                break
            if verdict not in {"supported", "unknown"} or original not in {"supported", "unknown"}:
                disallowed = "missing_verdict_provenance"
                break
            errors = result.get("citation_errors")
            if not isinstance(errors, list):
                disallowed = "missing_citation_validation_receipt"
                break
            evidence.append(
                {
                    "audit_id": job["audit_id"],
                    "validated_verdict": verdict,
                    "original_verdict": original,
                    "citation_error_count": len(errors),
                    "applicability_label": "unresolved_applicability_due_to_citation_errors"
                    if verdict == "unknown" and errors
                    else "unresolved_applicability"
                    if verdict == "unknown"
                    else "sampled_source_supported",
                }
            )
        if disallowed:
            rejected[disallowed] += 1
            continue
        allowed.add(eid)
        edge_evidence[eid] = evidence
    reviewed = {
        row["path_id"]: row.get("review_verdict")
        for row in sources["task_draft_summary.json"].get("attempts", [])
    }
    rejected_path_ids = {
        row["path_id"]
        for row in sources["task_draft_summary.json"].get("attempts", [])
        if row.get("review_verdict") == "contradicted"
    }
    attempted = {row["path_id"] for row in primary.get("attempts", [])}
    attempted_signatures = {
        tuple(step["transition_id"] for step in path["transitions"])
        for path in sources["task_path_proposals.json"]
        if path["path_id"] in attempted
    }
    rejected_signatures = {
        tuple(step["transition_id"] for step in path["transitions"])
        for path in sources["task_path_proposals.json"]
        if path["path_id"] in rejected_path_ids
    }
    candidates = [(path, "saved_path", None) for path in sources["task_path_proposals.json"]]
    if resample_filtered_paths:
        filtered = {
            **graph,
            "edges": [
                {**edge, "traversable": True} for edge in graph["edges"] if edge["id"] in allowed
            ],
        }
        candidates.extend(
            (path, "resampled_filtered_observed_graph", 20260919)
            for path in select_grounded_paths(
                filtered, histories.values(), assignments, count=24, lengths=(2,), seed=20260919
            )
        )
    selected, excluded_paths, path_ids, signatures = [], Counter(), set(), set()
    for path, origin, resampling_seed in candidates:
        pid = path.get("path_id")
        require(
            isinstance(pid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,120}", pid) is not None,
            "Invalid or duplicate saved path identifier",
        )
        if pid in path_ids:
            require(origin != "saved_path", "Duplicate saved path identifier")
            continue
        path_ids.add(pid)
        if pid in attempted:
            excluded_paths["already_attempted_in_primary_stage"] += 1
            continue
        if pid in rejected_path_ids:
            excluded_paths["draft_review_contradicted"] += 1
            continue
        steps = path.get("transitions", [])
        signature = tuple(step.get("transition_id") for step in steps)
        if signature in attempted_signatures:
            excluded_paths["edge_sequence_already_attempted_in_primary_stage"] += 1
            continue
        if signature in rejected_signatures:
            excluded_paths["edge_sequence_rejected_by_primary_draft_review"] += 1
            continue
        if (
            len(steps) != 2
            or len(set(signature)) != 2
            or any(eid not in allowed for eid in signature)
        ):
            excluded_paths["ineligible_edge"] += 1
            continue
        if signature in signatures:
            continue
        state_ids, clean_steps, witness_ok = [], [], True
        for step in steps:
            eid = step["transition_id"]
            edge = edges[eid]
            source, target = step.get("source_history_id"), step.get("target_history_id")
            if not state_ids:
                state_ids.append(edge["source"])
            witness = next(
                (
                    w
                    for w in edge.get("witnesses", [])
                    if w.get("source_id", w.get("source_history_id")) == source
                    and w.get("target_id", w.get("target_history_id")) == target
                ),
                None,
            )
            recorded = transitions.get(witness.get("transition_id")) if witness else None
            if source not in histories or target not in histories or recorded is None:
                witness_ok = False
                break
            left, right = histories[source], histories[target]
            if (
                edge["source"] != state_ids[-1]
                or assignments[source]["state_id"] != edge["source"]
                or assignments[target]["state_id"] != edge["target"]
                or recorded["source_history_id"] != source
                or recorded["target_history_id"] != target
                or right["step"] != left["step"] + 1
                or left["rollout_id"] != right["rollout_id"]
                or left["task_id"] != right["task_id"]
                or witness["rollout_id"] != left["rollout_id"]
                or witness["task_id"] != left["task_id"]
                or any(step.get(key) != value for key, value in _edge_view(edge).items())
            ):
                witness_ok = False
                break
            state_ids.append(edge["target"])
            clean_steps.append(
                {
                    **_edge_view(edge),
                    "transition_id": eid,
                    "source_history_id": source,
                    "target_history_id": target,
                    "source_task_id": left["task_id"],
                    "rollout_id": left["rollout_id"],
                }
            )
        if (
            not witness_ok
            or path.get("state_ids") != state_ids
            or path.get("nodes") != [_node_view(nodes[sid]) for sid in state_ids]
            or len({step["source_task_id"] for step in clean_steps}) < 2
        ):
            excluded_paths["missing_or_inconsistent_witness_evidence"] += 1
            continue
        signatures.add(signature)
        clean = {
            "path_id": pid,
            "state_ids": state_ids,
            "nodes": [_node_view(nodes[sid]) for sid in state_ids],
            "transitions": clean_steps,
            "source_task_ids": sorted({s["source_task_id"] for s in clean_steps}),
            "cross_task": True,
            "selection_evidence": LABEL,
            "individual_edges_observed": True,
            "composed_path_executed": False,
            "universal_contract_certified": False,
        }
        selected.append(
            {
                "path": clean,
                "selection_origin": origin,
                "resampling_seed": resampling_seed,
                "draft_review": reviewed.get(pid),
                "edge_audit_evidence": {
                    step["transition_id"]: edge_evidence[step["transition_id"]] for step in steps
                },
            }
        )
    selected.sort(
        key=lambda item: (
            item["selection_origin"] != "saved_path",
            {"plausible": 0, "unresolved": 1}.get(item["draft_review"], 2),
            item["path"]["path_id"],
        )
    )
    selected = selected[:max_paths]
    plan = {
        "format": "exploratory-witness-task-plan-v1",
        "evidence_label": LABEL,
        "original_run_completed": True,
        "original_supported_edges": sum(e.get("traversable") is True for e in graph["edges"]),
        "original_executed_examples": primary.get("locally_executed_consistent_count"),
        "qualifying_exploratory_edges": len(allowed),
        "maximum_path_attempts": max_paths,
        "resample_filtered_paths_requested": resample_filtered_paths,
        "resampling_seed": 20260919 if resample_filtered_paths else None,
        "selected_paths": selected,
        "excluded_edge_reason_counts": dict(rejected),
        "excluded_path_reason_counts": dict(excluded_paths),
        "source_artifact_sha256": hashes,
        "limitations": LIMITATIONS,
    }
    plan["plan_sha256"] = digest(plan)
    verify_snapshot(plan, run, corpus)
    return plan


def public_plan(plan: dict) -> dict:
    return {
        **{k: v for k, v in plan.items() if k != "selected_paths"},
        "selected_paths": [
            {
                "path_id": item["path"]["path_id"],
                "edge_ids": [step["transition_id"] for step in item["path"]["transitions"]],
                "draft_review": item["draft_review"],
                "selection_origin": item["selection_origin"],
                "resampling_seed": item["resampling_seed"],
                "edge_audit_evidence": item["edge_audit_evidence"],
            }
            for item in plan["selected_paths"]
        ],
    }


def prefix_provider(corpus: Path):
    runs = {
        row["id"]: row
        for row in (
            json.loads(line) for line in (corpus / "rollouts.jsonl").read_text().splitlines()
        )
    }

    def prefix(hid):
        rid, step = hid.rsplit(":h", 1)
        return render_history(runs[rid], int(step))

    return prefix


def verify_snapshot(plan: dict, run: Path, corpus: Path):
    for key, expected in plan["source_artifact_sha256"].items():
        kind, name = key.split("/", 1)
        require(
            fingerprint((run if kind == "run" else corpus) / name) == expected,
            "Original source artifacts changed after exploratory selection",
        )


def safe_attempt(result: dict, directory: Path) -> dict:
    validation = result.get("local_validation", {})
    success = result.get("status") == "locally_executed_consistent"
    hashes = {k: v for k, v in result.get("artifact_sha256", {}).items() if k in ARTIFACT_NAMES}
    if success:
        require(
            result.get("review_verdict") == "faithful"
            and all(
                validation.get(k) is True
                for k in ("reference_query_executed", "independent_query_executed", "queries_agree")
            ),
            "Successful task lacks faithful-review and execution receipts",
        )
        require(set(hashes) == ARTIFACT_NAMES, "Successful task is missing required artifacts")
    require(
        all(
            (directory / name).is_file() and fingerprint(directory / name) == value
            for name, value in hashes.items()
        ),
        "Task artifact hashes do not match their receipt",
    )
    return {
        "path_id": result["path_id"],
        "status": result["status"],
        "evidence_label": LABEL,
        "independent_review_verdict": result.get("review_verdict"),
        "locally_executed_consistent": success,
        "reference_query_executed": validation.get("reference_query_executed") is True,
        "independent_query_executed": validation.get("independent_query_executed") is True,
        "queries_agree": validation.get("queries_agree") is True,
        "result_sha256": fingerprint(directory / "result.json"),
        "artifact_sha256": hashes,
        "learner_evaluated": False,
        "universal_contract_certified": False,
    }


async def execute_plan(
    plan: dict, provider, llm, output: Path, *, target=3, source_guard=None
) -> dict:
    require(type(target) is int and 1 <= target <= 3, "Example target must be between 1 and 3")
    model = {
        key: value
        for key, value in llm.runtime.items()
        if key in {"model", "revision", "model_revision", "tokenizer_revision", "quantization"}
    }
    identity = {
        "plan_sha256": plan["plan_sha256"],
        "target": target,
        "public_model": model,
        "script_sha256": fingerprint(Path(__file__)),
        "constructor_sha256": fingerprint(Path(graph_task_examples.__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "exploratory_task_summary.json"
    summary = {
        "format": "exploratory-witness-task-experiment-v1",
        "identity": identity,
        "evidence_label": LABEL,
        "status": "running",
        "requested_examples": target,
        "maximum_path_attempts": plan["maximum_path_attempts"],
        "attempts": [],
        "selected_examples": [],
        "locally_executed_consistent_count": 0,
        "target_reached": False,
        "limitations": LIMITATIONS,
    }
    if checkpoint.exists():
        summary = json.loads(checkpoint.read_text())
        require(
            summary.get("identity") == identity,
            "Supplementary checkpoint identity changed; use a new output directory",
        )
    selected_ids = [item["path"]["path_id"] for item in plan["selected_paths"]]
    require(
        [row["path_id"] for row in summary["attempts"]] == selected_ids[: len(summary["attempts"])],
        "Supplementary attempt ledger is inconsistent with frozen plan",
    )
    for row in summary["attempts"]:
        directory = output / "examples" / row["path_id"]
        require(
            set(row["artifact_sha256"]) <= ARTIFACT_NAMES
            and fingerprint(directory / "result.json") == row["result_sha256"]
            and all(
                fingerprint(directory / name) == value
                for name, value in row["artifact_sha256"].items()
            ),
            "Previously attempted task artifacts changed",
        )
        require(
            safe_attempt(json.loads((directory / "result.json").read_text()), directory) == row,
            "Supplementary checkpoint result differs from its original receipt",
        )
    successes = [row for row in summary["attempts"] if row["locally_executed_consistent"]]
    require(
        summary["selected_examples"] == successes
        and summary["locally_executed_consistent_count"] == len(successes),
        "Supplementary success ledger is inconsistent",
    )
    if source_guard:
        source_guard()
    write_json(output / "selection.json", public_plan(plan))
    write_json(checkpoint, summary)
    for item in plan["selected_paths"][len(summary["attempts"]) :]:
        if sum(row["locally_executed_consistent"] for row in summary["attempts"]) >= target:
            break
        path = item["path"]
        directory = output / "examples" / path["path_id"]
        result = await construct_executable_example(
            path, provider, llm, directory, timeout_seconds=30, thinking=True
        )
        require(result.get("path_id") == path["path_id"], "Constructor returned a different path")
        if source_guard:
            source_guard()
        summary["attempts"].append(safe_attempt(result, directory))
        summary["selected_examples"] = [
            row for row in summary["attempts"] if row["locally_executed_consistent"]
        ]
        summary["locally_executed_consistent_count"] = len(summary["selected_examples"])
        summary["attempt_status_counts"] = dict(
            Counter(row["status"] for row in summary["attempts"])
        )
        write_json(checkpoint, summary)
    summary["target_reached"] = summary["locally_executed_consistent_count"] >= target
    summary["status"] = (
        "target_reached" if summary["target_reached"] else "exploratory_paths_exhausted"
    )
    write_json(checkpoint, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly activate supplementary inference after original completion",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        action="append",
        help="Private model runtime configuration; read only with --execute",
    )
    parser.add_argument(
        "--output", type=Path, help="Separate experiment directory, outside original run/corpus"
    )
    parser.add_argument("--target", type=int, default=3)
    parser.add_argument("--max-paths", type=int, default=24)
    parser.add_argument(
        "--resample-filtered-paths",
        action="store_true",
        help="Fill remaining candidates with deterministic two-edge cross-task paths, seed 20260919; in-memory copy only",
    )
    args = parser.parse_args()
    try:
        plan = select_paths(
            args.run,
            args.corpus,
            max_paths=args.max_paths,
            resample_filtered_paths=args.resample_filtered_paths,
        )
        if not args.execute:
            print(json.dumps(public_plan(plan), indent=2))
            return
        require(
            args.runtime and args.output,
            "Execution requires explicit runtime and separate output directory",
        )
        output, run, corpus = args.output.resolve(), args.run.resolve(), args.corpus.resolve()
        require(
            all(
                output != source and source not in output.parents and output not in source.parents
                for source in (run, corpus)
            ),
            "Supplementary output must be separate from original run and corpus",
        )
        from superstate_graphs.graph_llm import GraphLLMPool

        async def activate():
            verify_snapshot(plan, run, corpus)
            async with GraphLLMPool(
                args.runtime, cache_dir=output / "llm_cache", concurrency=2
            ) as llm:
                try:
                    return await execute_plan(
                        plan,
                        prefix_provider(corpus),
                        llm,
                        output,
                        target=args.target,
                        source_guard=lambda: verify_snapshot(plan, run, corpus),
                    )
                finally:
                    write_json(output / "llm_usage.json", llm.stats)

        summary = asyncio.run(activate())
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "status",
                        "evidence_label",
                        "locally_executed_consistent_count",
                        "target_reached",
                    )
                }
            )
        )
    except Exception as error:
        # Runtime errors can contain connection details. Never print their text.
        parser.exit(
            1,
            f"Supplementary experiment stopped ({type(error).__name__}); no primary artifacts changed.\n",
        )


if __name__ == "__main__":
    main()
