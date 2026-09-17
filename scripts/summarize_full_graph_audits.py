#!/usr/bin/env python3
"""Describe final saved audit verdicts and citation validation without rejudging them."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


KINDS = ("within_state_coherence", "boundary_and_redundancy", "edge_source_applicability")
VERDICTS = {"supported", "contradicted", "unknown"}
SOURCE_FILES = (
    "independent_audit_jobs.json",
    "independent_audit_results.json",
    "independent_audit_summary.json",
    "graph.json",
    "assignments_all.json",
)
ERROR_CATEGORIES = {
    "Evidence is not a list": "evidence_not_list",
    "Evidence citation is not a mapping": "citation_not_mapping",
    "Missing or out-of-scope citation": "missing_or_out_of_scope_citation",
    "Definitive verdict has no grounded citations": "definitive_verdict_without_grounded_citations",
    "Definitive verdict must cite each tested source history": "missing_tested_source_coverage",
}
LIMITATIONS = [
    "Operational completion, exact citation validity, and semantic reliability are separate properties.",
    "Request-failure records are the runner's error category and may include response-processing errors; they are not a count of transport failures or model calls.",
    "A citation-valid quote need not entail the judgment; any invalid extra quote can also downgrade otherwise cited evidence.",
    "Unknown combines raw semantic uncertainty, citation-invalidated judgments, and request failures; the counts separate those recorded causes without resolving them.",
    "These are separately prompted LLM proxies, not gold labels, calibrated semantic accuracy, a random population sample, or execution certificates.",
    "All-source support refers only to planned sampled sources, never every history in a state or executable composed paths.",
    "No verdict, graph flag, formation decision, or evaluation score is changed by this diagnostic.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def error_category(message):
    if isinstance(message, str) and message.startswith("Quote is not an exact prefix substring: "):
        return "quote_not_exact_prefix_substring"
    return ERROR_CATEGORIES.get(message, "other_validator_error")


def summarize(run: Path):
    raw = {name: (run / name).read_bytes() for name in SOURCE_FILES}
    data = {name: json.loads(value) for name, value in raw.items()}
    jobs, results, aggregate, graph, assignments = (data[name] for name in SOURCE_FILES)
    job_index = {job["audit_id"]: job for job in jobs}
    result_index = {result["audit_id"]: result for result in results}
    require(
        len(job_index) == len(jobs)
        and len(result_index) == len(results)
        and set(job_index) == set(result_index),
        "Final audit census is incomplete or duplicated",
    )
    require(all(job["kind"] in KINDS for job in jobs), "Unexpected audit kind")
    # Reuse publication's membership/edge-eligibility consistency checks, never its labels as truth.
    spec = importlib.util.spec_from_file_location(
        "audit_diagnostic_export_checks", Path(__file__).with_name("export_full_graph_artifacts.py")
    )
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    eligibility = checks.sampled_edge_eligibility(graph, assignments, jobs, results, aggregate)
    groups, by_edge = defaultdict(list), defaultdict(list)
    for job in jobs:
        result = result_index[job["audit_id"]]
        require(result.get("kind") == job["kind"], "Result kind differs from audit plan")
        failed = result.get("execution_status") == "error"
        validated, original = result.get("verdict"), result.get("original_verdict")
        errors = result.get("citation_errors", [])
        if failed:
            require(
                validated == "unknown" and original is None,
                "Request failure cannot contain a normal raw verdict",
            )
            original = "unavailable_request_failure"
        else:
            require(
                result.get("schema_and_citations_checked") is True
                and original in VERDICTS
                and validated in VERDICTS
                and isinstance(errors, list)
                and all(isinstance(e, str) for e in errors),
                "Normal response is missing raw/verdict/citation-validation provenance",
            )
            require(
                validated == ("unknown" if errors else original),
                "Stored verdict does not follow the frozen citation-validation rule",
            )
        row = {
            "raw": original,
            "validated": validated,
            "request_failure": failed,
            "citation_categories": [error_category(e) for e in errors],
        }
        groups[job["kind"]].append(row)
        if job["kind"] == "edge_source_applicability":
            by_edge[job["edge_id"]].append(row)
    by_kind = {}
    for kind in KINDS:
        rows = groups[kind]
        categories = sorted({category for row in rows for category in row["citation_categories"]})
        contingency = Counter((row["raw"], row["validated"]) for row in rows)
        by_kind[kind] = {
            "planned": sum(job["kind"] == kind for job in jobs),
            "result_records": len(rows),
            "normal_response_records": sum(not row["request_failure"] for row in rows),
            "request_failure_records": sum(row["request_failure"] for row in rows),
            "validated_verdict_counts": dict(Counter(row["validated"] for row in rows)),
            "raw_to_validated": [
                {"raw": left, "validated": right, "count": count}
                for (left, right), count in sorted(contingency.items())
            ],
            "responses_with_citation_errors": sum(bool(row["citation_categories"]) for row in rows),
            "definitive_verdicts_downgraded_by_citation_errors": sum(
                row["raw"] in {"supported", "contradicted"}
                and row["validated"] == "unknown"
                and bool(row["citation_categories"])
                for row in rows
            ),
            "citation_error_categories": {
                category: {
                    "affected_results": sum(category in row["citation_categories"] for row in rows),
                    "error_occurrences": sum(
                        row["citation_categories"].count(category) for row in rows
                    ),
                }
                for category in categories
            },
        }
        require(
            by_kind[kind]["validated_verdict_counts"] == aggregate["by_kind"].get(kind, {}),
            "Recomputed diagnostic counts differ from saved aggregate",
        )
    edges = []
    for edge in graph["edges"]:
        rows = by_edge[edge["id"]]
        edges.append(
            {
                "edge_id": edge["id"],
                "planned_source_checks": len(rows),
                "all_validated_source_checks_supported": bool(rows)
                and all(r["validated"] == "supported" for r in rows),
                "all_raw_source_checks_supported": bool(rows)
                and all(r["raw"] == "supported" for r in rows),
                "has_validated_contradiction": any(r["validated"] == "contradicted" for r in rows),
                "has_raw_contradiction": any(r["raw"] == "contradicted" for r in rows),
                "has_unknown_validated_verdict": any(r["validated"] == "unknown" for r in rows),
                "has_request_failure": any(r["request_failure"] for r in rows),
                "has_citation_errors": any(r["citation_categories"] for r in rows),
                "proposer_plausible": edge["proposal"]["universal_source_plausible"] is True,
                "proposer_rejected": edge["proposal"]["universal_source_plausible"] is False,
                "sampled_supported_eligible": edge["id"] in eligibility["eligible_ids"],
            }
        )
    flags = (
        {
            key: sum(row[key] for row in edges)
            for key in edges[0]
            if key not in {"edge_id", "planned_source_checks"}
        }
        if edges
        else {}
    )
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    require(
        all(sha(run / name) == expected for name, expected in hashes.items()),
        "Audit artifacts changed while collecting diagnostics",
    )
    return {
        "format": "final-superstate-audit-diagnostics-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_audit_ledger_verified": True,
        "planned_checks": len(jobs),
        "result_records": len(results),
        "request_failure_records": sum(r.get("execution_status") == "error" for r in results),
        "by_kind": by_kind,
        "observed_edge_count": len(edges),
        "edge_flag_counts_nonexclusive": flags,
        "edges": edges,
        "source_artifact_sha256": hashes,
        "diagnostic_script_sha256": sha(__file__),
        "limitations": LIMITATIONS,
    }


def markdown(report):
    lines = [
        "# Final full-corpus independent audit diagnostics",
        "",
        f"Generated {report['created_utc']} from the completed saved audit ledger. "
        "This is a read-only accounting of frozen outputs; no judgment was rerun or relabeled.",
        "",
        f"**{report['result_records']} / {report['planned_checks']} planned checks** have result records; "
        f"**{report['request_failure_records']}** are request-failure records. "
        "A result record or valid citation is not evidence of semantic correctness.",
        "",
        "| Kind | Planned / recorded | Request failures | Supported | Contradicted | Unknown | Citation-error responses | Definitive verdicts downgraded |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, row in report["by_kind"].items():
        v = row["validated_verdict_counts"]
        lines.append(
            f"| {kind} | {row['planned']} / {row['result_records']} | {row['request_failure_records']} | "
            f"{v.get('supported', 0)} | {v.get('contradicted', 0)} | {v.get('unknown', 0)} | "
            f"{row['responses_with_citation_errors']} | {row['definitive_verdicts_downgraded_by_citation_errors']} |"
        )
    lines += [
        "",
        "## Raw → citation-validated verdicts",
        "",
        "The table includes only nonzero cells. `unavailable_request_failure` denotes a "
        "failed request with no normal raw judgment, rather than a raw unknown judgment.",
        "",
        "| Kind | Raw | Citation-validated | Count |",
        "|---|---|---|---:|",
    ]
    for kind, row in report["by_kind"].items():
        for cell in row["raw_to_validated"]:
            lines.append(f"| {kind} | {cell['raw']} | {cell['validated']} | {cell['count']} |")
    lines += [
        "",
        "## Citation diagnostics",
        "",
        "Affected-result counts can overlap across categories. Error occurrences also count "
        "repeated failures within a response. These counts describe formatting and source "
        "coverage checks, not semantic accuracy.",
        "",
        "| Kind | Category | Affected results | Error occurrences |",
        "|---|---|---:|---:|",
    ]
    for kind, row in report["by_kind"].items():
        for category, counts in row["citation_error_categories"].items():
            lines.append(
                f"| {kind} | {category} | {counts['affected_results']} | {counts['error_occurrences']} |"
            )
    lines += [
        "",
        "## Edge-level outcomes",
        "",
        f"The graph has {report['observed_edge_count']} observed edge groups. "
        "The following flags overlap. All-supported means all *sampled* source checks; "
        "eligibility additionally requires proposer plausibility.",
        "",
        "| Flag | Edge groups |",
        "|---|---:|",
    ]
    lines += [
        f"| {key} | {value} |" for key, value in report["edge_flag_counts_nonexclusive"].items()
    ]
    lines += ["", "## Interpretation and reproduction", ""]
    lines += ["- " + limitation for limitation in report["limitations"]]
    lines += [
        "",
        "The [ten-response qualitative audit review](INDEPENDENT_AUDIT_REVIEW.md) explains "
        "examples of citation failures and semantic objections without replacing labels. "
        "The [census assignment review](CENSUS_QUALITATIVE_REVIEW.md) and "
        "[frozen graph review](FROZEN_GRAPH_REVIEW.md) examine other aspects of formation. "
        "Those purposive samples do not establish population error rates.",
        "",
        "```sh",
        ".venv/bin/python scripts/summarize_full_graph_audits.py \\",
        "  --run results/full_graph/run_v1 \\",
        "  --output reports/full-corpus-2026-09-17/audit_diagnostics.json \\",
        "  --markdown docs/FULL_CORPUS_AUDIT_RESULTS.md",
        "```",
        "",
        "The JSON receipt also contains every edge's diagnostic flags. Both outputs contain "
        "counts, IDs, and hashes; source prefixes, raw findings, and error messages are omitted.",
        "",
        "| Source | SHA-256 |",
        "|---|---|",
    ]
    lines += [
        f"| `{name}` | `{value}` |" for name, value in report["source_artifact_sha256"].items()
    ]
    lines += [f"| Diagnostic script | `{report['diagnostic_script_sha256']}` |", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    for destination in (args.output, args.markdown):
        require(
            destination is None or args.run.resolve() not in destination.resolve().parents,
            "Diagnostics must be written outside the original run",
        )
    report = summarize(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report))
    print(
        json.dumps(
            {
                "planned": report["planned_checks"],
                "recorded": report["result_records"],
                "request_failures": report["request_failure_records"],
                "eligible_edges": report["edge_flag_counts_nonexclusive"].get(
                    "sampled_supported_eligible", 0
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
