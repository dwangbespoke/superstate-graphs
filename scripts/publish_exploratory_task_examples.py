#!/usr/bin/env python3
"""Publish finalized witness-specific exploratory examples to a separate namespace.

Read-only verification of original and supplementary sources; no model calls.
Never adds examples to the primary graph export or its supported-path counts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from superstate_graphs import graph_task_examples as task_code


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment_code = load_script("prepare_exploratory_task_examples")
export_checks = experiment_code.export_validator()
require = experiment_code.require
fingerprint = experiment_code.fingerprint
LABEL = experiment_code.LABEL
FORMAT = "published-exploratory-witness-tasks-v1"
COPY_FILES = (
    "instruction.md",
    "fixture.duckdb",
    "oracle.sql",
    "independent.sql",
    "expected_result.json",
)
STATUSES = {
    "locally_executed_consistent",
    "unsupported",
    "independent_review_not_supported",
    "consistency_failed",
    "construction_error",
    "execution_failed",
    "execution_timeout",
}
PRIVACY_LIMITATION = (
    "Allowlisted publication and automated credential/transcript-marker scans are not a semantic "
    "privacy guarantee. Inspect generated instructions, titles, SQL literals, expected values, and "
    "synthetic fixture cells for copied private/source material before public distribution."
)


def read_json(path):
    return json.loads(path.read_bytes())


def encode(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode()
        + b"\n"
    )


def load_manual_reviews(path: Path | None, successful: list[dict]):
    """An optional review annotates fixed outcomes; it never changes their counts."""
    if path is None:
        return {}, None
    raw = path.read_bytes()
    export_checks.assert_publishable(raw, "manual review receipt")
    value = json.loads(raw)
    require(
        isinstance(value, dict)
        and set(value) == {"format", "reviews"}
        and value["format"] == "exploratory-task-manual-review-v1"
        and isinstance(value["reviews"], list),
        "Invalid manual review format",
    )
    by_path = {row["path_id"]: row for row in successful}
    reviews = {}
    fields = {
        "path_id",
        "verdict",
        "findings",
        "review_method",
        "reviewer_kind",
        "reviewed_at_utc",
        "reviewed_artifact_sha256",
    }
    for row in value["reviews"]:
        require(isinstance(row, dict) and set(row) == fields, "Unexpected manual review fields")
        pid = row["path_id"]
        require(
            isinstance(pid, str) and pid in by_path and pid not in reviews,
            "Manual review path is duplicate or outside exported examples",
        )
        require(
            row["verdict"] in {"failed", "no_defect_identified", "unresolved"}
            and row["reviewer_kind"] in {"coding_agent", "human"},
            "Invalid manual semantic verdict or reviewer kind",
        )
        require(
            isinstance(row["findings"], list)
            and 1 <= len(row["findings"]) <= 30
            and all(
                isinstance(text, str) and 0 < len(text.strip()) <= 5000 for text in row["findings"]
            )
            and isinstance(row["review_method"], str)
            and 0 < len(row["review_method"].strip()) <= 2000,
            "Manual review requires bounded finding and method summaries",
        )
        stamp = datetime.fromisoformat(row["reviewed_at_utc"])
        require(
            stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0,
            "Manual review timestamp must be UTC",
        )
        require(
            row["reviewed_artifact_sha256"]
            == {name: by_path[pid]["artifact_sha256"][name] for name in COPY_FILES},
            "Manual review does not bind the current five task artifacts",
        )
        reviews[pid] = row
    return reviews, hashlib.sha256(raw).hexdigest()


def public_manual_review(pid: str, reviews: dict) -> dict:
    row = reviews.get(pid)
    return {
        "format": "published-exploratory-task-manual-review-v1",
        "path_id": pid,
        "evidence_label": LABEL,
        "review_record_supplied": row is not None,
        "semantic_verdict": row["verdict"] if row else "not_reviewed",
        "findings": row["findings"] if row else [],
        **(
            {
                key: row[key]
                for key in (
                    "review_method",
                    "reviewer_kind",
                    "reviewed_at_utc",
                    "reviewed_artifact_sha256",
                )
            }
            if row
            else {}
        ),
        "learner_readiness_certified": False,
        "original_query_agreement_outcome_preserved": True,
        "interpretation": "Manual review is scoped additional evidence. Local SQL agreement, an LLM faithfulness verdict, and no_defect_identified do not establish semantic validity or benchmark readiness.",
    }


def manual_review_warning(review: dict) -> str:
    verdict = review["semantic_verdict"]
    if verdict == "failed":
        return (
            "**FAILED RESEARCH CASE — NOT LEARNER-READY.** Manual review identified a semantic defect. "
            "The preserved queries and expected output may encode an incorrect or underspecified task; "
            "do not use this case as a validated learner benchmark. Local query agreement remains the original recorded outcome."
        )
    if verdict == "not_reviewed":
        return "**NOT MANUALLY REVIEWED — SEMANTIC VALIDITY UNESTABLISHED.** Local query agreement does not establish that the instruction or oracle is correct."
    if verdict == "unresolved":
        return "**MANUAL SEMANTIC REVIEW UNRESOLVED — NOT CERTIFIED LEARNER-READY.** Do not interpret local query agreement as semantic validation."
    return "**NO DEFECT IDENTIFIED IN THE RECORDED REVIEW SCOPE.** This is not benchmark validation or a learner-readiness certificate; inspect the review's methods and limits."


def verify_experiment(experiment: Path, run: Path, corpus: Path):
    selection = read_json(experiment / "selection.json")
    summary = read_json(experiment / "exploratory_task_summary.json")
    require(
        summary.get("status") in {"target_reached", "exploratory_paths_exhausted"},
        "Supplementary experiment has not finalized",
    )
    require(
        selection.get("evidence_label") == summary.get("evidence_label") == LABEL,
        "Supplementary evidence label changed",
    )
    plan = experiment_code.select_paths(
        run,
        corpus,
        max_paths=selection["maximum_path_attempts"],
        resample_filtered_paths=selection["resample_filtered_paths_requested"],
    )
    require(
        experiment_code.public_plan(plan) == selection,
        "Saved selection differs from recomputed plan and source hashes",
    )
    identity = summary.get("identity", {})
    target = identity.get("target")
    model = identity.get("public_model")
    require(
        type(target) is int
        and 1 <= target <= 3
        and isinstance(model, dict)
        and set(model)
        <= {"model", "revision", "model_revision", "tokenizer_revision", "quantization"},
        "Invalid supplementary target or public model identity",
    )
    require(
        identity
        == {
            "plan_sha256": plan["plan_sha256"],
            "target": target,
            "public_model": model,
            "script_sha256": fingerprint(Path(experiment_code.__file__)),
            "constructor_sha256": fingerprint(Path(task_code.__file__)),
        },
        "Supplementary implementation or plan identity changed",
    )
    attempted = summary.get("attempts", [])
    selected_ids = [item["path"]["path_id"] for item in plan["selected_paths"]]
    require(
        [row["path_id"] for row in attempted] == selected_ids[: len(attempted)],
        "Attempt ledger is not an ordered prefix of the selected paths",
    )
    source_hashes = dict(plan["source_artifact_sha256"])
    for name in ("selection.json", "exploratory_task_summary.json"):
        source_hashes["experiment/" + name] = fingerprint(experiment / name)
    for row in attempted:
        directory = experiment / "examples" / row["path_id"]
        require(
            directory.resolve().parent == (experiment / "examples").resolve(),
            "Example source is outside its expected directory",
        )
        result = read_json(directory / "result.json")
        require(
            result.get("status") in STATUSES
            and result.get("review_verdict") in {None, "faithful", "unsupported", "unclear"},
            "Unrecognized result status",
        )
        require(
            experiment_code.safe_attempt(result, directory) == row,
            "Attempt ledger differs from its result or artifact receipts",
        )
        require(result.get("public_model") == model, "Task model identity differs from experiment")
        for name, sha in {"result.json": row["result_sha256"], **row["artifact_sha256"]}.items():
            source_hashes[f"experiment/examples/{row['path_id']}/{name}"] = sha
    passed = [row for row in attempted if row["locally_executed_consistent"]]
    require(
        summary.get("selected_examples") == passed
        and summary.get("locally_executed_consistent_count") == len(passed)
        and summary.get("requested_examples") == target
        and summary.get("maximum_path_attempts") == plan["maximum_path_attempts"]
        and summary.get("target_reached") is (len(passed) >= target)
        and summary.get("attempt_status_counts", {})
        == dict(Counter(row["status"] for row in attempted)),
        "Supplementary outcome counts do not match the verified attempt ledger",
    )
    if summary["status"] == "target_reached":
        require(
            len(passed) == target and attempted[-1]["locally_executed_consistent"],
            "Target-reached ledger is inconsistent with stopping rule",
        )
    else:
        require(
            len(passed) < target and len(attempted) == len(selected_ids),
            "Exhausted ledger still has unattempted candidates or reached its target",
        )
    return plan, summary, source_hashes


def verify_success(directory: Path, path: dict):
    """Check exact saved artifacts and scan actual database cells without executing model SQL."""
    spec = task_code.validate_fixture_spec(read_json(directory / "task.json"), path)
    result = read_json(directory / "result.json")
    review = read_json(directory / "independent_solution_review.json")
    validation = read_json(directory / "local_validation.json")
    expected = read_json(directory / "expected_result.json")
    require(
        spec.get("status") == "task"
        and spec.get("path_id") == path["path_id"]
        and read_json(directory / "source_path.json") == path,
        "Successful task source path differs from verified selection",
    )
    require(
        review.get("verdict") == "faithful"
        and validation == result.get("local_validation")
        and validation.get("status") == "locally_executed_consistent"
        and all(
            validation.get(key) is True
            for key in (
                "reference_query_executed",
                "independent_query_executed",
                "queries_agree",
                "nonempty_result",
                "output_schema_matches",
            )
        ),
        "Successful task does not have consistent review/execution receipts",
    )
    require(
        (directory / "instruction.md").read_text() == spec["learner_instruction"] + "\n"
        and (directory / "oracle.sql").read_text().strip() == spec["reference_sql"].strip()
        and (directory / "independent.sql").read_text().strip()
        == review["independent_sql"].strip(),
        "Published instructions or SQL differ from generated and reviewed versions",
    )
    task_code._select(review["independent_sql"])
    require(
        expected == validation.get("independent_result")
        and task_code.compare_query_results(
            validation["reference_result"],
            expected,
            ordered=bool(spec.get("ordered_output", False)),
        )
        and expected["columns"] == spec["output_columns"],
        "Expected output differs from the execution receipt",
    )
    # Database storage may compress strings: scan decoded rows as well as file bytes.
    export_checks.assert_publishable(encode(spec["tables"]), "synthetic fixture specification")
    database = task_code._connect(directory / "fixture.duckdb", read_only=True)
    try:
        tables = database.execute(
            "SELECT table_schema, table_name, table_type FROM information_schema.tables "
            "WHERE table_catalog=current_database()"
        ).fetchall()
        require(
            set(tables) == {("main", table["name"], "BASE TABLE") for table in spec["tables"]},
            "Fixture contains unexpected tables or views",
        )
        for table in spec["tables"]:
            values = task_code._query(database, "SELECT * FROM " + task_code._quote(table["name"]))
            require(
                values["columns"] == [column["name"] for column in table["columns"]]
                and len(values["rows"]) == len(table["rows"]),
                "Fixture shape differs from task",
            )
            export_checks.assert_publishable(encode(values), "synthetic fixture cell values")
    finally:
        database.close()
    return spec, validation


def publish(
    experiment: Path, run: Path, corpus: Path, output: Path, *, manual_review: Path | None = None
) -> dict:
    experiment, run, corpus, output = map(Path.resolve, (experiment, run, corpus, output))
    require(not output.exists(), "Refusing to replace an existing supplementary export")
    require(
        all(
            output != source and source not in output.parents and output not in source.parents
            for source in (experiment, run, corpus)
        ),
        "Export must be separate from all sources",
    )
    plan, summary, source_hashes = verify_experiment(experiment, run, corpus)
    reviews, review_sha = load_manual_reviews(manual_review, summary["selected_examples"])
    if manual_review is not None:
        require(
            output not in manual_review.resolve().parents,
            "Manual review source must be outside the export directory",
        )
        source_hashes["manual_review/receipt.json"] = review_sha
    planned = {item["path"]["path_id"]: item for item in plan["selected_paths"]}
    transition_index = {
        (row["source_history_id"], row["target_history_id"]): row["transition_id"]
        for row in map(json.loads, (corpus / "transitions.jsonl").read_text().splitlines())
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".exploratory-tasks-", dir=output.parent))
    files = {}

    def save(name, raw):
        export_checks.assert_publishable(raw, name)
        destination = staging / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def save_json(name, value):
        save(name, encode(value))

    try:
        attempts = [
            {
                key: row[key]
                for key in (
                    "path_id",
                    "status",
                    "evidence_label",
                    "independent_review_verdict",
                    "locally_executed_consistent",
                    "reference_query_executed",
                    "independent_query_executed",
                    "queries_agree",
                    "result_sha256",
                )
            }
            for row in summary["attempts"]
        ]
        published = []
        for row in summary["selected_examples"]:
            pid = row["path_id"]
            item, directory = planned[pid], experiment / "examples" / pid
            path = item["path"]
            semantic_review = public_manual_review(pid, reviews)
            spec, validation = verify_success(directory, path)
            base = f"examples/{pid}/"
            save_json(base + "review.json", semantic_review)
            for name in COPY_FILES:
                raw = (directory / name).read_bytes()
                require(
                    hashlib.sha256(raw).hexdigest() == row["artifact_sha256"][name],
                    "Successful artifact changed during publication",
                )
                save(base + name, raw)
            save_json(
                base + "task.json",
                {
                    **{
                        key: spec[key]
                        for key in (
                            "status",
                            "title",
                            "path_id",
                            "ordered_output",
                            "output_columns",
                            "path_transition_ids",
                        )
                    },
                    "format": FORMAT,
                    "synthetic_fixture": True,
                    "evidence_label": LABEL,
                    "manual_semantic_verdict": semantic_review["semantic_verdict"],
                    "learner_readiness_certified": False,
                },
            )
            save_json(
                base + "local_validation.json",
                {
                    **{
                        key: validation[key]
                        for key in (
                            "status",
                            "reference_query_executed",
                            "independent_query_executed",
                            "queries_agree",
                            "reference_result",
                            "independent_result",
                            "nonempty_result",
                            "output_schema_matches",
                        )
                    },
                    "evidence_label": LABEL,
                    "synthetic_fixture": True,
                    "learner_evaluated": False,
                    "universal_graph_contract_certified": False,
                    "manual_semantic_verdict": semantic_review["semantic_verdict"],
                },
            )
            save_json(
                base + "provenance.json",
                {
                    "path_id": pid,
                    "state_ids": path["state_ids"],
                    "evidence_label": LABEL,
                    "selection_origin": item["selection_origin"],
                    "resampling_seed": item["resampling_seed"],
                    "draft_review": item["draft_review"],
                    "edge_audit_evidence": item["edge_audit_evidence"],
                    "cross_task": True,
                    "universal_contract_certified": False,
                    "witnesses": [
                        {
                            "edge_id": step["transition_id"],
                            "source_history_id": step["source_history_id"],
                            "target_history_id": step["target_history_id"],
                            "task_id": step["source_task_id"],
                            "rollout_id": step["rollout_id"],
                            "witness_transition_id": transition_index[
                                step["source_history_id"], step["target_history_id"]
                            ],
                        }
                        for step in path["transitions"]
                    ],
                    "independent_review_verdict": "faithful",
                    "source_result_sha256": row["result_sha256"],
                    "manual_review": "review.json",
                },
            )
            save(
                base + "README.md",
                (
                    "# Witness-specific exploratory SQL task\n\n"
                    f"Evidence label: `{LABEL}`. Reusable edge applicability remains unresolved.\n\n"
                    + manual_review_warning(semantic_review)
                    + "\n\n"
                    + "See the [manual semantic review](review.json). The original artifacts remain unchanged, including any documented defects.\n\n"
                    "The task-input artifacts are [instruction.md](instruction.md) and "
                    "[fixture.duckdb](fixture.duckdb). [oracle.sql](oracle.sql), "
                    "[independent.sql](independent.sql), and [expected_result.json](expected_result.json) "
                    "are preserved reference outputs, not independently certified truth.\n\n"
                    "From this directory in the repository's Python environment:\n\n"
                    "```sh\npython -m superstate_graphs.graph_task_examples verify --task-dir . --submission answer.sql\n```\n\n"
                    "Two separately prompted queries ran and agreed on this synthetic fixture after a "
                    "LLM faithfulness review. No learner or training lift was evaluated. "
                    "See [provenance](provenance.json) and [execution receipt](local_validation.json).\n"
                ).encode(),
            )
            published.append(
                {
                    "path_id": pid,
                    "directory": base.rstrip("/"),
                    "evidence_label": LABEL,
                    "manual_semantic_verdict": semantic_review["semantic_verdict"],
                    "learner_readiness_certified": False,
                }
            )
        public = {
            "format": FORMAT,
            "evidence_label": LABEL,
            "status": summary["status"],
            "requested_examples": summary["requested_examples"],
            "maximum_path_attempts": plan["maximum_path_attempts"],
            "selected_path_count": len(plan["selected_paths"]),
            "attempted_path_count": len(attempts),
            "locally_executed_consistent_count": len(published),
            "target_reached": summary["target_reached"],
            "target_semantics": "Original stopping target counts local query agreement, regardless of later manual semantic failures.",
            "attempt_status_counts": dict(Counter(row["status"] for row in attempts)),
            "attempts": attempts,
            "examples": published,
            "identity": summary["identity"],
            "limitations": experiment_code.LIMITATIONS,
            "privacy_review_note": PRIVACY_LIMITATION,
            "primary_graph_and_counts_unchanged": True,
            "manual_review_supplied": manual_review is not None,
            "manual_semantic_verdict_counts": dict(
                Counter(row["manual_semantic_verdict"] for row in published)
            ),
            "learner_readiness_certified": False,
        }
        save_json("summary.json", public)
        save_json("selection.json", experiment_code.public_plan(plan))
        links = "\n".join(
            f"- [{row['path_id']}]({row['directory']}/README.md): manual semantic verdict `{row['manual_semantic_verdict']}`"
            + (
                " — **FAILED RESEARCH CASE, NOT LEARNER-READY**"
                if row["manual_semantic_verdict"] == "failed"
                else ""
            )
            for row in published
        )
        save(
            "README.md",
            (
                "# Supplementary witness-specific task examples\n\n"
                f"Evidence label: `{LABEL}`.\n\n"
                f"Published {len(published)} locally consistent examples from {len(attempts)} attempts "
                f"(target {summary['requested_examples']}). Status: `{summary['status']}`. "
                "Zero examples is a valid recorded outcome. These results are outside the primary "
                "graph's sampled-supported paths and executable-example counts.\n\n"
                "**Local query agreement is not semantic validation.** Manual review findings are recorded separately and do not replace the original run counts. Failed cases remain research evidence and are not learner-ready; missing reviews never count as passes.\n\n"
                "[Verified summary](summary.json) · [Selection and audit labels](selection.json) · "
                "[File and source hashes](manifest.json)\n\n"
                + (links + "\n\n" if links else "")
                + "Each example concerns particular cross-task witnesses. Unknown applicability remains "
                "unknown, including citation-invalidated judgments. Path selection admitted no raw "
                "or validated source-applicability audit contradiction; later manual task-semantic "
                "failures are disclosed separately. Selection was exploratory after the primary audit. "
                "Success establishes reviewed local fixture/query agreement, not reusable edges, "
                "arbitrary path executability, benchmark reproduction, learner success, or training lift.\n\n"
                + PRIVACY_LIMITATION
                + "\n"
            ).encode(),
        )
        manifest = {
            "format": FORMAT,
            "evidence_label": LABEL,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "publisher_sha256": fingerprint(Path(__file__)),
            "source_artifact_sha256": source_hashes,
            "files": files,
            "primary_graph_and_counts_unchanged": True,
            "publication_privacy_review_required": True,
            "privacy_review_note": PRIVACY_LIMITATION,
            "manual_review_supplied": manual_review is not None,
        }
        save_json("manifest.json", manifest)
        # Recheck every source, including failed-attempt files, before atomic publication.
        for name, expected in source_hashes.items():
            kind, relative = name.split("/", 1)
            source = (
                manual_review
                if kind == "manual_review"
                else {"run": run, "corpus": corpus, "experiment": experiment}[kind] / relative
            )
            require(fingerprint(source) == expected, "A source artifact changed during publication")
        staging.rename(output)
        return {
            "status": "published",
            "evidence_label": LABEL,
            "examples": len(published),
            "attempts": len(attempts),
            "manifest_sha256": fingerprint(output / "manifest.json"),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "run", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--manual-review",
        type=Path,
        help="Optional artifact-bound semantic review receipt; never changes fixed run outcomes",
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                publish(
                    args.experiment,
                    args.run,
                    args.corpus,
                    args.output,
                    manual_review=args.manual_review,
                )
            )
        )
    except Exception as error:
        parser.exit(
            1, f"Supplementary publication refused ({type(error).__name__}); sources unchanged.\n"
        )


if __name__ == "__main__":
    main()
