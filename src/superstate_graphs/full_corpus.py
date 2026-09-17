"""Lossless, reward-separated indexing of the complete historical rollout corpus.

One state boundary follows one *observed agent turn*: the assistant action batch
and all its ensuing tool observations. The export combines multiple shell
commands into one aggregate terminal observation, so splitting a batch into
individual commands would invent unavailable intermediate observations. Parser
errors are observations of attempted actions and therefore advance the history.

History zero contains the original delivered query and initial environment
screen. History k contains that query and exactly k completed action/observation
groups. A trailing assistant message without an observation is retained as
``trailing_transcript`` but creates no state. No such unmatched tails occur in
the selected 1,030-rollout corpus. Terminal tool acknowledgements do create a
state; they are policy-visible observations, not external verifier rewards.

The rendered transcript is stored once per rollout, with character offsets for
all complete prefixes. Outcomes, provenance, split labels, and external grading
feedback never enter ``render_history`` or ``render_step``. The index contains
all selected rollouts and all boundaries; no sampling or truncation is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_SPLIT_SEED = "superstate-full-corpus-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def history_id(rollout_id: str, step: int) -> str:
    """Return a stable history identifier; step counts completed turns."""
    if step < 0:
        raise ValueError("History step must be nonnegative")
    return f"{rollout_id}:h{step:04d}"


def split_tasks(
    task_ids: Iterable[str],
    *,
    train_tasks: int = 73,
    pareto_tasks: int = 15,
    seed: str = DEFAULT_SPLIT_SEED,
) -> dict[str, str]:
    """Assign whole original tasks by a reward-independent seeded SHA-256 rank.

    The remainder is test. This ensures task-ID separation; it does not claim
    that semantic task families or closely related task variants are disjoint.
    """
    task_ids = sorted(set(task_ids))
    if min(train_tasks, pareto_tasks) < 0 or train_tasks + pareto_tasks > len(task_ids):
        raise ValueError("Split sizes exceed the number of distinct tasks")
    ordered = sorted(task_ids, key=lambda task: sha256_bytes(f"{seed}:{task}".encode()))
    return {
        task: "train" if i < train_tasks else "pareto" if i < train_tasks + pareto_tasks else "test"
        for i, task in enumerate(ordered)
    }


def _render_message(message: Mapping[str, Any]) -> str:
    """Render policy-visible fields only, preserving all content verbatim."""
    role = message["role"]
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError(f"Non-text message content for role {role!r}")
    result = f"=== {role.upper()} MESSAGE ===\n{content or ''}\n"
    if message.get("content_json") is not None:
        result += "=== STRUCTURED CONTENT ===\n" + _json(message["content_json"]) + "\n"
    return result + "\n"


def index_messages(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Index complete action/observation groups without inventing boundaries.

    Adjacent tool messages belong to the preceding assistant action batch and
    are kept together. Complete call-ID matching cannot be required here: this
    archive emits a single aggregate terminal observation without call IDs for
    a batch containing several tool calls. The next assistant turn determines
    that the aggregate observation is complete in the recorded transcript.
    """
    if not messages:
        raise ValueError("Empty transcript")
    sequences = [m["sequence_number"] for m in messages]
    if sequences != sorted(set(sequences)):
        raise ValueError("Messages must have unique, strictly increasing sequence numbers")
    cursor = 0
    initial: list[Mapping[str, Any]] = []
    while cursor < len(messages) and messages[cursor]["role"] in {"user", "system"}:
        initial.append(messages[cursor])
        cursor += 1
    if not any(m["role"] == "user" for m in initial):
        raise ValueError("A delivered user query must precede the first action")
    query = "\n\n".join(m.get("content") or "" for m in initial if m["role"] == "user")
    parts = ["".join(_render_message(m) for m in initial)]
    offset = len(parts[0])
    history_end_offsets = [offset]
    steps: list[dict[str, Any]] = []
    trailing = ""
    while cursor < len(messages):
        if messages[cursor]["role"] != "assistant":
            raise ValueError(
                f"Expected assistant at sequence {messages[cursor]['sequence_number']}"
            )
        action_messages: list[Mapping[str, Any]] = []
        while cursor < len(messages) and messages[cursor]["role"] == "assistant":
            action_messages.append(messages[cursor])
            cursor += 1
        if cursor == len(messages):
            trailing = "".join(_render_message(m) for m in action_messages)
            break
        observations: list[Mapping[str, Any]] = []
        while cursor < len(messages) and messages[cursor]["role"] == "tool":
            observations.append(messages[cursor])
            cursor += 1
        if not observations:
            raise ValueError("Assistant action is followed by neither tools nor end of transcript")
        action = "".join(_render_message(m) for m in action_messages)
        observation = "".join(_render_message(m) for m in observations)
        calls = [
            call
            for m in action_messages
            for call in (m.get("content_json") or {}).get("tool_calls", [])
        ]
        steps.append(
            {
                "step": len(steps),
                "start_char": offset,
                "action_end_char": offset + len(action),
                "end_char": offset + len(action) + len(observation),
                "action_sequence_numbers": [m["sequence_number"] for m in action_messages],
                "observation_sequence_numbers": [m["sequence_number"] for m in observations],
                "tool_call_count": len(calls),
                "tool_names": [c.get("function_name", "unknown") for c in calls],
                "has_structured_tool_calls": bool(calls),
            }
        )
        parts.extend([action, observation])
        offset += len(action) + len(observation)
        history_end_offsets.append(offset)
    return {
        "query": query,
        "transcript": "".join(parts),
        "history_end_offsets": history_end_offsets,
        "steps": steps,
        "history_count": len(history_end_offsets),
        "message_count": len(messages),
        "trailing_transcript": trailing,
    }


def render_history(rollout: Mapping[str, Any], step: int) -> str:
    """Return every recorded policy-visible message through completed step k."""
    offsets = rollout["history_end_offsets"]
    if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < len(offsets):
        raise IndexError(f"Invalid history step {step!r}; expected 0..{len(offsets) - 1}")
    return rollout["transcript"][: offsets[step]]


def render_step(rollout: Mapping[str, Any], step: int) -> str:
    """Return transition k's complete action plus observation (zero based)."""
    if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < len(rollout["steps"]):
        raise IndexError(f"Invalid transition step {step!r}")
    boundary = rollout["steps"][step]
    return rollout["transcript"][boundary["start_char"] : boundary["end_char"]]


def iter_history_records(rollout: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield compact prefix references; reward fields are analysis-only metadata."""
    digest = hashlib.sha256()
    previous = 0
    for step, end in enumerate(rollout["history_end_offsets"]):
        digest.update(rollout["transcript"][previous:end].encode("utf-8"))
        previous = end
        yield {
            "history_id": history_id(rollout["id"], step),
            "rollout_id": rollout["id"],
            "task_id": rollout["task_id"],
            "step": step,
            "split": rollout["split"],
            "prefix_char_end": end,
            "prefix_sha256": digest.hexdigest(),
            "reward": rollout["reward"],
            "reward_provenance": rollout["reward_provenance"],
            "last_observed_history": step == len(rollout["steps"]),
        }


def iter_transition_records(rollout: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for step, boundary in enumerate(rollout["steps"]):
        yield {
            "transition_id": f"{rollout['id']}:t{step:04d}",
            "rollout_id": rollout["id"],
            "task_id": rollout["task_id"],
            "split": rollout["split"],
            "source_history_id": history_id(rollout["id"], step),
            "target_history_id": history_id(rollout["id"], step + 1),
            **boundary,
        }


def load_corpus(
    dataset_root: Path | str,
    *,
    manifest_name: str = "manifest.json",
    train_tasks: int = 73,
    pareto_tasks: int = 15,
    split_seed: str = DEFAULT_SPLIT_SEED,
    verify_source_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Load every manifest-selected rollout, with no filtering based on reward."""
    root = Path(dataset_root)
    manifest = _read_json(root / manifest_name)
    associations = manifest["rollouts"]
    identifiers = [a["rollout_id"] for a in associations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Manifest contains duplicate rollout IDs")
    splits = split_tasks(
        (a["public_task"] for a in associations),
        train_tasks=train_tasks,
        pareto_tasks=pareto_tasks,
        seed=split_seed,
    )
    assignments = {}
    assignment_path = manifest.get("grade_assignment_audit")
    if assignment_path:
        record = _read_json(root / assignment_path)
        assignments[record["rollout_id"]] = (record, assignment_path)
    rollouts = []
    for association in sorted(associations, key=lambda a: (a["public_task"], a["rollout_id"])):
        rid = association["rollout_id"]
        directory = root / "rollouts" / rid
        raw_messages = (directory / "messages.json").read_bytes()
        source_hash = sha256_bytes(raw_messages)
        messages = json.loads(raw_messages)
        complete = _read_json(directory / "complete.json")
        if verify_source_hashes and source_hash != complete["messages.json"]:
            raise ValueError(f"Transcript SHA-256 mismatch: {rid}")
        if len(messages) != complete["message_count"]:
            raise ValueError(f"Transcript message-count mismatch: {rid}")
        detail_raw = (directory / "detail.json").read_bytes()
        detail = json.loads(detail_raw)
        reward = detail["grade_result"]["score"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError(f"Non-numeric terminal reward: {rid}")
        if not math.isfinite(reward) or reward != detail["extracted_score"]:
            raise ValueError(f"Missing, nonfinite, or inconsistent terminal reward: {rid}")
        provenance: dict[str, Any] = {
            "kind": "original_verifier",
            "source": f"rollouts/{rid}/detail.json",
            "source_sha256": sha256_bytes(detail_raw),
            "field": "grade_result.score",
        }
        if rid in assignments:
            assignment, path = assignments[rid]
            if reward != assignment["assigned_score"]:
                raise ValueError(f"Assigned grade disagrees with provenance: {rid}")
            provenance.update(
                {
                    "kind": "manual_trace_review",
                    "assignment_record": path,
                    "assignment_sha256": sha256_bytes((root / path).read_bytes()),
                    "original_verifier_executed": assignment["original_verifier_executed"],
                    "reason": assignment["reason"],
                }
            )
        indexed = index_messages(messages)
        rollouts.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": rid,
                "rollout_id": rid,
                "task_id": association["public_task"],
                "horizon_task_id": detail["task_id"],
                "task_version_id": association["task_version_id"],
                "cohort_id": association.get("cohort_id"),
                "split": splits[association["public_task"]],
                "reward": reward,
                "reward_provenance": provenance,
                "source_messages_sha256": source_hash,
                "transcript_sha256": sha256_bytes(indexed["transcript"].encode("utf-8")),
                **indexed,
            }
        )
    expected = manifest.get("counts", {}).get("rollouts")
    if expected is not None and len(rollouts) != expected:
        raise ValueError("Manifest membership does not match the declared rollout count")
    return rollouts


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p90": ordered[math.ceil(0.90 * len(ordered)) - 1],
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p99": ordered[math.ceil(0.99 * len(ordered)) - 1],
        "max": max(values),
        "total": sum(values),
    }


def corpus_statistics(rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rollouts:
        raise ValueError("Cannot summarize an empty corpus")
    tasks: dict[str, str] = {}
    for r in rollouts:
        if r["task_id"] in tasks and tasks[r["task_id"]] != r["split"]:
            raise ValueError("Original task crosses dataset splits")
        tasks[r["task_id"]] = r["split"]
    history_chars = [end for r in rollouts for end in r["history_end_offsets"]]
    return {
        "tasks": len(tasks),
        "rollouts": len(rollouts),
        "messages": sum(r["message_count"] for r in rollouts),
        "histories": sum(r["history_count"] for r in rollouts),
        "transitions": sum(len(r["steps"]) for r in rollouts),
        "task_split_counts": dict(Counter(tasks.values())),
        "rollout_split_counts": dict(Counter(r["split"] for r in rollouts)),
        "history_split_counts": {
            split: sum(r["history_count"] for r in rollouts if r["split"] == split)
            for split in ("train", "pareto", "test")
        },
        "rollouts_per_task": dict(sorted(Counter(r["task_id"] for r in rollouts).items())),
        "reward_counts": dict(Counter(str(r["reward"]) for r in rollouts)),
        "reward_provenance_counts": dict(Counter(r["reward_provenance"]["kind"] for r in rollouts)),
        "histories_per_rollout": _distribution([r["history_count"] for r in rollouts]),
        "full_rollout_characters": _distribution([len(r["transcript"]) for r in rollouts]),
        "history_characters": _distribution(history_chars),
        "history_tokens_rough_chars_divided_by_four": _distribution(
            [math.ceil(value / 4) for value in history_chars]
        ),
        "token_estimate_warning": "Character heuristic only; use target-model tokenization for limits.",
        "unstructured_action_groups": sum(
            not step["has_structured_tool_calls"] for r in rollouts for step in r["steps"]
        ),
        "multi_call_action_groups": sum(
            step["tool_call_count"] > 1 for r in rollouts for step in r["steps"]
        ),
        "unmatched_trailing_assistant_rollouts": sum(
            bool(r["trailing_transcript"]) for r in rollouts
        ),
        "task_disjoint": True,
        "semantic_family_disjointness_verified": False,
    }


def _atomic_write(path: Path, chunks: Iterable[str]) -> str:
    """Publish a complete artifact atomically; refuse changes to an existing one."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for chunk in chunks:
                data = chunk.encode("utf-8")
                digest.update(data)
                handle.write(data)
        if path.exists():
            current = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    current.update(chunk)
            if current.hexdigest() != digest.hexdigest():
                raise FileExistsError(f"Refusing to replace a different corpus artifact: {path}")
            temporary.unlink()
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def export_corpus(
    dataset_root: Path | str,
    output: Path | str,
    *,
    rollouts: Sequence[Mapping[str, Any]] | None = None,
    split_seed: str = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Write the entire corpus as lossless transcripts and compact prefix refs."""
    root, destination = Path(dataset_root), Path(output)
    if rollouts is None:
        rollouts = load_corpus(root, split_seed=split_seed)
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    records = {
        "rollouts.jsonl": iter(rollouts),
        "histories.jsonl": (h for r in rollouts for h in iter_history_records(r)),
        "transitions.jsonl": (t for r in rollouts for t in iter_transition_records(r)),
    }
    for name, rows in records.items():
        hashes[name] = _atomic_write(destination / name, (_json(row) + "\n" for row in rows))
    stats = corpus_statistics(rollouts)
    split_rows: dict[str, dict[str, list[str]]] = {}
    for split in ("train", "pareto", "test"):
        members = [r for r in rollouts if r["split"] == split]
        split_rows[split] = {
            "task_ids": sorted({r["task_id"] for r in members}),
            "rollout_ids": sorted(r["id"] for r in members),
        }
    hashes["splits.json"] = _atomic_write(
        destination / "splits.json",
        [json.dumps({"seed": split_seed, **split_rows}, indent=2) + "\n"],
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": sha256_bytes((root / "manifest.json").read_bytes()),
        "split_seed": split_seed,
        "statistics": stats,
        "artifact_sha256": hashes,
        "history_semantics": (
            "Initial delivered query plus each complete assistant action batch and aggregate "
            "tool observation. No partial action/observation groups. All prefixes are retained."
        ),
        "model_input": "render_history and render_step exclude external reward and grade metadata",
        "storage": "transcript stored once per rollout; history offsets are Unicode character offsets",
        "historical_benchmark_notice": (
            "Archived Horizon/Sonnet 4.5 trajectories mapped to Data Eng Bench task families; "
            "not executions of the current public benchmark."
        ),
    }
    _atomic_write(destination / "manifest.json", [json.dumps(manifest, indent=2) + "\n"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()
    manifest = export_corpus(args.dataset_root, args.output, split_seed=args.split_seed)
    print(json.dumps(manifest["statistics"], indent=2))


if __name__ == "__main__":
    main()
