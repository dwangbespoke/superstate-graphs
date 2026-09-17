"""Lossless prefix boundaries, provenance, and dataset-separation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from superstate_graphs.full_corpus import (
    corpus_statistics,
    export_corpus,
    history_id,
    index_messages,
    iter_history_records,
    iter_transition_records,
    load_corpus,
    render_history,
    render_step,
    split_tasks,
)


def _message(role: str, content: str, sequence: int, calls=None):
    return {
        "id": f"message-{sequence}",
        "role": role,
        "content": content,
        "sequence_number": sequence,
        "content_json": {"tool_calls": calls} if calls is not None else None,
    }


def _messages():
    return [
        _message("user", "ORIGINAL_QUERY\nInitial screen: π", 1),
        _message(
            "assistant",
            "FIRST_ACTION",
            2,
            [
                {"function_name": "bash_command", "arguments": {"keystrokes": "one\n"}},
                {"function_name": "bash_command", "arguments": {"keystrokes": "two\n"}},
            ],
        ),
        _message("tool", "FIRST_OBSERVATION", 3),
        _message("assistant", "FUTURE_ACTION", 4),
        _message("tool", "FUTURE_PARSER_ERROR", 5),
    ]


def _rollout():
    return {
        "id": "r1",
        "rollout_id": "r1",
        "task_id": "task1",
        "split": "train",
        "reward": 1,
        "reward_provenance": {"kind": "original_verifier"},
        **index_messages(_messages()),
    }


def test_all_complete_prefixes_have_no_future_or_reward_leakage():
    rollout = _rollout()
    rollout["reward_provenance"]["secret_verifier"] = "EXTERNAL_GRADE_SECRET"
    assert rollout["history_count"] == 3
    assert "ORIGINAL_QUERY\nInitial screen: π" in render_history(rollout, 0)
    assert "FIRST_ACTION" not in render_history(rollout, 0)
    prefix = render_history(rollout, 1)
    assert "FIRST_ACTION" in prefix and "FIRST_OBSERVATION" in prefix
    assert "FUTURE_ACTION" not in prefix and "FUTURE_PARSER_ERROR" not in prefix
    assert "EXTERNAL_GRADE_SECRET" not in render_history(rollout, 2)
    assert "reward_provenance" not in render_history(rollout, 2)
    assert render_history(rollout, 2) == rollout["transcript"]
    assert render_history(rollout, 0) + render_step(rollout, 0) == prefix


def test_parallel_commands_and_multiple_observations_are_atomic():
    messages = _messages()[:3] + [_message("tool", "SECOND_TOOL_OBSERVATION", 4)]
    rollout = index_messages(messages)
    assert rollout["history_count"] == 2
    assert rollout["steps"][0]["tool_call_count"] == 2
    assert rollout["steps"][0]["observation_sequence_numbers"] == [3, 4]
    assert "SECOND_TOOL_OBSERVATION" in render_history(rollout, 1)


def test_unstructured_attempt_and_parser_error_are_a_real_step():
    rollout = index_messages(_messages())
    assert not rollout["steps"][1]["has_structured_tool_calls"]
    assert "FUTURE_PARSER_ERROR" in render_history(rollout, 2)


def test_final_unobserved_assistant_message_does_not_invent_history():
    messages = _messages() + [_message("assistant", "UNOBSERVED_FINAL_MESSAGE", 6)]
    rollout = index_messages(messages)
    assert rollout["history_count"] == 3
    assert "UNOBSERVED_FINAL_MESSAGE" not in rollout["transcript"]
    assert "UNOBSERVED_FINAL_MESSAGE" in rollout["trailing_transcript"]


def test_prefix_hashes_match_full_unicode_rendering_and_transitions_connect():
    rollout = _rollout()
    records = list(iter_history_records(rollout))
    assert len(records) == 3
    for record in records:
        prefix = render_history(rollout, record["step"])
        assert record["prefix_sha256"] == hashlib.sha256(prefix.encode()).hexdigest()
    transitions = list(iter_transition_records(rollout))
    for step, transition in enumerate(transitions):
        assert transition["source_history_id"] == history_id("r1", step)
        assert transition["target_history_id"] == history_id("r1", step + 1)
    assert records[-1]["last_observed_history"]


@pytest.mark.parametrize("step", [-1, 3, True, 0.5])
def test_invalid_history_boundaries_fail(step):
    with pytest.raises(IndexError):
        render_history(_rollout(), step)


def test_bad_order_or_unpaired_observations_fail():
    messages = _messages()
    messages[2]["sequence_number"] = 2
    with pytest.raises(ValueError, match="sequence numbers"):
        index_messages(messages)
    with pytest.raises(ValueError, match="Expected assistant"):
        index_messages([_message("user", "query", 1), _message("tool", "orphan", 2)])


def test_whole_task_split_is_deterministic_and_has_requested_counts():
    tasks = [f"task-{i}" for i in range(103)]
    a = split_tasks(tasks)
    assert a == split_tasks(list(reversed(tasks)) * 10)
    assert list(a.values()).count("train") == 73
    assert list(a.values()).count("pareto") == 15
    assert list(a.values()).count("test") == 15
    assert a != split_tasks(tasks, seed="different")


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    associations = []
    for number in range(3):
        rid = f"r{number}"
        directory = root / "rollouts" / rid
        directory.mkdir(parents=True)
        messages = json.dumps(_messages()).encode()
        (directory / "messages.json").write_bytes(messages)
        (directory / "complete.json").write_text(
            json.dumps(
                {
                    "messages.json": hashlib.sha256(messages).hexdigest(),
                    "message_count": 5,
                }
            )
        )
        (directory / "detail.json").write_text(
            json.dumps(
                {
                    "task_id": f"horizon-{number}",
                    "extracted_score": number % 2,
                    "grade_result": {"score": number % 2, "SECRET": "VERIFIER_SECRET"},
                }
            )
        )
        associations.append(
            {
                "rollout_id": rid,
                "public_task": f"task-{number}",
                "task_version_id": "v1",
            }
        )
    (root / "assignment.json").write_text(
        json.dumps(
            {
                "rollout_id": "r0",
                "assigned_score": 0,
                "original_verifier_executed": False,
                "reason": "Assigned from trace review; original verifier unavailable.",
            }
        )
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "rollouts": associations,
                "counts": {"rollouts": 3},
                "grade_assignment_audit": "assignment.json",
            }
        )
    )
    return root


def test_manifest_loader_includes_failed_runs_and_grade_provenance(tmp_path: Path):
    root = _dataset(tmp_path)
    corpus = load_corpus(root, train_tasks=1, pareto_tasks=1)
    assert len(corpus) == 3
    assert sum(r["reward"] for r in corpus) == 1
    assert corpus[0]["reward_provenance"]["kind"] == "manual_trace_review"
    assert corpus[1]["reward_provenance"]["kind"] == "original_verifier"
    assert all("VERIFIER_SECRET" not in r["transcript"] for r in corpus)
    assert corpus_statistics(corpus)["histories"] == 9


def test_source_integrity_failure_is_not_silently_accepted(tmp_path: Path):
    root = _dataset(tmp_path)
    (root / "rollouts/r0/messages.json").write_text("[]")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_corpus(root, train_tasks=1, pareto_tasks=1)


def test_export_is_complete_idempotent_and_refuses_changed_corpus(tmp_path: Path):
    root = _dataset(tmp_path)
    corpus = load_corpus(root, train_tasks=1, pareto_tasks=1)
    output = tmp_path / "output"
    manifest = export_corpus(root, output, rollouts=corpus)
    assert manifest == export_corpus(root, output, rollouts=corpus)
    assert len((output / "rollouts.jsonl").read_text().splitlines()) == 3
    assert len((output / "histories.jsonl").read_text().splitlines()) == 9
    assert len((output / "transitions.jsonl").read_text().splitlines()) == 6
    for artifact, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / artifact).read_bytes()).hexdigest() == digest
    corpus[0]["reward"] = 1
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        export_corpus(root, output, rollouts=corpus)
