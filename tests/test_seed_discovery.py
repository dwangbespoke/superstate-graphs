"""Initialization uses exact before/after evidence and a reasoning teacher."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superstate_graphs.full_corpus import index_messages, render_history, render_step
from superstate_graphs.graph_evolution import GRAPH_CONTRACT, discover_seed


def rollout(task="task-0", rid="rollout-0", steps=9):
    messages = [
        {"role": "user", "content": f"QUERY_{task}", "content_json": None, "sequence_number": 1}
    ]
    for i in range(steps):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"ACTION_{i}",
                    "content_json": None,
                    "sequence_number": 2 * i + 2,
                },
                {
                    "role": "tool",
                    "content": f"OBSERVATION_{i}: command not found",
                    "content_json": None,
                    "sequence_number": 2 * i + 3,
                },
            ]
        )
    return {
        "id": rid,
        "task_id": task,
        "split": "train",
        "reward": 0,
        "reward_provenance": {"reason": "EXTERNAL_REWARD_SECRET"},
        **index_messages(messages),
    }


class Teacher:
    def __init__(self, failure=None):
        self.calls = []
        self.shards = []
        self.runtime = {"model": "teacher-model", "revision": "pinned", "api_key": "PRIVATE_KEY"}
        self.failure = failure

    def shard(self, rid):
        self.shards.append(rid)
        return self

    async def complete_json(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if kwargs["cache_namespace"] == "local-transition-graph-synthesis-v3":
            return {
                "router_instructions": "Use observed agent knowledge",
                "states": [
                    {
                        "id": "S",
                        "name": "Command unavailable",
                        "description": "Attempted command is unavailable; task remains unresolved",
                        "exclusions": ["Do not assume the requested task is complete"],
                    }
                ],
                "edges": [],
            }
        prefix = (
            messages[1]["content"]
            .split("<full_source_history>\n", 1)[1]
            .split("</full_source_history>", 1)[0]
        )
        observation = (
            messages[1]["content"]
            .split("<focal_tool_observation>\n", 1)[1]
            .split("</focal_tool_observation>", 1)[0]
        )
        source_quote = next(line for line in prefix.splitlines() if line.startswith("QUERY_"))
        target_quote = next(
            line for line in observation.splitlines() if line.startswith("OBSERVATION_")
        )
        if self.failure == "always":
            target_quote = "FABRICATED_OBSERVATION"
        if self.failure == "first_target_from_action" and kwargs["cache_namespace"].endswith("-0"):
            target_quote = "ACTION_" + target_quote.split("_", 1)[1].split(":", 1)[0]
        return {
            "source": {
                "name": "Unresolved requested work",
                "description": "The requested outcome remains to be established",
                "exclusions": [],
                "evidence_quote": source_quote,
            },
            "target": {
                "name": "Command unavailable",
                "description": "The attempted command is unavailable",
                "exclusions": ["No completed artifact was established"],
                "evidence_quote": target_quote,
            },
            "operation": "Attempt to run a command",
            "effect": "Command is unavailable",
            "observation_quote": next(
                line for line in observation.splitlines() if line.startswith("OBSERVATION_")
            ),
        }


class ForbiddenStudent:
    async def complete_json(self, *_args, **_kwargs):
        raise AssertionError("Initialization bypassed the teacher")


def test_discovery_uses219_exact_pairs_from73_tasks_with_reasoning_teacher(tmp_path: Path):
    teacher = Teacher()
    runtime = SimpleNamespace(directory=tmp_path, llm=ForbiddenStudent(), teacher_llm=teacher)
    runs = [rollout(f"task-{i}", f"r{i:03d}") for i in range(73)]
    candidate = asyncio.run(discover_seed(runtime, runs, tmp_path / "seed_candidate.json"))
    assert candidate
    records = json.loads((tmp_path / "seed_discoveries.json").read_text())
    assert len(records) == 219
    assert {record["task_id"] for record in records} == {run["task_id"] for run in runs}
    assert {record["step"] for record in records} == {0, 3, 6}
    assert all(record["status"] == "quotes_verified" for record in records)
    assert all(record["teacher"]["model"] == "teacher-model" for record in records)
    assert "PRIVATE_KEY" not in json.dumps(records)
    extraction = teacher.calls[:-1]
    assert len(extraction) == 219
    for (messages, options), record in zip(extraction, records):
        run = next(r for r in runs if r["id"] == record["rollout_id"])
        step = record["step"]
        assert render_history(run, step) in messages[1]["content"]
        assert render_step(run, step) in messages[1]["content"]
        assert f"OBSERVATION_{step + 1}:" not in messages[1]["content"]
        assert record["observation_quote"] in messages[1]["content"]
        assert messages[1]["content"].count(record["observation_quote"]) == 2
        assert GRAPH_CONTRACT not in messages[0]["content"]
        assert options["thinking"] is True and options["max_tokens"] == 8192
        assert "EXTERNAL_REWARD_SECRET" not in json.dumps(messages)
    assert teacher.calls[-1][1]["thinking"] is True
    assert teacher.calls[-1][1]["cache_namespace"] == "local-transition-graph-synthesis-v3"


def test_old_rejected_v2_discoveries_are_preserved_before_replacement(tmp_path: Path):
    old = [{"status": "rejected_quote_grounding", "proposal": {"bad": "OLD_FAILURE"}}]
    previous_bytes = json.dumps(old).encode()
    (tmp_path / "seed_discoveries.json").write_bytes(previous_bytes)
    runtime = SimpleNamespace(directory=tmp_path, llm=Teacher())
    asyncio.run(discover_seed(runtime, [rollout()], tmp_path / "seed_candidate.json"))
    assert (
        tmp_path / "initialization_rejected/v2/seed_discoveries.json"
    ).read_bytes() == previous_bytes
    new = json.loads((tmp_path / "seed_discoveries.json").read_text())
    assert all(row["discovery_version"] == "local-transition-v3" for row in new)
    assert all(row["status"] == "quotes_verified" for row in new)


def test_target_quote_must_come_from_focal_observation_not_assistant_action(tmp_path: Path):
    teacher = Teacher(failure="first_target_from_action")
    runtime = SimpleNamespace(directory=tmp_path, llm=teacher)
    asyncio.run(discover_seed(runtime, [rollout()], tmp_path / "seed_candidate.json"))
    extraction = [call for call in teacher.calls if "discovery-v3" in call[1]["cache_namespace"]]
    assert len(extraction) == 6
    repairs = [
        messages for messages, options in extraction if options["cache_namespace"].endswith("-1")
    ]
    assert len(repairs) == 3
    assert all("target.evidence_quote" in messages[-1]["content"] for messages in repairs)


def test_failed_grounding_cannot_seed_graph_and_is_retained_for_diagnosis(tmp_path: Path):
    teacher = Teacher(failure="always")
    runtime = SimpleNamespace(directory=tmp_path, llm=teacher)
    with pytest.raises(ValueError, match="Insufficient grounded"):
        asyncio.run(discover_seed(runtime, [rollout()], tmp_path / "seed_candidate.json"))
    assert len(teacher.calls) == 9
    assert not (tmp_path / "seed_candidate.json").exists()
    records = json.loads((tmp_path / "seed_discoveries.json").read_text())
    assert len(records) == 3
    assert all(record["status"] == "rejected_quote_grounding" for record in records)
    assert all(record["invalid_quote_fields"] == ["target.evidence_quote"] for record in records)
