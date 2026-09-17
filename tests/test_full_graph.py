"""Offline full-corpus orchestration, fallback, and transition-accounting tests."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superstate_graphs.full_corpus import index_messages, render_history, split_tasks
from superstate_graphs.full_graph import (
    DistinctTaskSampler,
    build_witnessed_edges,
    complete_unassigned,
    exhaustive_assign,
)
from superstate_graphs.graph_evolution import candidate_from_graph
from superstate_graphs.graph_schemas import HISTORY_SUFFIX


def _corpus(tasks=103, repetitions=10):
    task_ids = [f"task-{i:03d}" for i in range(tasks)]
    splits = split_tasks(task_ids) if tasks == 103 else {task: "train" for task in task_ids}
    return [
        {"id": f"{task}-r{j}", "task_id": task, "split": splits[task]}
        for task in task_ids
        for j in range(repetitions)
    ]


def _run(rid="r1", task="task1", suffix=""):
    pairs = [
        ("user", "QUERY"),
        ("assistant", "ACTION_ONE"),
        ("tool", "OBSERVATION_ONE"),
        ("assistant", "ACTION_TWO"),
        ("tool", "OBSERVATION_TWO"),
    ]
    return {
        "id": rid,
        "task_id": task,
        "split": "train",
        "reward": 1,
        "reward_provenance": {"reason": "OUTCOME_SECRET"},
        **index_messages(
            [
                {
                    "role": role,
                    "content": text + suffix,
                    "content_json": None,
                    "sequence_number": i + 1,
                }
                for i, (role, text) in enumerate(pairs)
            ]
        ),
    }


def _graph():
    return {
        "router_instructions": "Use current evidence",
        "states": [
            {"id": "A", "name": "Initial", "description": "Uninspected query", "exclusions": []},
            {"id": "B", "name": "Observed", "description": "Observed facts", "exclusions": []},
        ],
        "edges": [
            {
                "id": "AB",
                "source": "A",
                "target": "B",
                "operation": "Inspect",
                "effect": "Observe facts",
                "bindings": "Rename source",
            }
        ],
    }


def _assignment_rows(run, states=("A", "B", "B")):
    return [
        {
            "history_id": f"{run['id']}:h{step:04d}",
            "step": step,
            "state_id": state,
            "evidence": f"Evidence {step}",
            "rollout_id": run["id"],
            "task_id": run["task_id"],
            "split": run["split"],
        }
        for step, state in enumerate(states)
    ]


def _assignments(run, states=("A", "B", "B")):
    return {row["history_id"]: row for row in _assignment_rows(run, states)}


def test_sampler_covers_all730_training_rollouts_without_task_repeats_in_batch():
    corpus = _corpus()
    assert len(corpus) == 1030
    train = [r for r in corpus if r["split"] == "train"]
    assert len(train) == 730
    assert sum(r["split"] == "pareto" for r in corpus) == 150
    assert sum(r["split"] == "test" for r in corpus) == 150
    sampler = DistinctTaskSampler(train, size=6, seed=17)
    repeated = DistinctTaskSampler(train, size=6, seed=17)
    visited = set()
    for i in range(122):
        indices = sampler.next_minibatch_ids(None, SimpleNamespace(i=i))
        assert indices == repeated.next_minibatch_ids(None, SimpleNamespace(i=i))
        assert len({train[j]["task_id"] for j in indices}) == 6
        assert all(train[j]["split"] == "train" for j in indices)
        visited.update(indices)
    assert visited == set(range(730))


@pytest.mark.parametrize("size", [0, 3])
def test_sampler_rejects_impossible_distinct_task_batch(size):
    with pytest.raises(ValueError, match="distinct task count"):
        DistinctTaskSampler(_corpus(tasks=2, repetitions=2), size=size, seed=17)


def test_exhaustive_assignment_preserves_all_histories_and_writes_snapshot(tmp_path: Path):
    corpus = [_run("r1"), _run("r2", "task2")]

    class Runtime:
        async def batch(self, candidate, rollouts, *, judge):
            assert not judge
            assert rollouts == corpus
            return [_assignment_rows(r) for r in rollouts]

    output = tmp_path / "all.json"
    assignments = asyncio.run(
        exhaustive_assign(Runtime(), candidate_from_graph(_graph()), corpus, output)
    )
    assert len(assignments) == 6
    assert set(assignments) == {f"r{r}:h{step:04d}" for r in (1, 2) for step in range(3)}
    assert json.loads(output.read_text()) == assignments


def test_exhaustive_assignment_rejects_wrong_ids_even_when_count_matches(tmp_path: Path):
    run = _run()

    class Runtime:
        async def batch(self, *args, **kwargs):
            rows = _assignment_rows(run)
            rows[0]["history_id"] = "invented:h0000"
            return [rows]

    with pytest.raises(ValueError, match="exactly every expected history"):
        asyncio.run(
            exhaustive_assign(
                Runtime(), candidate_from_graph(_graph()), [run], tmp_path / "all.json"
            )
        )


class CompletionLLM:
    def __init__(self, *, malformed=False, never_assign=False):
        self.runtime = {"model": "fake", "revision": "pinned", "api_key": "CREDENTIAL_SECRET"}
        self.calls = []
        self.round = 0
        self.malformed = malformed
        self.never_assign = never_assign

    def shard(self, _):
        return self

    async def complete_json(self, messages, **kwargs):
        self.calls.append(messages)
        system, user = messages[0]["content"], messages[1]["content"]
        if "Describe the precise current local situation" in system:
            return {
                "name": "Uncovered fact",
                "description": "A distinct observation",
                "exclusions": [],
            }
        if "Create a fallback state codebook" in user:
            self.round += 1
            return {
                "router_instructions": "Use local facts",
                "states": [
                    {
                        "id": f"X{self.round}_new",
                        "name": "New local state",
                        "description": "A distinct observation",
                        "exclusions": [],
                    }
                ],
                "edges": [],
            }
        if system.startswith("Classify this full policy-visible history"):
            if self.malformed:
                return {"evidence": "Missing required state"}
            state = (
                None
                if self.never_assign or self.round == 1 and "OBSERVATION_TWO" in user
                else f"X{self.round}_new"
            )
            return {
                "state_id": state,
                "evidence": "Full-prefix evidence",
                "history_id": "MODEL_MUST_NOT_OVERRIDE",
                "step": 999,
            }
        raise AssertionError("Unexpected request")


def test_completion_routes_every_unknown_full_prefix_and_preserves_old_rows_and_snapshot(
    tmp_path: Path,
):
    run = _run()
    initial = _assignments(run, ("A", None, None))
    original = copy.deepcopy(initial)
    snapshot = tmp_path / "optimized_assignments_all.json"
    snapshot.write_text(json.dumps(initial))
    llm = CompletionLLM()
    graph, assignments = asyncio.run(
        complete_unassigned(
            SimpleNamespace(llm=llm),
            candidate_from_graph(_graph()),
            [run],
            initial,
            tmp_path,
        )
    )
    assert assignments["r1:h0000"] == original["r1:h0000"]
    assert assignments["r1:h0001"]["state_id"] == "X1_new"
    assert assignments["r1:h0002"]["state_id"] == "X2_new"
    assert assignments["r1:h0001"]["step"] == 1
    assert assignments["r1:h0002"]["history_id"] == "r1:h0002"
    assert json.loads(snapshot.read_text()) == original
    routed = [
        messages[1]["content"]
        for messages in llm.calls
        if messages[0]["content"].startswith("Classify this full policy-visible history")
    ]
    assert routed == [render_history(run, step) + HISTORY_SUFFIX for step in (1, 2, 2)]
    assert "OUTCOME_SECRET" not in json.dumps(llm.calls)
    assert len(graph["routing_stages"]) == 3
    checkpoint = json.loads((tmp_path / "completion_checkpoint.json").read_text())
    assert "CREDENTIAL_SECRET" not in json.dumps(checkpoint)


def test_completion_resumes_matching_inputs_but_rejects_changed_corpus(tmp_path: Path):
    run, original = _run(), _assignments(_run(), ("A", None, None))
    candidate = candidate_from_graph(_graph())
    llm = CompletionLLM()
    runtime = SimpleNamespace(llm=llm)
    completed, assignments = asyncio.run(
        complete_unassigned(runtime, candidate, [run], copy.deepcopy(original), tmp_path)
    )
    prior_calls = len(llm.calls)
    again_graph, again_assignments = asyncio.run(
        complete_unassigned(runtime, candidate, [run], copy.deepcopy(original), tmp_path)
    )
    assert completed == again_graph and assignments == again_assignments
    assert len(llm.calls) == prior_calls
    with pytest.raises(ValueError, match="checkpoint does not match"):
        asyncio.run(
            complete_unassigned(
                runtime, candidate, [_run(suffix="changed")], copy.deepcopy(original), tmp_path
            )
        )
    with pytest.raises(ValueError, match="checkpoint does not match"):
        revised = copy.deepcopy(original)
        revised["r1:h0000"]["evidence"] = "Different routing result"
        asyncio.run(complete_unassigned(runtime, candidate, [run], revised, tmp_path))


def test_completion_rejects_missing_state_id_and_honestly_keeps_unassigned_after_limit(
    tmp_path: Path,
):
    run = _run()
    with pytest.raises(ValueError, match="invalid state or evidence"):
        asyncio.run(
            complete_unassigned(
                SimpleNamespace(llm=CompletionLLM(malformed=True)),
                candidate_from_graph(_graph()),
                [run],
                _assignments(run, ("A", None, None)),
                tmp_path / "malformed",
            )
        )
    graph, assignments = asyncio.run(
        complete_unassigned(
            SimpleNamespace(llm=CompletionLLM(never_assign=True)),
            candidate_from_graph(_graph()),
            [run],
            _assignments(run, ("A", None, None)),
            tmp_path / "uncovered",
            max_rounds=1,
        )
    )
    assert sum(row["state_id"] is None for row in assignments.values()) == 2
    assert len(graph["routing_stages"]) == 2


def test_witnessed_edges_account_for_every_transition_once_including_unassigned(tmp_path: Path):
    runs = [_run("r1", "task1"), _run("r2", "task2")]
    assignments = {**_assignments(runs[0]), **_assignments(runs[1], ("A", "B", None))}
    original = _graph()
    graph = {"nodes": original["states"], "edges": original["edges"], "routing_stages": []}

    class ContractLLM:
        def __init__(self):
            self.requests = []

        async def complete_json(self, messages, **kwargs):
            assert kwargs.get("schema") is not None
            evidence = json.loads(messages[1]["content"].removesuffix(HISTORY_SUFFIX))
            self.requests.append(evidence)
            for witness in evidence["witnesses"]:
                run = next(r for r in runs if r["id"] == witness["rollout_id"])
                assert witness["source_history"] == render_history(run, witness["step"])
            return {
                "operation": "Inspect",
                "effect": "Observe",
                "bindings": "Rename only",
                "universal_source_plausible": False,
                "limitations": "Only witnessed applicability",
            }

    llm = ContractLLM()
    output = asyncio.run(
        build_witnessed_edges(SimpleNamespace(llm=llm), graph, assignments, runs, tmp_path)
    )
    accounted = [w["transition_id"] for edge in output["edges"] for w in edge["witnesses"]]
    accounted.extend(w["transition_id"] for w in output["unassigned_transitions"])
    assert len(accounted) == len(set(accounted)) == 4
    assert set(accounted) == {f"r{rid}:t{step:04d}" for rid in (1, 2) for step in range(2)}
    assert output["observed_transition_count"] == 3
    assert len(output["unassigned_transitions"]) == 1
    assert all(not edge["traversable"] for edge in output["edges"])
    assert all(edge["witness_count"] == len(edge["witnesses"]) for edge in output["edges"])
    ab = next(edge for edge in output["edges"] if edge["source"] == "A")
    assert ab["witness_count"] == 2 and ab["distinct_task_count"] == 2
    assert output["optimized_edge_spec"] == original["edges"]
    assert "OUTCOME_SECRET" not in json.dumps(llm.requests)
