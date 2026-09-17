"""Offline integration checks for full-prefix graph evolution and retention."""

from __future__ import annotations

import asyncio
import copy
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from gepa.core.adapter import EvaluationBatch
from gepa.proposer.base import CandidateProposal

from superstate_graphs.full_corpus import index_messages, render_history
from superstate_graphs.graph_schemas import HISTORY_SUFFIX
from superstate_graphs.graph_evolution import (
    CoverageAcceptance,
    GraphGEPAAdapter,
    GraphRuntime,
    candidate_from_graph,
    canonical,
    digest,
    numbered_transcript,
    supported_sets,
    validate_spec,
)


def graph(label="baseline"):
    return {
        "router_instructions": label,
        "states": [
            {
                "id": "A",
                "name": "Uninspected",
                "description": "Query awaiting inspection",
                "exclusions": [],
            },
            {
                "id": "B",
                "name": "Inspected",
                "description": "Observed source evidence",
                "exclusions": [],
            },
        ],
        "edges": [
            {
                "id": "inspect",
                "source": "A",
                "target": "B",
                "operation": "Inspect source",
                "effect": "Source facts observed",
                "bindings": "Source names may differ",
            },
            {
                "id": "inspect_more",
                "source": "B",
                "target": "B",
                "operation": "Inspect further",
                "effect": "Additional facts observed",
                "bindings": "Source names may differ",
            },
        ],
    }


def rollout(rid="r1", task="task1", split="train", suffix=""):
    messages = [
        {"role": role, "content": text + suffix, "content_json": None, "sequence_number": i + 1}
        for i, (role, text) in enumerate(
            [
                ("user", "QUERY"),
                ("assistant", "ACTION_ONE"),
                ("tool", "OBSERVATION_ONE"),
                ("assistant", "ACTION_TWO"),
                ("tool", "OBSERVATION_TWO"),
            ]
        )
    ]
    return {
        "id": rid,
        "task_id": task,
        "split": split,
        "reward": 0,
        "reward_provenance": {"kind": "manual_trace_review", "reason": "SECRET_REWARD_DATA"},
        **index_messages(messages),
    }


def verdict():
    return {
        "membership": [True, True, True],
        "transitions": [
            {"supported": True, "edge_id": "inspect"},
            {"supported": True, "edge_id": "inspect_more"},
        ],
        "coherence": 1.0,
        "outgoing_applicability": 1.0,
        "feedback": [
            {
                "kind": "edge",
                "step": 0,
                "problem": "Needs evidence",
                "evidence": "ACTION_ONE",
                "suggestion": "Review boundary",
            }
        ],
        "redundant_states": [],
    }


class FakeLLM:
    def __init__(self, verdicts=None, model="fake-v1"):
        self.runtime = {"model": model, "revision": "pinned", "api_key": "SECRET_KEY"}
        self.calls = []
        self.verdicts = list(verdicts or [verdict()])

    async def complete_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if "frozen evaluator" in messages[0]["content"]:
            return copy.deepcopy(
                self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
            )
        return {
            "state_id": "B" if "OBSERVATION_ONE" in messages[1]["content"] else "A",
            "evidence": "Visible prefix evidence",
        }


def test_spec_rejects_dangling_duplicate_and_hidden_overriding_definitions():
    original = graph()
    assert validate_spec(candidate_from_graph(original)) == original
    dangling = copy.deepcopy(original)
    dangling["edges"][0]["target"] = "MISSING"
    with pytest.raises(ValueError, match="Dangling"):
        candidate_from_graph(dangling)
    duplicate = copy.deepcopy(original)
    duplicate["states"][1]["id"] = "A"
    with pytest.raises(ValueError, match="unique"):
        candidate_from_graph(duplicate)
    overridden = candidate_from_graph(original)
    overridden["edge_spec"] = canonical({"edges": original["edges"], "states": original["states"]})
    with pytest.raises(ValueError, match="edge_spec"):
        validate_spec(overridden)


def test_router_classifies_every_full_prefix_without_future_or_edges(tmp_path: Path):
    llm = FakeLLM()
    runtime = GraphRuntime(llm, tmp_path)
    run = rollout()
    candidate = candidate_from_graph(graph())
    assigned = asyncio.run(runtime.route_rollout(candidate, run))
    assert len(assigned) == run["history_count"] == 3
    assert [a["state_id"] for a in assigned] == ["A", "B", "B"]
    for step, request in enumerate(llm.calls):
        assert request["messages"][1]["content"] == render_history(run, step) + HISTORY_SUFFIX
        assert "inspect_more" not in request["messages"][0]["content"]
        assert "SECRET_REWARD_DATA" not in canonical(request)
        assert "SECRET_KEY" not in canonical(request)
    assert "OBSERVATION_ONE" not in llm.calls[0]["messages"][1]["content"]
    assert "OBSERVATION_TWO" not in llm.calls[1]["messages"][1]["content"]
    assert "OBSERVATION_TWO" in llm.calls[2]["messages"][1]["content"]


def test_routing_and_judging_use_same_rollout_shard(tmp_path: Path):
    class Pool:
        def __init__(self):
            self.child = FakeLLM()
            self.runtime = self.child.runtime
            self.shards = []

        def shard(self, key):
            self.shards.append(key)
            return self.child

        async def complete_json(self, *args, **kwargs):
            raise AssertionError("A rollout request bypassed shard affinity")

    pool = Pool()
    runtime = GraphRuntime(pool, tmp_path)
    asyncio.run(runtime.evaluate_rollout(candidate_from_graph(graph()), rollout()))
    assert pool.shards == ["r1", "r1"]
    assert len(pool.child.calls) == 4


def test_router_cache_reuses_edge_only_change_but_invalidates_changed_prefix_and_model(
    tmp_path: Path,
):
    llm = FakeLLM()
    runtime = GraphRuntime(llm, tmp_path)
    run = rollout()
    candidate = candidate_from_graph(graph())
    first = asyncio.run(runtime.route_rollout(candidate, run))
    assert asyncio.run(runtime.route_rollout(candidate, run)) == first
    assert len(llm.calls) == 3
    changed_graph = graph()
    changed_graph["edges"][0]["effect"] = "Refined evidence description"
    asyncio.run(runtime.route_rollout(candidate_from_graph(changed_graph), run))
    assert len(llm.calls) == 3
    asyncio.run(runtime.route_rollout(candidate, rollout(suffix="changed")))
    assert len(llm.calls) == 6
    changed_model = FakeLLM(model="fake-v2")
    asyncio.run(GraphRuntime(changed_model, tmp_path).route_rollout(candidate, run))
    assert len(changed_model.calls) == 3
    for path in (tmp_path / "assignments").rglob("*.json"):
        assert "SECRET_KEY" not in path.read_text()


def test_corrupted_router_cache_cannot_return_invented_or_misaligned_assignments(tmp_path: Path):
    runtime = GraphRuntime(FakeLLM(), tmp_path)
    candidate, run = candidate_from_graph(graph()), rollout()
    asyncio.run(runtime.route_rollout(candidate, run))
    path = next((tmp_path / "assignments").rglob("*.json"))
    cached = json.loads(path.read_text())
    cached["assignments"][0]["state_id"] = "INVENTED"
    path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="invalid history order, state IDs"):
        asyncio.run(runtime.route_rollout(candidate, run))


def test_judge_receives_numbered_full_transcript_and_valid_support_sets(tmp_path: Path):
    llm = FakeLLM()
    runtime = GraphRuntime(llm, tmp_path)
    candidate, run = candidate_from_graph(graph()), rollout()
    result = asyncio.run(runtime.evaluate_rollout(candidate, run))
    assert result["metrics"]["history_coverage"] == 1
    assert result["metrics"]["transition_coverage"] == 1
    assert result["score"] == pytest.approx(1 - 0.03 * len(canonical(graph())) / 100_000)
    histories, transitions = supported_sets(result)
    assert histories == {f"r1:h{i:04d}" for i in range(3)}
    assert transitions == {"r1:t0000", "r1:t0001"}
    judge_request = llm.calls[-1]["messages"][1]["content"]
    assert numbered_transcript(run) in judge_request
    assert "GROUP 1: TRANSITION 0, ENDING AT HISTORY 1" in judge_request
    assert "GROUP 2: TRANSITION 1, ENDING AT HISTORY 2" in judge_request
    assert "SECRET_REWARD_DATA" not in judge_request
    assert asyncio.run(runtime.evaluate_rollout(candidate, run)) == result
    assert len(llm.calls) == 4


def test_impossible_edge_and_unsupported_membership_never_get_coverage_credit(tmp_path: Path):
    response = verdict()
    response["membership"][1] = False
    response["transitions"][0]["edge_id"] = "invented"
    runtime = GraphRuntime(FakeLLM([response]), tmp_path)
    result = asyncio.run(runtime.evaluate_rollout(candidate_from_graph(graph()), rollout()))
    assert result["metrics"]["history_coverage"] == pytest.approx(2 / 3)
    assert result["metrics"]["transition_coverage"] == 0
    assert supported_sets(result)[1] == set()


@pytest.mark.parametrize(
    "bad",
    [
        {**verdict(), "membership": [True]},
        {**verdict(), "membership": ["true", True, True]},
        {**verdict(), "coherence": float("nan")},
        {**verdict(), "outgoing_applicability": 2},
    ],
)
def test_malformed_judge_output_is_repaired_with_same_evidence(tmp_path: Path, bad):
    llm = FakeLLM([bad, verdict()])
    runtime = GraphRuntime(llm, tmp_path)
    result = asyncio.run(runtime.evaluate_rollout(candidate_from_graph(graph()), rollout()))
    assert result["metrics"]["history_coverage"] == 1
    assert len(llm.calls) == 5
    assert llm.calls[-2]["messages"][1]["content"] in llm.calls[-1]["messages"][1]["content"]
    assert "STRICT LENGTHS" in llm.calls[-1]["messages"][1]["content"]


def test_incomplete_judge_output_after_repair_raises_and_is_not_cached(tmp_path: Path):
    malformed = {**verdict(), "transitions": []}
    runtime = GraphRuntime(FakeLLM([malformed]), tmp_path)
    with pytest.raises(ValueError, match="incomplete or malformed"):
        asyncio.run(runtime.evaluate_rollout(candidate_from_graph(graph()), rollout()))
    assert not list((tmp_path / "evaluations").rglob("*.json"))


def test_changed_edge_invalidates_evaluation_but_keeps_routing_cache(tmp_path: Path):
    llm, run = FakeLLM(), rollout()
    runtime = GraphRuntime(llm, tmp_path)
    asyncio.run(runtime.evaluate_rollout(candidate_from_graph(graph()), run))
    changed = graph()
    changed["edges"][0]["operation"] = "Inspect with revised scope"
    asyncio.run(runtime.evaluate_rollout(candidate_from_graph(changed), run))
    assert len(llm.calls) == 5


def test_corrupted_evaluation_cache_is_rejected(tmp_path: Path):
    runtime = GraphRuntime(FakeLLM(), tmp_path)
    candidate, run = candidate_from_graph(graph()), rollout()
    asyncio.run(runtime.evaluate_rollout(candidate, run))
    path = next((tmp_path / "evaluations").rglob("*.json"))
    cached = json.loads(path.read_text())
    cached["transitions"][0]["edge_id"] = "invented"
    path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="unsupported transition"):
        asyncio.run(runtime.evaluate_rollout(candidate, run))


def test_reflection_has_actual_action_observation_and_rejects_pareto_evidence(tmp_path: Path):
    run = rollout()
    runtime = GraphRuntime(FakeLLM(), tmp_path)
    with asyncio.Runner() as runner:
        adapter = GraphGEPAAdapter(runtime, runner, [run], [])
        candidate = candidate_from_graph(graph())
        evaluated = adapter.evaluate([run], candidate, capture_traces=True)
        records = adapter.make_reflective_dataset(candidate, evaluated, ["state_spec", "edge_spec"])
        evidence = records["state_spec"][0]["evidence_groups"][0]["group"]
        assert "ACTION_ONE" in evidence and "OBSERVATION_ONE" in evidence
        assert "SECRET_REWARD_DATA" not in canonical(records)
        assert adapter.seen[digest(candidate)] == {"r1"}
        evaluated.outputs[0]["split"] = "pareto"
        with pytest.raises(ValueError, match="Pareto/test"):
            adapter.make_reflective_dataset(candidate, evaluated, ["state_spec"])


def test_historical_retrieval_is_training_only_complete_and_reports_omissions(tmp_path: Path):
    run, long = rollout(), rollout("long", suffix="z" * 310_000)
    runtime = GraphRuntime(FakeLLM(), tmp_path)
    with asyncio.Runner() as runner:
        adapter = GraphGEPAAdapter(runtime, runner, [run, long], [])
        adapter.retention_failures = [
            {
                "losses": [
                    {
                        "rollout_id": "r1",
                        "lost_histories": ["r1:h0000", "r1:h0001", "r1:h0002"],
                        "lost_transitions": ["r1:t0000"],
                    },
                    {"rollout_id": "long", "lost_histories": ["long:h0000"]},
                    {"rollout_id": "pareto-not-in-training", "lost_histories": ["heldout:h0000"]},
                ]
            }
        ]
        retrieved = adapter.historical_counterexamples()
        assert retrieved["available_distinct_failure_prefixes"] == 4
        assert retrieved["omitted_failure_prefixes"] == 2
        assert len(retrieved["examples"]) == 2
        for example in retrieved["examples"]:
            assert example["full_history"] == render_history(run, example["step"])
            assert "SECRET_REWARD_DATA" not in canonical(example)
        assert retrieved["examples"][0]["step"] == 0
        assert retrieved["examples"][1]["step"] == 1


def _supported_result(rid, membership=(True, True, True), transition_support=(True, True)):
    return {
        "rollout_id": rid,
        "assignments": [{"history_id": f"{rid}:h{i:04d}", "state_id": "A"} for i in range(3)],
        "membership_supported": list(membership),
        "transitions": [
            {"step": i, "supported": valid} for i, valid in enumerate(transition_support)
        ],
        "score": 0.9,
        "metrics": {},
    }


class RetentionAdapter:
    def __init__(self, directory, candidates, results, seen):
        self.runtime = SimpleNamespace(directory=directory)
        self.train = {"r1": rollout(), "unseen": rollout("unseen")}
        self.pareto = [rollout(f"p{i}", split="pareto") for i in range(3)]
        self.seen = defaultdict(set, {digest(candidates[i]): set(rids) for i, rids in seen.items()})
        self.second_parents = {}
        self.retention_failures = []
        self.results = results
        self.calls = []

    def evaluate(self, batch, candidate):
        self.calls.append((digest(candidate), [r["id"] for r in batch]))
        outputs = [copy.deepcopy(self.results[(digest(candidate), r["id"])]) for r in batch]
        return EvaluationBatch(outputs=outputs, scores=[o["score"] for o in outputs])


def _retention_case(tmp_path, old_result, new_result):
    parent, child = candidate_from_graph(graph("parent")), candidate_from_graph(graph("child"))
    adapter = RetentionAdapter(
        tmp_path,
        [parent, child],
        {
            (digest(parent), "r1"): old_result,
            (digest(child), "r1"): new_result,
        },
        {0: ["r1"]},
    )
    state = SimpleNamespace(program_candidates=[parent, child])
    proposal = CandidateProposal(
        candidate=child,
        parent_program_ids=[0],
        subsample_scores_before=[0.6],
        subsample_scores_after=[0.7],
    )
    return adapter, state, proposal


def test_retention_preserves_ids_not_counts_and_checks_history_and_transition_sets(tmp_path: Path):
    old = _supported_result("r1", (True, True, False), (False, False))
    new = _supported_result("r1", (False, True, True), (False, False))
    adapter, state, proposal = _retention_case(tmp_path, old, new)
    criterion = CoverageAcceptance(adapter)
    assert not criterion.should_accept(proposal, state)
    assert adapter.retention_failures[0]["losses"][0]["lost_histories"] == ["r1:h0000"]
    old, new = _supported_result("r1"), _supported_result("r1", transition_support=(True, False))
    adapter, state, proposal = _retention_case(tmp_path, old, new)
    assert not CoverageAcceptance(adapter).should_accept(proposal, state)
    assert adapter.retention_failures[0]["losses"][0]["lost_transitions"] == ["r1:t0001"]


def test_retention_accepts_relabeling_and_is_explicitly_seen_ledger_not_census(tmp_path: Path):
    old, new = _supported_result("r1"), _supported_result("r1")
    for assignment in new["assignments"]:
        assignment["state_id"] = "B"
    adapter, state, proposal = _retention_case(tmp_path, old, new)
    assert CoverageAcceptance(adapter).should_accept(proposal, state)
    assert all("unseen" not in ids for _, ids in adapter.calls)


def test_nonimproving_minibatch_rejected_before_retention_evaluation(tmp_path: Path):
    adapter, state, proposal = _retention_case(
        tmp_path, _supported_result("r1"), _supported_result("r1")
    )
    proposal.subsample_scores_after = proposal.subsample_scores_before
    assert not CoverageAcceptance(adapter).should_accept(proposal, state)
    assert not adapter.calls


def test_crossover_preserves_union_of_parent_supported_histories(tmp_path: Path):
    a, b, child = [candidate_from_graph(graph(name)) for name in ("a", "b", "child")]
    outputs = {
        (digest(a), "r1"): _supported_result("r1", (True, True, False), (False, False)),
        (digest(b), "r1"): _supported_result("r1", (False, True, True), (False, False)),
        (digest(child), "r1"): _supported_result("r1", (True, True, False), (False, False)),
    }
    for i in range(3):
        outputs[(digest(child), f"p{i}")] = _supported_result(f"p{i}")
    adapter = RetentionAdapter(tmp_path, [a, b, child], outputs, {0: ["r1"], 1: ["r1"]})
    adapter.second_parents[digest(child)] = 1
    state = SimpleNamespace(
        program_candidates=[a, b, child],
        prog_candidate_val_subscores=[{0: 0.8, 1: 0.5, 2: 0.6}, {0: 0.5, 1: 0.8, 2: 0.6}],
    )
    proposal = CandidateProposal(
        candidate=child,
        parent_program_ids=[0],
        subsample_scores_before=[0.8],
        subsample_scores_after=[0.7],
    )
    criterion = CoverageAcceptance(adapter)
    assert not criterion.should_accept(proposal, state)
    assert adapter.retention_failures[0]["losses"][0]["lost_histories"] == ["r1:h0002"]
    adapter.results[(digest(child), "r1")] = _supported_result("r1")
    assert criterion.should_accept(proposal, state)
    assert proposal.parent_program_ids == [0, 1]
