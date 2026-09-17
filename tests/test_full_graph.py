"""Offline full-corpus orchestration, fallback, and transition-accounting tests."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superstate_graphs import full_graph
from superstate_graphs.full_corpus import index_messages, render_history, split_tasks
from superstate_graphs.full_graph import (
    DistinctTaskSampler,
    build_witnessed_edges,
    complete_unassigned,
    evaluate_heldout_comparison,
    exhaustive_assign,
    prepare_optimizer_identity,
)
from superstate_graphs.graph_evolution import GraphRuntime, candidate_from_graph, digest
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


def _identity_inputs(tmp_path):
    router = SimpleNamespace(runtime={"model": "router", "revision": "r1", "api_key": "SECRET"})
    teacher = SimpleNamespace(runtime={"model": "teacher", "revision": "t1"})
    runtime = GraphRuntime(router, tmp_path, teacher_llm=teacher)
    train = [_run("t1", "task1"), _run("t2", "task2")]
    pareto = [{**_run("p1", "pareto-task"), "split": "pareto"}]
    return {
        "runtime": runtime, "candidate": candidate_from_graph(_graph()), "train": train,
        "pareto": pareto, "directory": tmp_path, "minibatch": 2, "random_seed": 17,
    }


def test_optimizer_identity_permits_exact_resume_and_excludes_outcomes_and_secrets(tmp_path):
    args = _identity_inputs(tmp_path)
    first = prepare_optimizer_identity(**args)
    (tmp_path / "gepa").mkdir()
    (tmp_path / "gepa/gepa_state.bin").write_bytes(b"checkpoint")
    args["train"][0]["reward"] = 0
    assert prepare_optimizer_identity(**args) == first
    saved = (tmp_path / "optimizer_identity.json").read_text()
    assert "SECRET" not in saved and "OUTCOME_SECRET" not in saved
    assert "OBSERVATION_ONE" not in saved


@pytest.mark.parametrize("changed", [
    "router", "teacher", "transcript", "boundaries", "train_order", "pareto", "candidate",
    "minibatch", "random_seed",
])
def test_optimizer_identity_rejects_semantically_different_resume(tmp_path, changed):
    args = _identity_inputs(tmp_path)
    prepare_optimizer_identity(**args)
    if changed in {"router", "teacher"}:
        model = args["runtime"].llm if changed == "router" else args["runtime"].teacher_llm
        model.runtime["revision"] = "changed"
    elif changed == "transcript":
        args["train"][0]["transcript"] += "CHANGED"
    elif changed == "boundaries":
        args["train"][0]["history_end_offsets"][0] += 1
    elif changed == "train_order":
        args["train"].reverse()
    elif changed == "pareto":
        args["pareto"][0]["transcript"] += "CHANGED"
    elif changed == "candidate":
        altered = _graph()
        altered["router_instructions"] = "Changed routing"
        args["candidate"] = candidate_from_graph(altered)
    else:
        args[changed] += 1
    previous = (tmp_path / "optimizer_identity.json").read_bytes()
    with pytest.raises(ValueError, match="Optimizer identity mismatch"):
        prepare_optimizer_identity(**args)
    assert (tmp_path / "optimizer_identity.json").read_bytes() == previous


def test_optimizer_identity_rejects_unverified_checkpoint_but_allows_empty_logs(tmp_path):
    args = _identity_inputs(tmp_path)
    (tmp_path / "gepa").mkdir()
    (tmp_path / "gepa/run_log.txt").write_text("")
    (tmp_path / "gepa/gepa_state.bin").write_bytes(b"old checkpoint")
    with pytest.raises(ValueError, match="no optimizer identity"):
        prepare_optimizer_identity(**args)
    (tmp_path / "gepa/gepa_state.bin").unlink()
    assert prepare_optimizer_identity(**args)


def test_completion_cache_identity_tracks_actual_decode_defaults(tmp_path, monkeypatch):
    args = _identity_inputs(tmp_path)
    run = args["train"][0]
    inputs = args["runtime"], args["candidate"], [run], _assignments(run)
    first = full_graph._completion_provenance(*inputs)
    router = first["generation"]["router"]
    assert router["thinking_token_budget"] == 1024
    assert router["temperature"] == 0.6
    original_policy = full_graph.qwen_generation_policy
    monkeypatch.setattr(full_graph, "qwen_generation_policy",
                        lambda **kwargs: {**original_policy(**kwargs), "top_k": 21})
    assert full_graph._completion_provenance(*inputs) != first


class HeldoutRuntime(GraphRuntime):
    def __init__(self, directory, differences=None):
        super().__init__(SimpleNamespace(runtime={"model": "fake", "revision": "v1"}), directory)
        self.calls = []
        self.differences = differences or {}

    async def batch(self, candidate, rollouts, *, judge):
        assert judge and all(row["split"] == "test" for row in rollouts)
        self.calls.append(digest(candidate))
        selected = json.loads(candidate["state_spec"])["router_instructions"] == "selected"
        output = []
        for row in rollouts:
            delta = self.differences.get(row["task_id"], 0.2) if selected else 0
            output.append({
                "rollout_id": row["id"], "task_id": row["task_id"], "split": "test",
                "candidate_hash": digest(candidate),
                "cache_provenance": self._cache_provenance(candidate, row, judge=True),
                "score": 0.4 + delta,
                "metrics": {
                    "history_coverage": 0.4 + delta, "transition_coverage": 0.4 + delta,
                    "coherence": 0.4 + delta, "outgoing_applicability": 0.4 + delta,
                    "complexity": 0.1 - delta / 10,
                },
            })
        return list(reversed(output))  # Matching must use rollout identity, never row position.


def _heldout_candidates():
    seed = candidate_from_graph(_graph())
    selected = _graph()
    selected["router_instructions"] = "selected"
    return seed, candidate_from_graph(selected)


def test_heldout_comparison_matches_by_id_and_bootstraps_paired_original_tasks(tmp_path):
    seed, selected = _heldout_candidates()
    test = [{**_run("a1", "A"), "split": "test"},
            {**_run("a2", "A"), "split": "test"},
            {**_run("b1", "B"), "split": "test"}]
    runtime = HeldoutRuntime(tmp_path, {"A": 0.2, "B": -0.1})
    result = asyncio.run(evaluate_heldout_comparison(
        runtime, seed, selected, test, tmp_path, bootstrap_draws=1000))
    score = result["components"]["score"]
    assert score["mean_difference"] == pytest.approx(0.1)
    assert score["task_balanced_difference"] == pytest.approx(0.05)
    assert score["paired_task_bootstrap_95ci"] == pytest.approx([-0.1, 0.2])
    assert result["per_task"]["A"]["components"]["score"]["difference"] == pytest.approx(0.2)
    assert result["bootstrap"]["resampling_units"] == 2
    assert result["bootstrap"]["equal_rollout_counts_per_task"] is False
    assert result["components"]["complexity"]["favorable_direction"] == "lower"
    assert result["optimization_uses_test_feedback"] is False
    assert result["before_transductive_completion"] is True
    assert result["common_rollout_ids"] == ["a1", "a2", "b1"]
    assert len(runtime.calls) == 2
    assert json.loads((tmp_path / "heldout_comparison.json").read_text()) == result
    assert (tmp_path / "baseline_heldout.json").exists()
    repeated = asyncio.run(evaluate_heldout_comparison(
        runtime, seed, selected, test, tmp_path, bootstrap_draws=1000))
    assert repeated == result and len(runtime.calls) == 2


def test_heldout_seed_selected_equality_reuses_all150_evaluations_and_has_zero_change(tmp_path):
    seed, _ = _heldout_candidates()
    test = [{**_run(f"t{task}-r{index}", f"task{task}"), "split": "test"}
            for task in range(15) for index in range(10)]
    runtime = HeldoutRuntime(tmp_path)
    result = asyncio.run(evaluate_heldout_comparison(
        runtime, seed, seed, test, tmp_path, bootstrap_draws=100))
    assert len(runtime.calls) == 1
    assert result["original_tasks"] == 15 and result["rollouts"] == 150
    assert result["selected_equals_seed"] is True
    assert result["bootstrap"]["equal_rollout_counts_per_task"] is True
    for measurement in result["components"].values():
        assert measurement["mean_difference"] == 0
        assert measurement["paired_task_bootstrap_95ci"] == [0, 0]
    baseline = json.loads((tmp_path / "baseline_heldout.json").read_text())
    assert baseline["reused_selected_evaluation"] is True


@pytest.mark.parametrize("change", ["model", "corpus"])
def test_heldout_comparison_invalidates_both_checkpoints_on_actual_input_change(tmp_path, change):
    seed, selected = _heldout_candidates()
    test = [{**_run(), "split": "test"}]
    runtime = HeldoutRuntime(tmp_path)
    first = asyncio.run(evaluate_heldout_comparison(runtime, seed, selected, test, tmp_path))
    if change == "model":
        runtime.llm.runtime["revision"] = "v2"
    else:
        test[0]["transcript"] += "New observed text"
    changed = asyncio.run(evaluate_heldout_comparison(runtime, seed, selected, test, tmp_path))
    assert len(runtime.calls) == 4
    assert changed["seed_evaluation_identity"] != first["seed_evaluation_identity"]
    assert changed["selected_evaluation_identity"] != first["selected_evaluation_identity"]
    assert len(list((tmp_path / "superseded_test_evaluations").glob("*.json"))) == 2
    assert changed["components"]["score"]["paired_task_bootstrap_95ci"] is None


@pytest.mark.parametrize("corruption", ["duplicate", "task", "candidate", "provenance", "nonfinite"])
def test_heldout_comparison_rejects_mismatched_or_corrupted_paired_rows(tmp_path, corruption):
    seed, selected = _heldout_candidates()
    test = [{**_run("r1", "A"), "split": "test"}, {**_run("r2", "B"), "split": "test"}]
    runtime = HeldoutRuntime(tmp_path)
    asyncio.run(evaluate_heldout_comparison(runtime, seed, selected, test, tmp_path, bootstrap_draws=5))
    path = tmp_path / "baseline_heldout.json"
    payload = json.loads(path.read_text())
    row = payload["rollouts"][0]
    if corruption == "duplicate":
        payload["rollouts"][1] = row
    elif corruption == "task":
        row["task_id"] = "wrong"
    elif corruption == "candidate":
        row["candidate_hash"] = "wrong"
    elif corruption == "provenance":
        row["cache_provenance"] = {}
    else:
        row["score"] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Heldout"):
        asyncio.run(evaluate_heldout_comparison(runtime, seed, selected, test, tmp_path))
    assert len(runtime.calls) == 2


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
            GraphRuntime(llm, tmp_path),
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
    from superstate_graphs.graph_evolution import routing_history

    assert routed == [routing_history(run, step) for step in (1, 2, 2)]
    assert "OUTCOME_SECRET" not in json.dumps(llm.calls)
    assert len(graph["routing_stages"]) == 3
    checkpoint = json.loads((tmp_path / "completion_checkpoint.json").read_text())
    assert "CREDENTIAL_SECRET" not in json.dumps(checkpoint)


def test_completion_resumes_matching_inputs_but_rejects_changed_corpus(tmp_path: Path):
    run, original = _run(), _assignments(_run(), ("A", None, None))
    candidate = candidate_from_graph(_graph())
    llm = CompletionLLM()
    runtime = GraphRuntime(llm, tmp_path)
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
                GraphRuntime(CompletionLLM(malformed=True), tmp_path),
                candidate_from_graph(_graph()),
                [run],
                _assignments(run, ("A", None, None)),
                tmp_path / "malformed",
            )
        )
    graph, assignments = asyncio.run(
        complete_unassigned(
            GraphRuntime(CompletionLLM(never_assign=True), tmp_path),
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
        build_witnessed_edges(GraphRuntime(llm, tmp_path), graph, assignments, runs, tmp_path)
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
