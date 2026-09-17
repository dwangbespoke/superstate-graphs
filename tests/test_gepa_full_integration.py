"""Exercise the installed GEPA optimizer with the production graph adapter.

Only model replies are synthetic. Candidate selection, distinct-task sampling,
reflection, acceptance, history retention, validation, and persistence use the
actual pinned GEPA package and production implementations.
"""

from __future__ import annotations

import asyncio
import json

import gepa
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

from superstate_graphs.full_corpus import index_messages
from superstate_graphs.full_graph import DistinctTaskSampler
from superstate_graphs.graph_evolution import (
    CoverageAcceptance,
    GraphGEPAAdapter,
    GraphRuntime,
    RememberingParetoSelector,
    candidate_from_graph,
    digest,
)


def make_rollout(rid, task_id, split):
    messages = [
        {"role": role, "content": content, "content_json": None, "sequence_number": i}
        for i, (role, content) in enumerate([
            ("user", f"Inspect the source for task {task_id}."),
            ("assistant", "Inspect source columns."),
            ("tool", "OBSERVED_COLUMNS: id, amount"),
            ("assistant", "Check uniqueness."),
            ("tool", "OBSERVED_UNIQUENESS: all ids distinct"),
        ], start=1)
    ]
    return {"id": rid, "task_id": task_id, "split": split, **index_messages(messages)}


def seed_candidate():
    return candidate_from_graph({
        "router_instructions": "revision-0",
        "states": [
            {"id": "A", "name": "Uninspected", "description": "Source not yet inspected.",
             "exclusions": []},
            {"id": "B", "name": "Inspected", "description": "Source columns observed.",
             "exclusions": []},
        ],
        "edges": [
            {"id": "inspect", "source": "A", "target": "B", "operation": "Inspect source",
             "effect": "Source facts observed", "bindings": "Source names may differ"},
            {"id": "check", "source": "B", "target": "B", "operation": "Check uniqueness",
             "effect": "Uniqueness observed", "bindings": "Identifier names may differ"},
        ],
    })


class ModelReplies:
    """Deterministic model outputs; no network or GPU operations."""

    runtime = {"model": "offline-integration-model", "revision": "fixed"}

    def __init__(self):
        self.proposals = 0
        self.reflection_inputs = []

    def shard(self, key):
        return self

    async def complete_json(self, messages, **kwargs):
        properties = kwargs["schema"]["properties"]
        if "state_upserts" in properties:
            self.proposals += 1
            self.reflection_inputs.append(messages)
            return {
                "router_instructions": f"revision-{self.proposals}",
                "state_upserts": [], "remove_state_ids": [],
                "edge_upserts": [], "remove_edge_ids": [],
                "rationale": "Clarify evidence requirements without losing coverage.",
            }
        if "membership" in properties:
            graph = json.loads(messages[1]["content"].split("\nASSIGNMENTS:")[0][len("GRAPH:\n"):])
            revision = int(graph["router_instructions"].split("-")[-1])
            return {
                "membership": [True, True, True],
                "transitions": [
                    {"edge_id": "inspect", "supported": True},
                    {"edge_id": "check", "supported": True},
                ],
                "coherence": 0.1 + revision * 0.4,
                "outgoing_applicability": 1.0,
                "feedback": [{"kind": "membership", "step": 0,
                              "problem": "Membership description can be clearer.",
                              "evidence": "Source not yet inspected.",
                              "suggestion": "Clarify the evidence requirements."}],
                "redundant_states": [],
            }
        assert set(properties) == {"state_id", "evidence"}
        return {
            "state_id": "B" if "OBSERVED_COLUMNS" in messages[1]["content"] else "A",
            "evidence": "Evidence present in the complete supplied prefix.",
        }


def optimize(tmp_path, proposals=2):
    train = [make_rollout(f"train-{task}-{i}", f"train-task-{task}", "train")
             for task in range(2) for i in range(2)]
    pareto = [make_rollout(f"pareto-{i}", f"pareto-task-{i}", "pareto") for i in range(2)]
    model = ModelReplies()
    runtime = GraphRuntime(model, tmp_path / "runtime")
    candidate = seed_candidate()
    with asyncio.Runner() as runner:
        adapter = GraphGEPAAdapter(runtime, runner, train, pareto, seed=17)
        selector = RememberingParetoSelector(adapter)
        acceptance = CoverageAcceptance(adapter)
        result = gepa.optimize(
            seed_candidate=candidate,
            trainset=train,
            valset=pareto,
            adapter=adapter,
            candidate_selection_strategy=selector,
            module_selector="all",
            batch_sampler=DistinctTaskSampler(train, size=2, seed=17),
            reflection_minibatch_size=None,
            skip_perfect_score=False,
            acceptance_criterion=acceptance,
            use_merge=False,
            stop_callbacks=[MaxCandidateProposalsStopper(proposals)],
            max_metric_calls=100,
            run_dir=str(tmp_path / "gepa"),
            seed=17,
            cache_evaluation=True,
            raise_on_exception=True,
        )
    return result, adapter, model, train


def test_pinned_gepa_runs_custom_sampler_selector_and_coverage_acceptance(tmp_path):
    result, adapter, model, train = optimize(tmp_path)
    assert adapter.proposals == model.proposals == 2
    assert result.num_candidates == 3
    assert result.num_val_instances == 2
    assert result.parents == [[None], [0], [1]]
    assert result.val_aggregate_scores[0] < result.val_aggregate_scores[1]
    assert result.val_aggregate_scores[1] < result.val_aggregate_scores[2]
    assert result.best_idx == 2
    assert json.loads(result.best_candidate["state_spec"])["router_instructions"] == "revision-2"
    assert adapter.state is not None
    assert adapter.parent_index == 1
    assert set(adapter.seen[digest(result.best_candidate)]) == {row["id"] for row in train}
    # Verify the same serialization used by the real runner is JSON-safe.
    assert json.loads(json.dumps(result.to_dict()))["best_idx"] == 2
    events = [json.loads(line) for line in (tmp_path / "runtime/events.jsonl").read_text().splitlines()]
    accepted = [event for event in events if event["event"] == "coverage_accepted"]
    assert [event["training_rollouts_checked"] for event in accepted] == [2, 4]
    for proposal_file in sorted((tmp_path / "runtime/proposals").glob("*.json")):
        proposal = json.loads(proposal_file.read_text())
        ids = proposal["training_rollout_ids"]
        tasks = {next(row["task_id"] for row in train if row["id"] == rid) for rid in ids}
        assert len(ids) == len(tasks) == 2
    assert all("pareto-task-" not in json.dumps(call) for call in model.reflection_inputs)
