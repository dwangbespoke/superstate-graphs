import json
from copy import deepcopy

import pytest

from superstate_graphs.graph_analysis import (
    aggregate_independent_audits,
    analyze_state_rewards,
    audit_messages,
    corpus_history_metadata,
    plan_independent_audits,
    select_grounded_paths,
    summarize_graph_structure,
    task_draft_messages,
    task_feasibility_messages,
    validate_audit_result,
    validate_task_draft,
)


def h(hid, task, rollout, step=0, prefix=None):
    return {"history_id": hid, "task_id": task, "rollout_id": rollout, "step": step,
            "prefix": prefix or f"Exact prefix {hid}"}


def test_weighting_estimands_are_distinct_and_revisits_do_not_duplicate_rollouts():
    histories = [h("a0", "A", "a"), h("a1", "A", "a", 1), h("a2", "A", "a", 2),
                 h("b0", "A", "b"), h("c0", "B", "c")]
    assignments = {item["history_id"]: {"state_id": "s"} for item in histories}
    result = analyze_state_rewards(histories, assignments, {"a": 1, "b": 0, "c": 0},
                                   bootstrap_draws=0)["states"][0]
    assert result["history_weighted"]["mean"] == pytest.approx(.6)
    assert result["history_weighted"]["population_variance"] == pytest.approx(.24)
    assert result["trajectory_deduplicated"]["mean"] == pytest.approx(1 / 3)
    assert result["task_balanced"]["mean"] == pytest.approx(.25)
    assert result["task_balanced"]["population_variance"] == pytest.approx(.1875)
    decomposition = result["task_balanced_decomposition"]
    assert decomposition["mean_within_task_population_variance"] == pytest.approx(.125)
    assert decomposition["between_task_mean_variance"] == pytest.approx(.0625)
    assert result["distinct_visiting_rollouts"] == 3


def test_manual_zero_sensitivity_and_missing_reward_are_explicit():
    histories = [h("h1", "A", "a"), h("h2", "B", "b"), h("h3", "C", "c")]
    rewards = {"a": {"reward": 1, "reward_provenance": {"kind": "original_verifier"}},
               "b": {"reward": 0, "reward_provenance": {"kind": "manual_trace_review"}},
               "c": {"reward": None}}
    state = analyze_state_rewards(histories, {item["history_id"]: "s" for item in histories},
                                  rewards, bootstrap_draws=0)["states"][0]
    assert state["trajectory_deduplicated"]["mean"] == .5
    assert state["manual_reward_rollouts"] == ["b"]
    assert state["missing_reward_rollouts"] == ["c"]
    sensitivity = state["sensitivity_excluding_manual_rewards"]
    assert sensitivity["trajectory_deduplicated"]["mean"] == 1
    assert sensitivity["trajectory_deduplicated"]["population_variance"] == 0
    assert sensitivity["excluded_manual_reward_rollouts"] == ["b"]
    assert sensitivity["rewarded_tasks"] == 1


def test_cluster_bootstrap_is_deterministic_and_does_not_claim_iid_prefix_intervals():
    histories = [h("a", "A", "a"), h("b", "B", "b")]
    kwargs = {"histories": histories, "assignments": {"a": "s", "b": "s"},
              "rewards": {"a": 0, "b": 1}, "bootstrap_draws": 300, "seed": 11}
    result = analyze_state_rewards(**kwargs)
    assert result == analyze_state_rewards(**kwargs)
    interval = result["states"][0]["task_balanced"]["task_cluster_bootstrap"]
    assert interval["mean_interval"] == [0, 1]
    assert interval["population_variance_interval"] == [0, .25]
    assert interval["task_clusters"] == 2


def test_one_task_has_no_task_cluster_interval_even_with_many_histories():
    histories = [h(str(i), "A", "a", i) for i in range(20)]
    state = analyze_state_rewards(histories, {str(i): "s" for i in range(20)}, {"a": 1},
                                  bootstrap_draws=10)["states"][0]
    assert state["history_weighted"]["task_cluster_bootstrap"]["status"] == "insufficient_task_clusters"
    assert state["trajectory_deduplicated"]["bessel_corrected_descriptive_variance"] is None


def test_missing_assignments_and_explicit_abstentions_are_not_counted_as_states():
    result = analyze_state_rewards([h("a", "A", "a"), h("b", "B", "b")],
                                    {"a": {"node_id": None}}, {}, bootstrap_draws=0)
    assert result["missing_assignment_count"] == result["unassigned_count"] == 1
    assert result["states"] == []


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "not a reward"])
def test_invalid_rewards_are_not_silently_coerced(invalid):
    with pytest.raises(ValueError, match="reward"):
        analyze_state_rewards([h("a", "A", "a")], {"a": "s"}, {"a": invalid})


def test_history_and_reward_identity_errors_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_state_rewards([h("a", "A", "a")] * 2, {"a": "s"}, {})
    with pytest.raises(ValueError, match="absent"):
        analyze_state_rewards([h("a", "A", "a")], {"a": "s", "extra": "s"}, {})
    with pytest.raises(ValueError, match="disagree"):
        analyze_state_rewards([h("a", "A", "a")], {"a": "s"},
                               {"a": {"reward": 1, "task_id": "OTHER"}})


def fixture():
    histories = [h("a0", "A", "a", 0), h("a1", "A", "a", 1),
                 h("b0", "B", "b", 0), h("b1", "B", "b", 1),
                 h("c0", "C", "c", 0), h("c1", "C", "c", 1)]
    assignments = {"a0": "s0", "a1": "s1", "b0": "s1", "b1": "s2",
                   "c0": "s0", "c1": "s0"}
    transitions = [{"source_id": "a0", "target_id": "a1", "operation": "inspect"},
                   {"source_id": "b0", "target_id": "b1", "operation": "join"},
                   {"source_id": "c0", "target_id": "c1", "operation": "retry"}]
    graph = {"nodes": [{"id": "s0", "description": "Inspect unknown join key"},
                       {"id": "s1", "description": "Validate proposed join key"},
                       {"id": "s2", "description": "Check aggregate output"}],
             "edges": [{"id": "e0", "source": "s0", "target": "s1", "operation": "inspect",
                        "witnesses": [transitions[0]], "traversable": True},
                       {"id": "e1", "source": "s1", "target": "s2", "operation": "join",
                        "witnesses": [transitions[1]], "traversable": True}]}
    return graph, histories, assignments, transitions


def test_audit_plan_includes_cross_task_pairs_and_untraversed_source_edges():
    graph, histories, assignments, transitions = fixture()
    jobs = plan_independent_audits(graph, histories, assignments, transitions,
                                   sources_per_edge=1, seed=1)
    assert any(job["kind"] == "within_state_coherence" and job["cross_task"] for job in jobs)
    edge = next(job for job in jobs if job.get("edge_id") == "e0")
    assert edge["history_ids"][0] in {"c0", "c1"}
    assert edge["source_observed_to_target"] is False
    assert any(job["kind"] == "boundary_and_redundancy" for job in jobs)
    assert jobs == plan_independent_audits(graph, histories, assignments, transitions,
                                          sources_per_edge=1, seed=1)


def test_structural_summary_keeps_cross_task_support_and_missing_edges_separate():
    graph, histories, assignments, transitions = fixture()
    report = summarize_graph_structure(graph, histories, assignments, transitions)
    assert report["histories"] == report["assigned_histories"] == 6
    assert report["nodes_spanning_multiple_tasks"] == 2
    assert report["transition_structure"]["total_transitions"] == 3
    assert report["transition_structure"]["existing_endpoint_pair"] == 2
    assert report["transition_structure"]["missing_endpoint_pair"] == 1
    assert "not established" in report["limitation"]


def test_structural_summary_does_not_call_a_declared_edge_certified():
    graph, histories, assignments, transitions = fixture()
    graph["edges"][0]["traversable"] = False
    report = summarize_graph_structure(graph, histories, assignments, transitions)
    assert report["transition_structure"]["existing_endpoint_pair"] == 2
    assert report["transition_structure"]["reported_traversable_endpoint_pairs"] == 1


def test_audit_prompts_preserve_full_prefixes_and_exclude_outcome_side_tables():
    graph, histories, assignments, transitions = fixture()
    for history in histories:
        history["reward"] = "SECRET_TERMINAL_OUTCOME"
    graph["nodes"][0]["reward_statistics"] = "SECRET_NODE_STATISTICS"
    graph["edges"][0]["audit"] = {"reward": "SECRET_PRIOR_JUDGMENT"}
    index = {item["history_id"]: item for item in histories}
    def provider(hid):
        return index[hid]["prefix"] + " end-of-complete-prefix"
    jobs = plan_independent_audits(graph, histories, assignments, transitions)
    for job in jobs:
        messages = audit_messages(job, graph, index, provider)
        text = json.dumps(messages)
        assert "SECRET" not in text
        for hid in job["history_ids"]:
            assert provider(hid) in text


def test_ungrounded_judge_verdict_is_downgraded_and_cannot_certify_graph():
    job = {"audit_id": "a", "kind": "within_state_coherence", "history_ids": ["h1", "h2"]}
    def provider(hid):
        return "Observed " + hid
    response = {"verdict": "supported", "evidence": [{"history_id": "h1", "quote": "invented"}]}
    result = validate_audit_result(job, response, provider)
    assert result["verdict"] == "unknown"
    assert result["original_verdict"] == "supported"
    assert not result["universal_contract_certified"]
    response["evidence"] = [{"history_id": hid, "quote": provider(hid)} for hid in ("h1", "h2")]
    result = validate_audit_result(job, response, provider)
    assert result["verdict"] == "supported"
    assert result["evidence_tier"] == "independent_llm_proxy"


def test_citing_target_alone_does_not_establish_source_applicability():
    job = {"audit_id": "a", "kind": "edge_source_applicability",
           "history_ids": ["source"], "target_history_ids": ["target"]}
    result = validate_audit_result(job, {"verdict": "supported", "evidence": [
        {"history_id": "target", "quote": "target"}]}, lambda hid: hid)
    assert result["verdict"] == "unknown"


def test_audit_aggregation_reports_missing_checks_and_observed_vs_proxy_evidence():
    graph, histories, assignments, transitions = fixture()
    jobs = plan_independent_audits(graph, histories, assignments, transitions)
    edge_job = next(job for job in jobs if job.get("edge_id") == "e0")
    result = {"audit_id": edge_job["audit_id"], "verdict": "contradicted"}
    report = aggregate_independent_audits(jobs, [result], graph)
    assert report["completed_checks"] == 1
    edge = report["edges"][0]
    assert edge["observed_witness_count"] == 1
    assert edge["has_independent_counterexample"]
    assert edge["universal_contract_certified"] is False
    assert report["by_kind"]["edge_source_applicability"]["not_run"] > 0


def test_cross_task_path_witnesses_are_grounded_and_never_marked_executed():
    graph, histories, assignments, _ = fixture()
    paths = select_grounded_paths(graph, histories, assignments, count=8)
    assert len(paths) == 1
    path = paths[0]
    assert path["state_ids"] == ["s0", "s1", "s2"]
    assert path["source_task_ids"] == ["A", "B"]
    assert path["individual_edges_observed"]
    assert not path["composed_path_executed"]
    assert path["transitions"][0]["source_history_id"] == "a0"


def test_stale_witness_assignments_and_disabled_edges_cannot_ground_paths():
    graph, histories, assignments, _ = fixture()
    changed = {**assignments, "a0": "s2"}
    assert select_grounded_paths(graph, histories, changed) == []
    graph["edges"][0]["traversable"] = False
    assert select_grounded_paths(graph, histories, assignments) == []


def test_same_task_witnesses_do_not_claim_cross_task_composition():
    graph, histories, assignments, _ = fixture()
    for item in histories:
        item["task_id"] = "ONE_TASK"
    assert select_grounded_paths(graph, histories, assignments) == []


def test_malformed_cross_trajectory_witness_is_rejected():
    graph, histories, assignments, _ = fixture()
    graph["edges"][0]["witnesses"][0]["target_id"] = "b0"
    with pytest.raises(ValueError, match="same-trajectory"):
        select_grounded_paths(graph, histories, assignments)


def test_task_drafts_keep_path_provenance_and_have_explicit_unexecuted_status():
    graph, histories, assignments, _ = fixture()
    path = select_grounded_paths(graph, histories, assignments)[0]
    draft = {"status": "draft", "learner_instruction": "Find the join relationship and aggregate.",
             "stages": [{key: transition[key] for key in (
                 "transition_id", "source_history_id", "target_history_id")}
                        for transition in path["transitions"]]}
    validated = validate_task_draft(path, draft)
    assert not validated["executed"]
    assert not validated["benchmark_validated"]
    modified = deepcopy(draft)
    modified["stages"][0]["source_history_id"] = "invented"
    with pytest.raises(ValueError, match="provenance"):
        validate_task_draft(path, modified)


def test_task_generator_and_reviewer_do_not_receive_outcome_metadata():
    graph, histories, assignments, _ = fixture()
    path = select_grounded_paths(graph, histories, assignments)[0]
    path["variance"] = "SECRET_OUTCOME"
    path["nodes"][0]["reward"] = "SECRET_OUTCOME"
    path["transitions"][0]["reward"] = "SECRET_OUTCOME"
    messages = task_draft_messages(path, lambda hid: "complete prefix " + hid)
    assert "SECRET_OUTCOME" not in json.dumps(messages)
    review = task_feasibility_messages(path, {"status": "draft", "reward": "SECRET_OUTCOME"},
                                      lambda hid: "complete prefix " + hid)
    assert "SECRET_OUTCOME" not in json.dumps(review)


def test_corpus_metadata_uses_every_prefix_without_outcome_fields():
    records = corpus_history_metadata([{"rollout_id": "r", "task_id": "t", "history_count": 3,
                                         "reward": 1, "split": "test"}])
    assert [item["history_id"] for item in records] == ["r:h0000", "r:h0001", "r:h0002"]
    assert all("reward" not in item for item in records)


def test_task_prompt_losslessly_packs_prefix_and_transition_extension():
    path = {"path_id": "p", "nodes": [{"id": "a"}, {"id": "b"}],
            "transitions": [{"transition_id": "e", "source_history_id": "h0",
                             "target_history_id": "h1", "operation": "inspect"}]}
    prefixes = {"h0": "FULL SOURCE\n", "h1": "FULL SOURCE\nACTION AND OBSERVATION\n"}
    messages = task_draft_messages(path, prefixes.__getitem__)
    witness = json.loads(messages[1]["content"])["witnesses"][0]
    assert witness["full_source_prefix"] == prefixes["h0"]
    assert witness["full_source_prefix"] + witness["exact_action_observation_extension"] == prefixes["h1"]
    assert "full_target_prefix" not in witness


def test_final_analysis_integration_is_restartable_and_separates_drafts_from_execution(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from superstate_graphs.full_graph import analyze_and_construct

    class FakeLLM:
        runtime = {"model": "synthetic-test-model", "api_key": "DO_NOT_WRITE_THIS_SECRET"}
        calls = 0

        def shard(self, key):
            return self

        async def complete_json(self, messages, **kwargs):
            self.calls += 1
            system = messages[0]["content"]
            data = json.loads(messages[1]["content"])
            if system.startswith("You independently audit"):
                prefixes = {h["history_id"]: h["prefix"] for h in data["histories"]}
                return {"verdict": "supported", "reason": "Synthetic integration response",
                        "evidence": [{"history_id": hid, "quote": prefixes[hid]}
                                     for hid in data["source_history_ids"]]}
            if system.startswith("Create a concrete task"):
                return {"status": "draft", "title": "Synthetic integration task",
                        "learner_instruction": "Inspect the relationship and calculate a result.",
                        "stages": [{key: t[key] for key in (
                            "transition_id", "source_history_id", "target_history_id")}
                                   for t in data["transitions"]]}
            if system.startswith("Independently review"):
                return {"verdict": "plausible", "findings": [], "missing_runtime_checks": ["execution"]}
            raise AssertionError("Unexpected model request")

    rollouts = []
    for rid, tid, split, reward in [("ra", "A", "train", 1), ("rb", "B", "test", 0)]:
        prefix, extension = f"Query {rid}\n", "Observed tool result\n"
        rollouts.append({"id": rid, "rollout_id": rid, "task_id": tid, "split": split,
                         "transcript": prefix + extension, "history_end_offsets": [len(prefix),
                                                                                    len(prefix + extension)],
                         "history_count": 2, "steps": [{"step": 0, "start_char": len(prefix),
                                                        "end_char": len(prefix + extension)}],
                         "reward": reward, "reward_provenance": {"kind": "original_verifier"}})
    assignments = {"ra:h0000": {"state_id": "s0"}, "ra:h0001": {"state_id": "s1"},
                   "rb:h0000": {"state_id": "s1"}, "rb:h0001": {"state_id": "s2"}}
    graph = {"nodes": [{"id": sid, "description": "Situation " + sid} for sid in ("s0", "s1", "s2")],
             "edges": [{"id": "e0", "source": "s0", "target": "s1", "operation": "inspect",
                        "witnesses": [{"source_id": "ra:h0000", "target_id": "ra:h0001"}],
                        "traversable": False, "proposal": {"universal_source_plausible": True}},
                       {"id": "e1", "source": "s1", "target": "s2", "operation": "calculate",
                        "witnesses": [{"source_id": "rb:h0000", "target_id": "rb:h0001"}],
                        "traversable": False, "proposal": {"universal_source_plausible": True}}]}
    llm = FakeLLM()
    runtime = SimpleNamespace(llm=llm)
    result = asyncio.run(analyze_and_construct(runtime, graph, assignments, rollouts, tmp_path))
    first_calls = llm.calls
    assert result["all_histories_analyzed"] == 4
    assert result["tasks"]["draft_count"] == 1
    assert result["executed_task_count"] == 0
    assert all(e["traversable"] for e in graph["edges"])
    assert all(not e["audit"]["universal_contract_certified"] for e in graph["edges"])
    assert "DO_NOT_WRITE_THIS_SECRET" not in (tmp_path / "independent_judge_provenance.json").read_text()
    assert (tmp_path / "TASK_DRAFTS.md").exists()
    asyncio.run(analyze_and_construct(runtime, graph, assignments, rollouts, tmp_path))
    assert llm.calls == first_calls
