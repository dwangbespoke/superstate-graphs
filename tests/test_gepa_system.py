import json
from dataclasses import replace

import pytest

from superstate_graphs.gepa_system import (
    Evidence, History, ObservedTransition, Probe, ProbeBank, SuperstateAdapter,
    build_graph, classifier_prompt, probe_result, summarize, synthetic_demo,
    validate_candidate,
)


def candidate(*ids):
    return {"codebook": json.dumps([{"id": i, "description": f"Situation {i}"} for i in ids]),
            "router_instructions": "Use prefix facts only."}


def histories():
    return [History("a", "task-a", "rollout-a", 0, "Available table"),
            History("b", "task-b", "rollout-b", 0, "Available table")]


def test_variable_cardinality_and_complexity_are_explicit():
    assert len(validate_candidate(candidate("x"))) == 1
    assert len(validate_candidate(candidate("x", "y"))) == 2
    with pytest.raises(ValueError):
        validate_candidate(candidate("x", "x"))
    with pytest.raises(ValueError):
        validate_candidate(candidate("x", "y"), max_states=1)


def test_classifier_input_rejects_outcomes_and_does_not_include_identity():
    h = histories()[0]
    with pytest.raises(ValueError):
        History.from_dict({**h.__dict__, "terminal_reward": 1})
    with pytest.raises(ValueError):
        History.from_dict({**h.__dict__, "future_observations": ["answer"]})
    prompt = classifier_prompt(candidate("x"), h)
    assert "task-a" not in prompt and "rollout-a" not in prompt
    assert "Available table" in prompt


def test_llm_proxy_never_becomes_grounded_even_if_marked_audited():
    evidence = Evidence("supported", "llm", "frozen-judge-output:hash", audited=True)
    assert evidence.effective_label("proxy") == "supported"
    assert evidence.effective_label("grounded") == "unknown"
    assert replace(evidence, independent=False).effective_label("proxy") == "unknown"
    assert Evidence("supported", "execution", "run:1").effective_label("grounded") == "unknown"


def test_fixed_denominator_and_contradiction_penalty():
    supported = Evidence("supported", "human", "audit:1", audited=True)
    contradicted = Evidence("contradicted", "execution", "execution:2", audited=True)
    unknown = Evidence("unknown", "llm", "judge:3")
    assignments = {"a": {"superstate_id": "x"}, "b": {"superstate_id": "x"}}
    probes = [Probe("p1", "a", "b", supported, supported),
              Probe("p2", "a", "b", supported, contradicted),
              Probe("p3", "a", "b", unknown, supported)]
    results = [probe_result(p, assignments, "grounded") for p in probes]
    assert summarize(results) == {"valid": 1, "invalid": 1, "unknown": 1,
                                  "not_connected": 0, "denominator": 3, "score": -1}
    assignments["b"]["superstate_id"] = "y"
    assert summarize([probe_result(p, assignments, "grounded") for p in probes])["score"] == 0


def test_graph_allows_alternative_actions_and_only_observed_witnesses():
    hs = [History("a0", "A", "A0", 0, "choice"), History("a1", "A", "A0", 1, "inspected"),
          History("b0", "B", "B0", 0, "choice"), History("b1", "B", "B0", 1, "edited")]
    assignments = {"a0": {"superstate_id": "choice"}, "b0": {"superstate_id": "choice"},
                   "a1": {"superstate_id": "inspected"}, "b1": {"superstate_id": "edited"}}
    transitions = [ObservedTransition("a0", "a1", "inspect", "trace:a"),
                   ObservedTransition("b0", "b1", "edit", "trace:b")]
    graph = build_graph(hs, transitions, assignments)
    assert len(graph["edges"]) == 2
    assert {e["source"] for e in graph["edges"]} == {"choice"}
    with pytest.raises(ValueError, match="same-rollout"):
        build_graph(hs, [ObservedTransition("a0", "b1", "invent", "no-witness")], assignments)


def test_bank_frozen_cache_and_distinct_evidence_reports():
    evidence = Evidence("supported", "llm", "frozen-v1")
    probe = Probe("p", "a", "b", evidence, evidence)
    bank = ProbeBank((probe,))
    adapter = SuperstateAdapter(histories(), [], bank, lambda _: {"superstate_id": "x"})
    report = adapter.evaluate_report(candidate("x"))
    assert report["metrics"]["score"] == 1
    assert report["grounded_metrics"]["valid"] == 0
    assert report["grounded_metrics"]["unknown"] == 1
    adapter.evaluate_report(candidate("x"))
    assert adapter.classifier_calls == 2
    with pytest.raises(ValueError, match="frozen bank"):
        adapter.evaluate_report(candidate("x"), [replace(probe, decision=replace(evidence, label="contradicted"))])
    assert bank.sha256 != ProbeBank((replace(probe, probe_id="different"),)).sha256


def test_synthetic_demo_is_not_grounded_or_trained():
    report = synthetic_demo()
    assert report["synthetic"] is True and report["trained"] is False
    assert report["metrics"]["valid"] == 1
    assert report["grounded_metrics"]["valid"] == 0
    assert report["proxy_metrics"]["valid"] == 0


def test_execution_alignment_preserves_repeated_occurrences_and_time_bounds():
    from superstate_graphs.analyze import align_prefix_events, trajectory_events
    command = json.dumps({"commands": [{"keystrokes": "ls\n"}]})
    steps = [
        {"step_id": 2, "source": "agent", "message": command, "timestamp": 10,
         "observation": {"results": [{"content": "Previous response had parsing errors"}]}},
        {"step_id": 3, "source": "agent", "message": command, "timestamp": 20,
         "observation": {"results": [{"content": "retail.duckdb"}]}},
    ]
    events = trajectory_events({"steps": steps}, "synthetic-trajectory")
    prefix = [{"role": "assistant", "content": command},
              {"role": "user", "content": "Previous response had parsing errors"}]
    first = align_prefix_events(prefix, events, cutoff_epoch=15)
    assert first[0]["step_id"] == 2
    assert first[0]["status"] == "rejected_parse_response"
    extended = prefix + [{"role": "assistant", "content": command},
                         {"role": "user", "content": "retail.duckdb"}]
    both = align_prefix_events(extended, events, cutoff_epoch=25)
    assert [event["step_id"] for event in both] == [2, 3]
    too_early = align_prefix_events(extended, events, cutoff_epoch=15)
    assert too_early[1]["status"] == "unmatched_assistant_occurrence"


def test_one_execution_cannot_certify_multicall_segment(tmp_path):
    from superstate_graphs.analyze import load_corpus
    agent = tmp_path / "job" / "task__trial" / "agent"
    agent.mkdir(parents=True)
    command = json.dumps({"commands": [{"keystrokes": "ls\n"}]})
    unconfirmed = json.dumps({"commands": [{"keystrokes": "echo UNCONFIRMED\n"}]})
    initial = [{"role": "user", "content": "Inspect"}]
    final = initial + [{"role": "assistant", "content": command},
                       {"role": "user", "content": "retail.duckdb"},
                       {"role": "assistant", "content": unconfirmed},
                       {"role": "user", "content": "UNCONFIRMED"}]
    calls = [{"call_index": 0, "messages": initial}, {"call_index": 2, "messages": final}]
    (agent / "policy_calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))
    (agent / "trajectory.json").write_text(json.dumps({"steps": [{
        "step_id": 2, "source": "agent", "message": command,
        "observation": {"results": [{"content": "retail.duckdb"}]}}]}))
    (agent.parent / "result.json").write_text(json.dumps({
        "config": {"task": {"path": "/tasks/taskA"}}, "verifier_result": None}))
    corpus = load_corpus(tmp_path)
    assert len(corpus["segments"]) == 0
    assert len(corpus["provisional_segments"]) == 1
    assert len(corpus["provisional_segments"][0]["unresolved_occurrences"]) == 1


def test_path_export_blocks_known_bad_splice_and_marks_unknown():
    from superstate_graphs.analyze import export_paths, junction_evidence
    hs = [History("a0", "A", "a", 0, "start"), History("a1", "A", "a", 1, "junction"),
          History("b0", "B", "b", 0, "junction"), History("b1", "B", "b", 1, "end")]
    assignments = {"a0": {"superstate_id": "start"}, "a1": {"superstate_id": "junction"},
                   "b0": {"superstate_id": "junction"}, "b1": {"superstate_id": "end"}}
    transitions = [ObservedTransition("a0", "a1", "prepare", "wa"),
                   ObservedTransition("b0", "b1", "aggregate", "wb")]
    graph = build_graph(hs, transitions, assignments)
    positive = Evidence("supported", "llm", "frozen:1")
    negative = Evidence("contradicted", "llm", "frozen:2")
    bank = ProbeBank((Probe("bad", "a1", "b0", positive, negative),))
    stats = [{"superstate_id": "junction", "pooled_binary_variance": .25, "distinct_rollouts": 2}]
    receipts = {h.history_id: {"extraction": {"prerequisites": []}} for h in hs}
    segments = {w: {"observed_messages": []} for w in ("wa", "wb")}
    result = export_paths(graph, hs, stats, receipts, segments, bank)
    assert result["paths"] == []
    assert len(result["known_contradicted_junctions_excluded"]) == 1
    unknown = Evidence("unknown", "llm", "frozen:3")
    unknown_bank = ProbeBank((Probe("unknown", "a1", "b0", positive, unknown),))
    result = export_paths(graph, hs, stats, receipts, segments, unknown_bank)
    assert result["paths"][0]["junction_evidence"]["status"] == "unknown"
    reverse_bank = ProbeBank((Probe("reverse", "b0", "a1", positive, negative),))
    assert junction_evidence("a1", "b0", reverse_bank)["status"] == "unknown"


def test_persisted_candidate_metrics_use_entire_frozen_bank(tmp_path):
    from superstate_graphs.analyze import ParallelAdapter
    positive = Evidence("supported", "llm", "judge:1")
    negative = Evidence("contradicted", "llm", "judge:2")
    probes = [Probe("positive", "a", "b", positive, positive),
              Probe("negative", "a", "b", positive, negative)]
    adapter = ParallelAdapter(histories(), [], ProbeBank(tuple(probes)),
                              lambda _: {"superstate_id": "x"}, report_dir=tmp_path)
    subset = adapter.evaluate_report(candidate("x"), probes[:1])
    assert subset["metrics"]["denominator"] == 1 and subset["metrics"]["score"] == 1
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["report"]["metrics"]["denominator"] == 2
    assert saved["report"]["metrics"]["score"] == -1.5
    assert saved["latest_requested_batch"]["probe_ids"] == ["positive"]
    assert adapter.classifier_calls == 2


def test_path_shortlist_covers_targets_and_task_pairs_before_duplicates():
    from superstate_graphs.analyze import select_diverse_paths
    paths = [{"path_id": str(i), "target_superstate": "S1", "source_task_ids": ["A", "B"]}
             for i in range(6)]
    paths.extend([{"path_id": "different-pair", "target_superstate": "S1", "source_task_ids": ["A", "C"]},
                  {"path_id": "different-target", "target_superstate": "S2", "source_task_ids": ["C", "D"]}])
    selected = select_diverse_paths(paths, 3)
    assert len({p["target_superstate"] for p in selected}) == 2
    assert len({tuple(p["source_task_ids"]) for p in selected}) == 3
