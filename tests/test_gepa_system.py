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
