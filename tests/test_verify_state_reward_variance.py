"""Offline variance verification detects changed estimates and interval receipts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from superstate_graphs import graph_analysis


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_state_reward_variance.py"
SPEC = importlib.util.spec_from_file_location("variance_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def write(root, name, value):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(value))


def fixture(tmp_path):
    run, corpus = tmp_path / "run", tmp_path / "corpus"
    histories, assignments, rewards = [], {}, {}
    # Unequal task sizes and visit frequencies make the three estimands distinct.
    for rid, task, split, reward, visits, manual in (
        ("a0", "A", "train", 0, 2, False),
        ("a1", "A", "train", 1, 1, False),
        ("a2", "A", "train", 1, 3, False),
        ("b0", "B", "test", 0, 2, True),
        ("c0", "C", "pareto", 0.25, 1, False),
    ):
        provenance = {"kind": "manual_trace_review" if manual else "original_verifier"}
        rewards[rid] = {"task_id": task, "reward": reward, "reward_provenance": provenance}
        for step in range(visits):
            hid = f"{rid}:h{step:04d}"
            row = {
                "history_id": hid,
                "rollout_id": rid,
                "task_id": task,
                "split": split,
                "step": step,
                "reward": reward,
                "reward_provenance": provenance,
                "private_prefix": "RAW_PRIVATE_HISTORY",
            }
            histories.append(row)
            assignments[hid] = {
                key: row[key] for key in ("history_id", "rollout_id", "task_id", "split", "step")
            }
            assignments[hid].update(
                state_id="M" if rid == "b0" and step == 1 else "S",
                evidence="RAW_PRIVATE_ASSIGNMENT",
            )
    corpus.mkdir()
    (corpus / "histories.jsonl").write_text("".join(json.dumps(row) + "\n" for row in histories))
    write(
        corpus,
        "manifest.json",
        {
            "statistics": {"histories": len(histories)},
            "artifact_sha256": {
                "histories.jsonl": verifier.sha((corpus / "histories.jsonl").read_bytes())
            },
        },
    )
    write(run, "assignments_all.json", assignments)
    full = graph_analysis.analyze_state_rewards(histories, assignments, rewards, seed=20260917)
    write(run, "state_reward_variance.json", full)
    by_split = {}
    for split in ("train", "pareto", "test"):
        subset = [h for h in histories if h["split"] == split]
        by_split[split] = graph_analysis.analyze_state_rewards(
            subset,
            {h["history_id"]: assignments[h["history_id"]] for h in subset},
            rewards,
            bootstrap_draws=500,
            seed=20260917,
        )
    write(run, "state_reward_variance_by_split.json", by_split)
    return run, corpus


def test_verifies_all_weightings_manual_sensitivity_and_intervals_without_generator(
    tmp_path, monkeypatch
):
    run, corpus = fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Verification must not recompute using production analysis")

    monkeypatch.setattr(graph_analysis, "analyze_state_rewards", forbidden)
    receipt = verifier.verify_variance(run, corpus)
    assert receipt["status"] == "passed"
    assert receipt["scopes"]["all"] == {
        "histories": 9,
        "occupied_states": 2,
        "moment_and_membership_validation": True,
    }
    assert receipt["task_cluster_intervals_recomputed"] == 6
    assert receipt["intervals_without_sufficient_task_clusters"] == 30
    assert receipt["max_absolute_error"] < 1e-10
    assert receipt["verification_script_sha256"] == verifier.sha(SCRIPT.read_bytes())
    assert "RAW_" not in json.dumps(receipt)
    assert receipt["model_calls"] == 0


@pytest.mark.parametrize("corruption", ["moment", "interval", "census"])
def test_rejects_changed_moments_intervals_or_state_census(tmp_path, corruption):
    run, corpus = fixture(tmp_path)
    name = "state_reward_variance.json"
    value = json.loads((run / name).read_text())
    state = next(row for row in value["states"] if row["state_id"] == "S")
    if corruption == "moment":
        state["task_balanced"]["population_variance"] += 0.01
    elif corruption == "interval":
        state["trajectory_deduplicated"]["task_cluster_bootstrap"]["mean_interval"][0] += 0.01
    else:
        value["states"] = [state]
    write(run, name, value)
    with pytest.raises(ValueError):
        verifier.verify_variance(run, corpus)
