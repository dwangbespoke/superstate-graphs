"""Export frozen, supported shared-start branch compositions without inference.

These are not sequential graph paths: two observed outgoing segments start at
different concrete initial histories assigned to the same superstate. Their
terminal goals are new compilation proposals, not witnessed successful endings.
The ordinary selected_paths.json and all original judgments remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from superstate_graphs.gepa_system import fingerprint
from superstate_graphs.task_constructor import normalize_path

ROOT = Path(__file__).resolve().parents[1]


def export(analysis: Path, probe_ids: list[str] | None = None, *, limit: int = 3) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    if probe_ids is not None and (not probe_ids or len(set(probe_ids)) != len(probe_ids)):
        raise ValueError("Explicit probe IDs must be nonempty and unique")

    def read(name: str):
        return json.loads((analysis / name).read_text())

    graph = read("selected_graph.json")
    histories = {item["history_id"]: item for item in read("histories.json")}
    receipts = read("extraction_provenance.json")
    candidate = read("selected_candidate.json")
    definitions = {item["id"]: item for item in json.loads(candidate["codebook"])}
    statistics = {item["superstate_id"]: item for item in read("pooled_outcomes.json")}
    outgoing = {}
    for edge in graph["edges"]:
        for witness in edge["witnesses"]:
            outgoing.setdefault(witness["source_id"], []).append((edge, witness))

    def eligibility(record: dict) -> tuple[bool, str]:
        judgment = record["judgment"]
        if any(judgment[key]["label"] != "supported" for key in ("decision", "local_transfer")):
            return False, "Frozen decision and local-transfer support both required"
        ids = (record["left_id"], record["right_id"])
        if any(identity not in histories or histories[identity]["step"] != 0 for identity in ids):
            return False, "Both histories must be recorded initial histories"
        if histories[ids[0]]["task_id"] == histories[ids[1]]["task_id"]:
            return False, "Distinct source tasks required"
        if any(len(outgoing.get(identity, [])) != 1 for identity in ids):
            return False, "One observed first segment per initial history required"
        if len({outgoing[identity][0][0]["source"] for identity in ids}) != 1:
            return False, "Both histories must be assigned to the same superstate"
        for identity in ids:
            _, witness = outgoing[identity][0]
            witness_path = Path(witness["witness_ref"])
            if not witness_path.is_absolute():
                witness_path = ROOT / witness_path
            if not witness_path.is_file():
                return False, "Observed segment file missing"
            segment = json.loads(witness_path.read_text())
            if segment["source_id"] != identity or segment["target_id"] != witness["target_id"]:
                return False, "Graph and segment identities differ"
        return True, "Eligible"

    frozen = {}
    for filename in sorted((analysis / "frozen_judgments").glob("*.json")):
        record = json.loads(filename.read_text())
        frozen[record["probe_id"]] = record
    eligible = {probe_id: record for probe_id, record in frozen.items() if eligibility(record)[0]}
    if probe_ids is None:
        selected, seen_combinations = [], set()
        for probe_id in sorted(eligible, key=lambda key: (not key.startswith("train-"), key)):
            record = eligible[probe_id]
            combination = tuple(sorted(histories[record[key]]["task_id"] for key in ("left_id", "right_id")))
            if combination in seen_combinations:
                continue
            selected.append(probe_id)
            seen_combinations.add(combination)
            if len(selected) >= limit:
                break
        selection_mode = "deterministic_unique_source_task_combinations_train_first"
    else:
        selected = probe_ids
        for probe_id in selected:
            if probe_id not in frozen:
                raise ValueError(f"Unknown explicit probe ID: {probe_id}")
            valid, reason = eligibility(frozen[probe_id])
            if not valid:
                raise ValueError(f"Ineligible explicit probe {probe_id}: {reason}")
        selection_mode = "explicit_probe_id_override"

    paths = []
    for probe_id in selected:
        judgment_file = analysis / "frozen_judgments" / f"{probe_id}.json"
        raw_judgment = judgment_file.read_bytes()
        record = json.loads(raw_judgment)
        judgment = record["judgment"]
        if any(judgment[key]["label"] != "supported" for key in ("decision", "local_transfer")):
            raise ValueError(f"{probe_id} does not have the required frozen local support")
        ids = (record["left_id"], record["right_id"])
        if any(histories[identity]["step"] != 0 for identity in ids):
            raise ValueError("Shared-start export requires actual initial histories")
        if any(len(outgoing.get(identity, [])) != 1 for identity in ids):
            raise ValueError("Expected one recorded first segment per initial history")
        transitions, superstates = [], set()
        for branch, identity in zip(("A", "B"), ids, strict=True):
            edge, witness = outgoing[identity][0]
            witness_path = Path(witness["witness_ref"])
            if not witness_path.is_absolute():
                witness_path = ROOT / witness_path
            segment = json.loads(witness_path.read_text())
            if segment["source_id"] != identity or segment["target_id"] != witness["target_id"]:
                raise ValueError("Graph witness does not identify its actual recorded segment")
            superstates.add(edge["source"])
            transitions.append({
                "transition_id": f"branch_{branch}",
                "branch": branch,
                "source_task_id": histories[identity]["task_id"],
                "source_history_id": identity,
                "target_history_id": witness["target_id"],
                "source_superstate": edge["source"],
                "target_superstate": edge["target"],
                "operation": edge["operation"],
                "prerequisites": receipts[identity]["extraction"].get("prerequisites", []),
                "effects": {
                    "observed_messages": segment["observed_messages"],
                    "execution_witnesses": segment.get("execution_witnesses", []),
                    "transition_evidence": segment.get("transition_evidence"),
                },
                "witness": str(witness_path),
                "role_bindings": witness.get("source_bindings", {}),
            })
        if len(superstates) != 1:
            raise ValueError("The two initial histories are not assigned to the same superstate")
        target = superstates.pop()
        source_tasks = sorted({transition["source_task_id"] for transition in transitions})
        if len(source_tasks) != 2:
            raise ValueError("Shared-start composition requires distinct source tasks")
        original_record = {**record, "source": str(judgment_file),
                           "source_file_sha256": hashlib.sha256(raw_judgment).hexdigest()}
        path = {
            "path_id": "shared-start-" + probe_id,
            "construction_topology": "shared_start_branches",
            "topology_explanation": "Two observed branches start at separate concrete histories sharing an unresolved initial decision; they are not sequentially connected. The constructor must author a coherent new terminal goal.",
            "composition_kind": "shared_start_branches_not_sequential_traversal",
            "branch_semantics": "Both witnesses leave the same abstract initial decision. Neither branch is claimed to start at the other branch's destination.",
            "target_superstate": target,
            "target_definition": definitions[target],
            "definition_candidate_sha256": fingerprint(candidate),
            "source_task_ids": source_tasks,
            "selection_reason": "Bounded fallback using an already supported initial decision and first-operation transfer; not a high-variance selection.",
            "target_statistics": statistics[target],
            "transitions": transitions,
            "junction_prefix_A": receipts[ids[0]]["extraction"],
            "junction_prefix_B": receipts[ids[1]]["extraction"],
            "junction_evidence": {
                "left_history_id": ids[0], "right_history_id": ids[1],
                "decision_status": judgment["decision"]["label"],
                "local_transfer_status": judgment["local_transfer"]["label"],
                "literal_replay_status": judgment["full_segment_transfer"]["label"],
                "compilation_status": "unassessed",
                "exact_probe_ids": [probe_id], "tier": "proxy",
                "evidence_scope": "Same initial local decision and first witnessed operation only; full replay judgment remains separate.",
            },
            "original_replay_judgments": [original_record],
            "compilation_obligations": "Preserve the unresolved backend-discovery decision, explicitly bind both branch patterns into a new goal, and resolve every original full-replay conflict. Do not claim sequential transfer or an observed terminal success.",
            "status": "proposed_shared_start_composition_requires_concrete_instantiation",
            "terminal_endpoint": "Constructor-authored new goal and verifier; neither observed segment is a successful terminal trajectory.",
            "splice_obligation": "No sequential splice is asserted. Compile two outgoing branch patterns from the supported shared initial decision into one coherent new task.",
            "development_split": "validation" if probe_id.startswith("validation-") else "train",
            "evaluation_scope": "Development artifact construction; selecting this pair is not held-out task-generation evaluation.",
        }
        paths.append(normalize_path(path))
    return {
        "path_export_protocol": "shared_start_branches_v1",
        "any_positive_variance": False,
        "paths": paths,
        "selection_mode": selection_mode,
        "selection_policy": "Require frozen decision and local-transfer support, actual initial histories in the same superstate, and real outgoing witnesses. Default: greedily select distinct source-task combinations in train-first, then development-validation, lexical probe-ID order. Explicit probe IDs override this order. Original selected paths and labels remain unchanged.",
        "eligible_probe_count": len(eligible),
        "selected_probe_ids": selected,
        "limitations": [
            "These branch compositions do not establish sequential cross-task graph traversal.",
            "Only backend discovery has frozen local support in this subset.",
            "The full recorded segments remain contradicted or unknown under the original task goals.",
            "Task construction authors a new terminal goal and must demonstrate executability separately.",
            "No positive pooled outcome variance or learner benefit has been observed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "results/analysis_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=3,
                        help="Maximum source-task combinations for deterministic default selection")
    parser.add_argument("--probe-id", action="append",
                        help="Explicit eligible frozen probe; repeat to override default selection")
    args = parser.parse_args()
    output = args.output or args.analysis / "shared_start_paths.json"
    result = export(args.analysis, args.probe_id, limit=args.limit)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "compositions": len(result["paths"]),
                      "protocol": result["path_export_protocol"]}))


if __name__ == "__main__":
    main()
