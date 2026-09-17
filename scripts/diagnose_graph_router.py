"""Full-prefix regression diagnostics for already observed grounding failures.

These hand-checked training examples diagnose the serving/prompt configuration;
they are not estimates of clustering quality or held-out benchmark scores.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from superstate_graphs.full_corpus import render_history
from superstate_graphs.graph_evolution import ROUTER_SYSTEM, routing_history
from superstate_graphs.graph_llm import GraphLLM
from superstate_graphs.graph_schemas import HISTORY_SUFFIX


STATES = [
    {"id": "initial", "description": "Only the task query is available. No tool action or observation has occurred."},
    {"id": "path_missing", "description": "Inspecting an expected project directory or source file found it missing. Existing alternatives may have been listed, but the missing path was not created."},
    {"id": "build_error", "description": "The last dbt execution failed because SQL referenced unavailable columns. Those SQL errors remain unresolved at this endpoint."},
    {"id": "build_success", "description": "The last dbt run successfully materialized its selected models with no errors; this establishes execution success, not independent correctness or complete task validation."},
    {"id": "query_tool_unavailable", "description": "The most recent direct database query could not execute because its CLI command was unavailable. Existing models may have built successfully earlier; their requested data verification remains unresolved."},
    {"id": "verified_complete", "description": "The task has been fully implemented and independently verified against its requirements, with all necessary checks finished and no unresolved failure."},
    {"id": "project_created", "description": "The most recent observation confirms a new project was created successfully at the requested path."},
]

CASES = [
    ("0054e063-0053-4243-a849-6573c658c164", 0, "initial"),
    ("0054e063-0053-4243-a849-6573c658c164", 13, "build_error"),
    ("0054e063-0053-4243-a849-6573c658c164", 16, "build_success"),
    ("0054e063-0053-4243-a849-6573c658c164", 17, "query_tool_unavailable"),
    ("007b5ce6-0402-4e07-bd01-de58da0e5d6c", 2, "path_missing"),
    ("00b9900d-8f0b-4bf5-b5a6-00fcf8fc7cd7", 1, "path_missing"),
    ("012a216e-05a7-44e3-844e-379f1c43fbed", 16, "build_error"),
]


async def diagnose(args):
    rows = {r["id"]: r for r in map(json.loads, Path(args.corpus).open())}
    id_mapping = {s["id"]: (f"S{i + 1:03d}" if args.opaque_ids else s["id"])
                  for i, s in enumerate(STATES)}
    states = [{**state, "id": id_mapping[state["id"]]} for state in STATES]
    schema = {
        "type": "object", "properties": {
            "state_id": {"type": ["string", "null"], "enum": [s["id"] for s in states] + [None]},
            "evidence": {"type": "string", "maxLength": 240},
        }, "required": ["state_id", "evidence"], "additionalProperties": False,
    }
    if args.evidence_first:
        schema["properties"] = {
            "evidence": {"type": "string", "maxLength": 500},
            "state_id": schema["properties"]["state_id"],
        }
        schema["required"] = ["evidence", "state_id"]
    system = ROUTER_SYSTEM + "\nSTATE SPECIFICATION:\n" + json.dumps(states)
    if args.evidence_first:
        system += (
            "\nBefore selecting a state, first identify the actual final observation and any "
            "unresolved local issue. Write this factual evidence first, then select the most "
            "specific fitting state. Evidence must describe the current endpoint, not the goal."
        )
    async with GraphLLM(args.runtime, cache_dir="results/cache/router_diagnosis", concurrency=8) as llm:
        async def one(rid, step, expected, focal, thinking):
            expected = id_mapping[expected]
            row = rows[rid]
            assert row["split"] == "train"
            full_history = render_history(row, step)
            content = routing_history(row, step) if focal else full_history + HISTORY_SUFFIX
            if args.observation_only:
                content = full_history + HISTORY_SUFFIX
                if step:
                    last = row["steps"][step - 1]
                    observation = row["transcript"][last["action_end_char"]:last["end_char"]]
                    content += (
                        "\nACTUAL FINAL OBSERVATION ONLY, verbatim from the end of this prefix:\n"
                        + observation
                        + "\nEND OF FINAL OBSERVATION. Classify the endpoint AFTER this observation, "
                        "not the situation before it. If this observation reports failure, do not "
                        "classify an earlier success. The initial state requires ZERO observations. "
                        "An unavailable command did not execute its query. A missing path was not "
                        "created. Choose the fitting supplied state ID or null."
                    )
                else:
                    content += "\nThis is the initial query. Exactly zero actions or observations have occurred."
            if args.boundary_count:
                content += (
                    f"\nPROGRAM-VERIFIED PREFIX BOUNDARY: {step} complete action-observation "
                    "groups have occurred in this exact history. The initial state requires "
                    "a count of exactly 0. This count describes only prefix boundaries; "
                    "derive the local situation from the actual observations above."
                )
            assert full_history in content
            started = time.monotonic()
            decoding = {}
            if thinking and args.sampled:
                decoding.update(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0)
            if thinking and args.thinking_budget is not None:
                decoding["thinking_token_budget"] = args.thinking_budget
            answer = await llm.complete_json(
                [{"role": "system", "content": system}, {"role": "user", "content": content}],
                schema=schema, max_tokens=2048 if thinking else (320 if args.evidence_first else 160),
                thinking=thinking,
                cache_namespace="router-grounding-diagnosis-v1",
                **decoding,
            )
            return {"rollout_id": rid, "step": step, "task_id": row["task_id"],
                    "full_history_characters": len(full_history), "focal_reminder": focal,
                    "observation_only_reminder": args.observation_only,
                    "evidence_first": args.evidence_first,
                    "opaque_ids": args.opaque_ids,
                    "boundary_count_supplied": args.boundary_count,
                    "decoding": decoding,
                    "thinking": thinking, "expected": expected, "answer": answer,
                    "correct": answer["state_id"] == expected,
                    "seconds": time.monotonic() - started}

        results = await asyncio.gather(*(
            one(*case, focal, thinking)
            for case in CASES
            for focal, thinking in (
                [(True, False), (True, True)] if args.observation_only
                else [(False, False), (True, False), (True, True)]
            )
        ))
        output = {"purpose": "training-only regression diagnostic, not clustering evaluation",
                  "model": llm.runtime["model"], "revision": llm.runtime["revision"],
                  "cases": results, "usage": llm.stats}
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="results/full_graph/corpus/rollouts.jsonl")
    parser.add_argument("--runtime", default="results/runtime/graph-three.json")
    parser.add_argument("--output", default="results/runtime/router_grounding_35b.json")
    parser.add_argument("--observation-only", action="store_true")
    parser.add_argument("--evidence-first", action="store_true")
    parser.add_argument("--opaque-ids", action="store_true")
    parser.add_argument("--boundary-count", action="store_true")
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--thinking-budget", type=int)
    asyncio.run(diagnose(parser.parse_args()))
