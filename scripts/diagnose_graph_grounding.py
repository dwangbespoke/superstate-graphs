"""Check a known production grounding failure without reducing corpus coverage."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from superstate_graphs.full_corpus import render_history, render_step
from superstate_graphs.graph_evolution import GRAPH_CONTRACT
from superstate_graphs.graph_llm import GraphLLM
from superstate_graphs.graph_schemas import HISTORY_SUFFIX, LOCAL_TRANSITION_SCHEMA


async def diagnose(args):
    rows = [json.loads(line) for line in Path(args.corpus).open()]
    rollout = min((r for r in rows if r["task_id"] == args.task), key=lambda r: r["id"])
    step = args.step
    prefix = render_history(rollout, step)
    transition = render_step(rollout, step)
    record = rollout["steps"][step]
    observation = rollout["transcript"][record["action_end_char"]:record["end_char"]]
    system = GRAPH_CONTRACT + (
        "\nAnalyze EXACTLY ONE source history and its immediately next action-observation group. "
        "Describe the source's CURRENT local knowledge, unresolved issue, or available operation; "
        "describe how this one observation changes it into the target. Do not turn dbt model names "
        "or dataflow layers into agent states. Source uses only source-prefix evidence. "
        "Target may use this next observation, never imagined later work. Use SHORT EXACT literal "
        "substrings as evidence_quote and observation_quote, without ellipses or invented quotes. "
        "The observation quote must come from the actual next tool response. "
        "Return source,target,operation,effect,observation_quote."
    )
    async with GraphLLM(args.runtime, cache_dir="results/cache/grounding_diagnosis") as llm:
        async def one(repeat_focal):
            content = ("SOURCE HISTORY:\n" + prefix
                       + "\n=== NEXT COMPLETE ACTION-OBSERVATION GROUP ===\n" + transition
                       + HISTORY_SUFFIX)
            if repeat_focal:
                content += (
                    "\nFOCAL OBSERVATION REPEATED FOR EVIDENCE CHECKING:\n" + observation
                    + "\nEND OF EVIDENCE. Describe only the local change actually established "
                    "by this focal observation. An attempted command and a successful operation "
                    "are different. Do not infer success from the original task requirements. "
                    "If the operation failed, describe the failure and unresolved issue. "
                    "Copy an exact short quote from this focal observation."
                )
            answer = await llm.complete_json(
                [{"role": "system", "content": system}, {"role": "user", "content": content}],
                schema=LOCAL_TRANSITION_SCHEMA, max_tokens=8192, thinking=True,
                cache_namespace="grounding-diagnosis-v1",
            )
            quotes = [(answer["source"]["evidence_quote"], prefix),
                      (answer["target"]["evidence_quote"], prefix + transition),
                      (answer["observation_quote"], observation)]
            return {"repeat_focal_observation": repeat_focal, "answer": answer,
                    "verbatim_quotes": [bool(quote) and quote in text for quote, text in quotes]}

        answers = await asyncio.gather(one(False), one(True))
        result = {"model": llm.runtime["model"], "revision": llm.runtime["revision"],
                  "rollout_id": rollout["id"], "task_id": args.task, "step": step,
                  "full_source_characters": len(prefix), "full_next_step_characters": len(transition),
                  "thinking": True, "answers": answers, "usage": llm.stats}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="results/full_graph/corpus/rollouts.jsonl")
    parser.add_argument("--runtime", default="results/runtime/graph.json")
    parser.add_argument("--task", default="fifo-inventory-cogs")
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--output", default="results/runtime/grounding_35b_thinking.json")
    asyncio.run(diagnose(parser.parse_args()))
