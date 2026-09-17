"""Inspect incomplete structured outputs on a previously identified full prefix."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from superstate_graphs.graph_llm import GraphLLM


async def diagnose(args):
    request = json.loads(Path(args.request).read_text())[0]
    schema = {
        "type": "object", "properties": {
            "evidence": {"type": "string", "maxLength": 500},
            "state_id": {"type": ["string", "null"],
                         "enum": [f"S{i + 1:03d}" for i in range(7)] + [None]},
        }, "required": ["evidence", "state_id"], "additionalProperties": False,
    }
    async with GraphLLM(args.runtime) as llm:
        async def one(compact):
            extra = {"chat_template_kwargs": {"enable_thinking": True},
                     "thinking_token_budget": 1024, "top_k": 20}
            if compact:
                # Request validation requires the constraint itself here, even
                # though response_format later supplies the identical schema.
                extra["structured_outputs"] = {"json": schema, "disable_any_whitespace": True}
            started = time.monotonic()
            response = await llm._client.chat.completions.create(
                model=llm.runtime["model"], messages=request["messages"],
                temperature=0.6, top_p=0.95, presence_penalty=0, max_tokens=2048, seed=17,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "graph_response", "strict": True, "schema": schema,
                }}, extra_body=extra,
            )
            content = response.choices[0].message.content or ""
            return {"compact": compact, "seconds": time.monotonic() - started,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump(), "content_characters": len(content),
                    "non_whitespace_characters": sum(not char.isspace() for char in content),
                    "content": content}

        responses = await asyncio.gather(one(False), one(True), return_exceptions=True)
        responses = [{"error": repr(value)} if isinstance(value, Exception) else value
                     for value in responses]
        result = {"case": {key: value for key, value in request.items() if key != "messages"},
                  "results": responses}
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps([{key: value for key, value in result.items() if key != "content"}
                          for result in responses], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="results/runtime/graph-five.json")
    parser.add_argument("--request", default="results/runtime/router_stuck_requests_v2.json")
    parser.add_argument("--output", default="results/runtime/router_output_format_diagnostic.json")
    asyncio.run(diagnose(parser.parse_args()))
