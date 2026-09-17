# Full-history model serving

The graph pipeline uses **Qwen/Qwen3.5-35B-A3B-FP8**, pinned to Hugging Face
revision `9d1823d2dee688a6b25e77009dc727688c44936e`. This is the official FP8
checkpoint of the 35B-parameter, 3B-active-parameter mixture-of-experts model.
The default Modal allocation is one H200, eight CPU cores, and 64 GiB RAM.

The server has a 262,144-token context limit, 32 concurrent sequences, chunked
prefill, and prefix caching. New launches set `OMP_NUM_THREADS=1` to avoid CPU
spin-wait contention; the initial September 17 four-server deployment inherited
eight threads and remained online rather than interrupting the experiment.
A complete prefix that exceeds the context limit
raises an error; the client never truncates, summarizes, or drops input to make
it fit. Cached output keys include the entire request and model revision.

References:

- [Official model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)
- [Official vLLM serving recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-35B-A3B)
- [Modal resource pricing](https://modal.com/pricing)

## Launch and stop

From the repository root, keep the following process running:

```bash
.venv/bin/modal run scripts/modal_graph_models.py --duration-seconds 28800
```

The bounded lifetime includes startup. Runtime connection details are written
with file mode `0600` to `results/runtime/graph.json`, which is ignored by git.
The adjacent `graph.ready` marker appears after the authenticated model-list
check succeeds. Stop this server by creating `results/runtime/graph.stop`.
Remove that marker intentionally before launching a replacement. Do not print
the runtime JSON: it contains the bearer credential.

vLLM's output is filtered before reaching Modal logs because its normal startup
log includes API key arguments. The endpoint permits transport-level access but
vLLM requires the generated bearer key for inference.

For additional independent replicas, set `SG_GRAPH_REPLICA` and use a distinct
runtime path. Each replica incurs its own GPU, CPU, and RAM charges:

```bash
SG_GRAPH_REPLICA=two .venv/bin/modal run scripts/modal_graph_models.py \
  --runtime-path results/runtime/graph-two.json --duration-seconds 28800
```

As of September 17, 2026, Modal advertises H200 at $0.001261 per second
($4.5396 per GPU-hour), before CPU, memory, and any account-specific discounts.
An eight-hour server allocation is bounded in time, not an exact monetary cap.

## Client

```python
from superstate_graphs.graph_llm import GraphLLM

async with GraphLLM(concurrency=32) as llm:
    assignment = await llm.complete_json(
        [
            {"role": "system", "content": routing_specification},
            {"role": "user", "content": full_history},
        ],
        schema=assignment_json_schema,
        max_tokens=128,
    )
```

Place the stable routing specification first. Schedule each rollout's prefixes
in increasing length to make its already processed prefix available to the
server cache; process independent rollouts concurrently. Prefix caching is a
compute optimization and does not change the supplied history.

The client uses non-thinking mode by default for short structured assignments.
Reflection can enable `thinking=True` and a larger output budget. A truncated
output is rejected and retried with a larger output allowance. Transient
network/server failures are retried with bounded backoff. Context overflow and
invalid request schemas fail directly.

Only complete JSON objects enter the atomic response cache. Cache records
include model revision, token usage, output allowance, and timing. They exclude
API credentials. Cache hit responses incur no additional model request.

## Verification

```bash
.venv/bin/pytest -q tests/test_graph_llm.py
.venv/bin/ruff check scripts/modal_graph_models.py src/superstate_graphs/graph_llm.py
```

The integrity checks verify that histories with different endings produce
different cache entries, model revisions invalidate cached results, and
incomplete output retries preserve the entire original input.

`scripts/benchmark_graph_models.py` measures serving capacity using 32 complete
real training prefixes followed by their next prefixes. It does not fit a graph.
Use a strict response schema and repeat the classification instruction after the
delimited raw history: unconstrained JSON can otherwise continue the embedded
terminal-agent task instead of classifying it. On the first H200 replica tested
in isolation, the constrained benchmark completed all 64 requests without retries:
1.38 histories/second for the initial prefixes and 2.14 for subsequent prefixes,
averaging approximately 46 generated tokens. These are serving measurements,
not graph-quality results or a guaranteed end-to-end pipeline rate.
