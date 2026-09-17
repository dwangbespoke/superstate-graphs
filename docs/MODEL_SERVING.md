# Full-history model serving

The graph pipeline uses **Qwen/Qwen3.5-35B-A3B-FP8**, pinned to Hugging Face
revision `9d1823d2dee688a6b25e77009dc727688c44936e`. This is the official FP8
checkpoint of the 35B-parameter, 3B-active-parameter mixture-of-experts model.
The default Modal allocation is one H200, eight CPU cores, and 64 GiB RAM.

The server has a 262,144-token context limit, 32 concurrent sequences, chunked
prefill, and prefix caching. New launches explicitly set `OMP_NUM_THREADS=1` on
the inference subprocess to avoid CPU spin-wait contention. Modal overwrites the
image-level variable from the allocated CPU count: the initial September 17
routers inherited eight threads and the first teacher inherited sixteen. Runtime
metadata records those observed settings; these warmed servers remained online.
A complete prefix that exceeds the context limit
raises an error; the client never truncates, summarizes, or drops input to make
it fit. Cached output keys include the entire request and model revision.

References:

- [Official model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)
- [Official vLLM serving recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-35B-A3B)
- [Modal resource pricing](https://modal.com/pricing)

A separate teacher configuration uses **Qwen/Qwen3.5-122B-A10B-FP8**, pinned to
revision `a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9`, on two H200 GPUs with tensor
parallelism two, 16 CPU cores, 128 GiB RAM, and 16 concurrent sequences. Its
context limit remains 262,144 tokens. See the [official teacher model
card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) and [two-H200 serving
recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-122B-A10B). Teacher and router caches
are separated by model and pinned revision; a replica pool rejects mixed models.

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

Launch the teacher separately:

```bash
SG_GRAPH_MODEL_ROLE=teacher .venv/bin/modal run scripts/modal_graph_models.py \
  --runtime-path results/runtime/graph-teacher.json --duration-seconds 28800
```

Model-role and serving settings are explicitly propagated into the remote image,
because Modal imports the module again inside its container. Verify the actual
remote model name and tensor-parallel setting in sanitized startup logs and wait
for the authenticated readiness check; local launch metadata alone is insufficient.

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

Place the stable routing specification first. The runtime routes independent
rollouts concurrently and keeps each rollout on the same replica. Batches of
at most 32 rollouts warm the first prefix, then use four workers per rollout to
avoid leaving GPUs idle during small GEPA minibatches. Workers take the next
prefix in increasing order; results are stored in prefix order even when calls
finish out of order. Larger batches process each rollout's prefixes serially to
favor cache reuse. Every request supplies its complete history. The endpoint's
32-request semaphore still bounds active router calls; scheduling does not alter
prompts, model settings, scores, or semantic cache keys.

The client supports non-thinking mode, but the production router uses
`thinking=True` after non-thinking failed concrete endpoint-grounding checks.
Thinking defaults to temperature 0.6, top-p 0.95, top-k 20, and presence penalty
zero, following Qwen's precise-task sampling preset. Its separate reasoning
budget is `min(4096, max_tokens // 2)`, leaving room for the JSON answer. These
parameters can be overridden explicitly. `qwen_generation_policy` exposes the
resolved settings for higher-level cache provenance. The reasoning budget only
controls generated output: the complete input history remains unchanged.
See [Qwen's recommended
settings](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8#best-practices) and
[vLLM reasoning budget
control](https://docs.vllm.ai/en/latest/features/reasoning_outputs/#thinking-budget-control).

Structured requests disable arbitrary JSON formatting whitespace. vLLM requires
the same JSON constraint in both `response_format` and `structured_outputs` to
accept this formatting option. The resolved policy records that duplication and
the whitespace setting so higher-level caches invalidate when it changes. This
guard passed the seven endpoint regressions under thinking mode. It is not proven
to eliminate every output-length retry: an eight-router pre-guard diagnostic
finished 255 of its first 256 requests and was interrupted after one long tail.
No completed eight-router throughput measurement is claimed from that run.

Reflection can use a larger output budget. A truncated
output is rejected and retried with a larger output allowance. Transient
network/server failures are retried with bounded backoff and, for a pool, fail
over to another replica with the identical model revision. There are at most
three transient failures per logical request. Read timeout is 900 seconds,
connection timeout 20 seconds, and output-format retries remain separate from
replica selection. Context overflow and invalid request schemas fail directly.

Only complete JSON objects enter the atomic response cache. Cache records
include model revision, token usage, output allowance, and timing. They exclude
API credentials. New cache records also identify the serving Modal app, hash the
complete messages and response schema, and record message lengths. Cache hit
responses incur no additional model request.

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

With the later production routing format (evidence first, repeated final
observation, completed-pair count, and bounded sampled thinking), four H200
replicas processed 128 complete training prefixes in 35.3 seconds and their
128 successors in 40.6 seconds. All 256 completed, with six output-length retries.
These rates, 3.62 and 3.15 histories/second, used a small diagnostic state set;
they exclude graph induction, judging, and scientific audits. An earlier greedy
thinking benchmark was interrupted after one request repeatedly exhausted its
output allowance; it is not counted as a successful benchmark.

`scripts/diagnose_graph_grounding.py` and `scripts/diagnose_graph_router.py`
reproduce observed grounding failures on complete training prefixes. Their
hand-checked examples are serving and prompt regressions, not held-out estimates
of clustering quality. Repeating the focal observation after a complete history
can improve attention to the actual endpoint. An exact evidence substring is
only a provenance check: it does not by itself establish the truth of a claim.
