# Overnight proof of concept — September 15–16, 2026

## Research target

Find recurring, learner-relevant decision situations across different tasks.
Represent them as interpretable superstates connected by observed operations.
Use cross-task paths through this graph to construct new executable tasks.
Compute descriptive outcome variance after formation, without optimizing the
partition to manufacture a 50% success rate or training a success predictor.

The immediate deliverables are actual rollouts, evaluated GEPA prompt revisions,
a graph with concrete transition witnesses, and a few generated task examples.
The user explicitly prioritizes an end-to-end POC over extensive validation.

## Environment and models

Snowflake data-eng-bench, pinned at
`a3278ad102829a6084dde086244a0ef665a8011c`, supplies a common synthetic retail
warehouse and dbt project. The DuckDB backend needs no Snowflake credentials.
The public AMD64 base image is pinned to
`sha256:ef92b6ef197a89ff1d8b371aaf5de19343003abc462991ddeedb1b1005e2e04b`.
Every task's COPY/RUN overlays are retained. Its redundant single-service
compose file is removed in a derived working copy so Harbor can use native Modal
sandboxes. Instructions, oracle, and verifier contents remain unchanged.

- Learner: Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Extractor/classifier: same frozen 9B weights, separate prompts and decoding.
- Reflector: Qwen3.5-35B-A3B-FP8, revision `9d1823d2dee688a6b25e77009dc727688c44936e`.
- Learner harness: Harbor 0.23.0 Terminus-2, with a prefix logging subclass.
- Learner decoding: temperature 0.6, top-p 0.95, top-k 20, thinking enabled;
  4,096 output-token cap, 30 turns, 20-minute episode limit, no summarization.
- Model server: one L40S learner container; one H100 reflector only when needed.
  A finite local server lifetime and explicit stop markers prevent unbounded deployments.
  One warm container avoids repeated model reloads between rollout tool calls.

## Run sequence

1. One oracle/environment smoke to expose broken setup.
2. Six-task × two-attempt learner pilot, then a bounded source collection as time
   permits. No extended model sweep is required for tonight's POC.
3. Cache prefix-only history records; retain goals, known facts, unresolved
   questions, prior attempts, and available operations.
4. Freeze a small cross-task compatibility bank. LLM-only decisions are explicit
   **proxy labels**, not execution-grounded truth.
5. Optimize codebook and router prompts with GEPA. Save seed, evaluated revisions,
   assignments, score components, feedback, and graph for comparison.
6. Derive edges from executed observations. Retry-only calls are not transitions.
7. Attach outcome statistics once per distinct rollout, reporting task support.
   Missing/infrastructure-error rewards remain unknown. If all measured rewards
   are zero, variance is zero; target supported groups for the construction POC
   without claiming observed high variance.
8. Construct 2–3 examples with witnesses from at least two source tasks. Bind one
   coherent database workflow; execute oracle and checks. Preserve source/path
   provenance and label validation limits explicitly.

## Evaluation boundary

Task IDs are split before fitting. Coarse task-family labels aid the split but
are not a validated semantic partition. This is an internal feasibility study,
not an untouched benchmark score. Generated descendants retain their lineage.

The proxy GEPA objective rewards compatible cross-task reuse and penalizes
contradicted joins. Decision judgments use prefixes only. Transfer judgments
distinguish the first witnessed execution episode from the entire observed
segment. Positive local transfer evidence may support formation; it does not
certify the full segment or path. Full-segment contradictions block construction
proposals. These proxies do not prove shared per-history difficulty, whole-path
executability, or training benefit. Constructed tasks are the next concrete test.

Two independent adversarial critics reviewed intent and evidence alignment.
Their actionable findings are incorporated above. Further critique focuses on
the actual resulting artifacts rather than expanding the scope of validation.

## Sources

- https://github.com/Snowflake-Labs/data-eng-bench
- https://huggingface.co/datasets/adyen/DABstep
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8
- https://arxiv.org/abs/2507.19457
- https://gepa-ai.github.io/gepa/guides/adapters/

## Executed scope correction

The complete collection contains 12 rollouts. A post-collection, instruction-based
world audit found that dbt-consolidate uses separate task-specific CSV inputs. Its
two rollouts remain auxiliary; the core graph/GEPA use the other five tasks and
ten rollouts. All nine graded core outcomes are zero; one context-limit error is
ungraded. See PILOT_AUDIT.md. The SQL construction POC therefore uses a support
fallback and cannot claim high-variance target discovery in this run.
