# Witness-specific exploratory SQL task

Evidence label: `witness_specific_exploratory_not_sampled_supported`. Reusable edge applicability remains unresolved.

**FAILED RESEARCH CASE — NOT LEARNER-READY.** Manual review identified a semantic defect. The preserved queries and expected output may encode an incorrect or underspecified task; do not use this case as a validated learner benchmark. Local query agreement remains the original recorded outcome.

See the [manual semantic review](review.json). The original artifacts remain unchanged, including any documented defects.

The task-input artifacts are [instruction.md](instruction.md) and [fixture.duckdb](fixture.duckdb). [oracle.sql](oracle.sql), [independent.sql](independent.sql), and [expected_result.json](expected_result.json) are preserved reference outputs, not independently certified truth.

From this directory in the repository's Python environment:

```sh
python -m superstate_graphs.graph_task_examples verify --task-dir . --submission answer.sql
```

Two separately prompted queries ran and agreed on this synthetic fixture after a LLM faithfulness review. No learner or training lift was evaluated. See [provenance](provenance.json) and [execution receipt](local_validation.json).
