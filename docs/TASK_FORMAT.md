# Composed SQL task format

The first task-construction backend uses the shared DuckDB warehouse directly.
It produces read-only analytical workflows. It does not reconstruct a full dbt
project or pretend that the original learner's information state has been replayed.

```python
from superstate_graphs.task_constructor import construct_task

result = construct_task(
    path_spec=selected_path,
    llm=lambda prompt: model_call_returning_json(prompt),
    db_path="data/runtime/retail.duckdb",
    output_dir="results/generated_tasks/example_01",
    max_repairs=2,
)
```

`path_spec` contains `path_id`, `target_superstate`, and `transitions`. Each transition
requires `source_task_id`, `source_history_id`, `target_history_id`, and `operation`.
Provide `transition_id` if available; otherwise IDs are assigned as `edge_0`, etc.
Include complete concrete witnesses, prerequisites, effects, and graph metadata as
additional fields. All fields are retained in the generated task's provenance.
There must be at least two transitions from at least two source tasks.

Use `data/runtime/retail.duckdb`, the pristine warehouse exported from the pinned
base environment after its official `fix_data.py` and dbt setup. The unmodified
vendor download has null `RAW_SAP.VBAP.product_id` values which the official setup
repairs. The constructor never modifies a database or silently applies that fix.

Schema retrieval is bounded. It loads a compact inventory of relation names,
prioritizes exact SQL and qualified-table mentions in the selected path, then uses
lexical relevance with a preference for underlying tables. It fetches columns and
samples for at most 24 relations (configurable up to 40), and caps serialized schema
context at 24,000 characters. Wide tables retain all column names and types in a
compact mapping; samples include at most 12 named columns and two rows. The context
records selection reasons, unresolved references, and omitted relation counts.
Unreferenced `main` namesakes do not displace explicitly qualified source tables.
For a large catalog with no relevant path references, construction fails clearly
instead of supplying arbitrary tables. A small-catalog fallback is explicitly labeled.

The complete prompt is capped at 60,000 characters, including repair feedback.
This is a character cap, not a tokenizer-specific token bound. Large path evidence
is visibly excerpted with hashes; source task/history/transition IDs remain intact,
and the full original path is retained in provenance. A caller-provided schema must
fit this cap too. No extra LLM call is used for schema retrieval.

The model returns the schema shown in `construction_prompt`. Intermediate stages
are dependent, named SELECT queries assembled into one CTE chain. Stage definitions
cite actual transition IDs. The final query must consume the last stage. Execution
checks every stage for nonempty output and checks the final query against a second
formulation and at least two result constraints. Failed candidates and feedback
are retained; the constructor makes at most `max_repairs + 1` model calls.

The generated directory contains:

- `instruction.md`: the solver-facing problem and query submission contract.
- `oracle.sql`: executable composed reference query.
- `alternate.sql`: separately formulated query from the same generator.
- `task.json`: stages, constraints, output schema, and generation explanations.
- `expected_result.json`: frozen oracle output on the construction warehouse.
- `verifier.py`: standalone Python script; requires only Python and DuckDB.
- `provenance.json`: source histories, transitions, tasks, target, and stage bindings.
- `schema_context.json`: the exact bounded schema retrieval supplied to construction.
- `validation.json`: actual stage counts/samples, output, checks, and limitations.
- `construction_attempts.json`: rejected candidates and repair feedback.

Give a solver only `instruction.md` and the database. Keep all reference and
verification files outside its environment. Grade a submission afterward:

```bash
python results/generated_tasks/example_01/verifier.py \
  --db data/runtime/retail.duckdb \
  --submission /path/to/answer.sql
```

The verifier compares column names and values, respecting declared row ordering
and a numeric tolerance of `max(1e-6, 1e-6 * magnitude)`. Unordered comparisons
preserve duplicate multiplicity. It checks the generated result constraints too.
Use the same frozen warehouse as construction; provenance records a filesystem
receipt, while the experiment's dataset manifest should supply its content hash.

All queries run through a read-only connection with external access disabled.
Only a single SELECT/WITH statement is accepted, with per-query interrupt timers,
a 1 GB memory limit, and at most 1,000 final result rows.

These checks establish executability and internal consistency. They do not prove
semantic preservation of the targeted decision, independence of the grader,
novelty relative to all source tasks, learner difficulty, or training benefit.
The alternate SQL and constraints share their generator with the oracle. Source
citations and CTE dependencies are auditable evidence, not a semantic proof that
an operation has been reused faithfully. The tiny unit-test database measures
software correctness only; it is not research evidence.
