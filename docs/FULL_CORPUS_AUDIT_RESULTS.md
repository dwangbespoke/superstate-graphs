# Final full-corpus independent audit diagnostics

Generated 2026-09-17T19:23:25.193157+00:00 from the completed saved audit ledger. This is a read-only accounting of frozen outputs; no judgment was rerun or relabeled.

**1776 / 1776 planned checks** have result records; **0** are request-failure records. A result record or valid citation is not evidence of semantic correctness.

| Kind | Planned / recorded | Request failures | Supported | Contradicted | Unknown | Citation-error responses | Definitive verdicts downgraded |
|---|---:|---:|---:|---:|---:|---:|---:|
| within_state_coherence | 101 / 101 | 0 | 14 | 16 | 71 | 69 | 69 |
| boundary_and_redundancy | 30 / 30 | 0 | 3 | 5 | 22 | 21 | 20 |
| edge_source_applicability | 1645 / 1645 | 0 | 115 | 459 | 1071 | 1043 | 1030 |

## Raw → citation-validated verdicts

The table includes only nonzero cells. `unavailable_request_failure` denotes a failed request with no normal raw judgment, rather than a raw unknown judgment.

| Kind | Raw | Citation-validated | Count |
|---|---|---|---:|
| within_state_coherence | contradicted | contradicted | 16 |
| within_state_coherence | contradicted | unknown | 46 |
| within_state_coherence | supported | supported | 14 |
| within_state_coherence | supported | unknown | 23 |
| within_state_coherence | unknown | unknown | 2 |
| boundary_and_redundancy | contradicted | contradicted | 5 |
| boundary_and_redundancy | contradicted | unknown | 14 |
| boundary_and_redundancy | supported | supported | 3 |
| boundary_and_redundancy | supported | unknown | 6 |
| boundary_and_redundancy | unknown | unknown | 2 |
| edge_source_applicability | contradicted | contradicted | 459 |
| edge_source_applicability | contradicted | unknown | 798 |
| edge_source_applicability | supported | supported | 115 |
| edge_source_applicability | supported | unknown | 232 |
| edge_source_applicability | unknown | unknown | 41 |

## Citation diagnostics

Affected-result counts can overlap across categories. Error occurrences also count repeated failures within a response. These counts describe formatting and source coverage checks, not semantic accuracy.

| Kind | Category | Affected results | Error occurrences |
|---|---|---:|---:|
| within_state_coherence | definitive_verdict_without_grounded_citations | 28 | 28 |
| within_state_coherence | missing_tested_source_coverage | 62 | 62 |
| within_state_coherence | quote_not_exact_prefix_substring | 60 | 98 |
| boundary_and_redundancy | definitive_verdict_without_grounded_citations | 6 | 6 |
| boundary_and_redundancy | missing_tested_source_coverage | 19 | 19 |
| boundary_and_redundancy | quote_not_exact_prefix_substring | 16 | 24 |
| edge_source_applicability | definitive_verdict_without_grounded_citations | 366 | 366 |
| edge_source_applicability | missing_or_out_of_scope_citation | 12 | 14 |
| edge_source_applicability | missing_tested_source_coverage | 715 | 715 |
| edge_source_applicability | quote_not_exact_prefix_substring | 944 | 1428 |

## Edge-level outcomes

The graph has 550 observed edge groups. The following flags overlap. All-supported means all *sampled* source checks; eligibility additionally requires proposer plausibility.

| Flag | Edge groups |
|---|---:|
| all_validated_source_checks_supported | 2 |
| all_raw_source_checks_supported | 34 |
| has_validated_contradiction | 314 |
| has_raw_contradiction | 512 |
| has_unknown_validated_verdict | 509 |
| has_request_failure | 0 |
| has_citation_errors | 503 |
| proposer_plausible | 370 |
| proposer_rejected | 180 |
| sampled_supported_eligible | 1 |

## Interpretation and reproduction

- Operational completion, exact citation validity, and semantic reliability are separate properties.
- Request-failure records are the runner's error category and may include response-processing errors; they are not a count of transport failures or model calls.
- A citation-valid quote need not entail the judgment; any invalid extra quote can also downgrade otherwise cited evidence.
- Unknown combines raw semantic uncertainty, citation-invalidated judgments, and request failures; the counts separate those recorded causes without resolving them.
- These are separately prompted LLM proxies, not gold labels, calibrated semantic accuracy, a random population sample, or execution certificates.
- All-source support refers only to planned sampled sources, never every history in a state or executable composed paths.
- No verdict, graph flag, formation decision, or evaluation score is changed by this diagnostic.

The [ten-response qualitative audit review](INDEPENDENT_AUDIT_REVIEW.md) explains examples of citation failures and semantic objections without replacing labels. The [census assignment review](CENSUS_QUALITATIVE_REVIEW.md) and [frozen graph review](FROZEN_GRAPH_REVIEW.md) examine other aspects of formation. Those purposive samples do not establish population error rates.

```sh
.venv/bin/python scripts/summarize_full_graph_audits.py \
  --run results/full_graph/run_v1 \
  --output reports/full-corpus-2026-09-17/audit_diagnostics.json \
  --markdown docs/FULL_CORPUS_AUDIT_RESULTS.md
```

The JSON receipt also contains every edge's diagnostic flags. Both outputs contain counts, IDs, and hashes; source prefixes, raw findings, and error messages are omitted.

| Source | SHA-256 |
|---|---|
| `independent_audit_jobs.json` | `40e24d6ed64d3eaa5c171a6949e1dff456d3c87043fef90a01138e6402b36b0a` |
| `independent_audit_results.json` | `4a46489938f228cf58d8cd1ccff9414e7f0a1f43f5fe90fa1119dfe3ac2c6cc0` |
| `independent_audit_summary.json` | `af3786259e13432c6b7504823579c47ac9fcf1c796cb1f0d50b2ce4dfdc13955` |
| `graph.json` | `03cda04302f45a928fd698304b3ffdfede4e314a7ed778504841a84684173fda` |
| `assignments_all.json` | `b23264bd1327c57df4a31e9730bf60b6790a4c43aa8a1481812876e39571339e` |
| Diagnostic script | `354dd3641dac9a2ae7a847b7dcf2ff5253179fd5e9dcc82291c0e23ef2cd05c8` |
