# Qualitative review of the selected frozen graph

This is a purposive coding-agent review of the selected **28-state, 34-edge frozen specification**, before all-corpus completion and reconstruction of operation contracts. It examines written definitions and selected original training evidence. It is not a human study, gold-label assessment, or measured semantic accuracy. No heldout feedback was used, and these findings were not supplied to optimization or formation.

## Verified specification changes

The selected candidate is proposal `0007`, archive index 2, with digest:

`a24da5e97a2156d80947ed79268f433d5456d587b00a97d338cd804bc85390f3`

Its `state_spec` is byte-for-byte identical to the seed. All 30 seed edges are unchanged; four were added. `explore_initial` substantially repeats `explore_directory`, while `run_model_after_deps` repeats `run_model`'s endpoints and operation. Only `explore_existing_project` and `run_model_from_files` introduce new endpoint pairs. The resulting graph has 29 distinct endpoint pairs, 13 states with no outgoing edge, and two entirely isolated states: `column_type_error` and `sql_syntax_error`. These counts were calculated directly from the saved specifications.

The selected result therefore changes the edge specification, not the routing definitions or state partition. State-changing candidates existed in the archive but were not selected. Subsequent all-corpus graph changes must be reported separately.

## Concrete findings

1. **Connection verification conflicts with a target exclusion.** `profiles_configured` permits histories in which model files already exist. Its `verify_connection` edge runs `dbt debug` and targets `dbt_debug_verified`, which excludes `models created`. The operation does not erase existing files or the fact of their creation. A relevant training ordering occurs in `00b9900d-8f0b-4bf5-b5a6-00fcf8fc7cd7` (`dbt-weekly-sales-growth`): models are present before `h0009`; `t0009` reports valid profile/project checks and a successful connection test, alongside a separate missing-git failure. The written exclusion prevents representing that ordering in the intended target. This is a source/target contract conflict, not a population failure rate.

2. **Writing a repair does not establish a rebuilt target.** `fix_type_cast` and `fix_sql_syntax` describe SQL edits, but target `model_rebuilt_with_fix`, which requires a rebuild invocation and excludes an original error still present. The edge records do not establish those conditions; `rebuild_after_fix` subsequently initiates a rebuild. Both repair edges also leave the broad `model_execution_failed` state, which admits missing-table, casting, and parser failures. A casting edit alone need not resolve every admitted blocker. These are limits of the advertised every-source-member applicability.

3. **A missing-dependency outcome is not justified by the source.** `check_packages` starts at `model_files_created`, which does not require dependencies to be missing, but promises a missing-packages error and targets `dbt_deps_needed`. That target explicitly excludes already installed dependencies. A source member with packages installed cannot reach that target by compiling while preserving the established facts. Recording this branch in some rollout would not establish the universal-source contract.

4. **Inspection success does not establish all task requirements.** `python_verification_succeeds` promises `data_verified`, whose description combines observed schema/count/sample facts with requirements being met. Its stated checks do not establish arbitrary task requirements. In rechecked training transition `5dffab08-7526-4d60-a462-2ee7253a3f34:t0036` (`dbt-inventory-turnover-analysis`), a Python query runs successfully and reports zero rows. This illustrates that a working inspection and an acceptable result are distinct; it does not assert that this history was finally assigned incorrectly. The separate `empty_result_set` state captures one outcome but does not repair the general verification contract.

These findings concern the frozen specification. Sinks are not automatically invalid, and counterexamples admitted by definitions do not estimate their frequency in the corpus. Final sampled audits and executable synthetic examples provide additional, narrower evidence; they do not establish universal composability or long-term state equivalence. See the [method](FULL_CORPUS_METHOD.md) and [claim boundaries](RESEARCH_CLAIMS.md).
