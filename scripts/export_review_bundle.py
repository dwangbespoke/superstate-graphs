"""Export compact research artifacts for Git; leave raw rollouts and caches local."""
from pathlib import Path
import argparse
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis',type=Path,default=ROOT/'results/analysis_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/poc-2026-09-16')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    names=['summary.json','corpus_manifest.json','task_split.json','initial_candidate.json',
           'selected_candidate.json','selected_graph.json','pooled_outcomes.json',
           'frozen_probe_bank.json','trajectory_coverage.json','probe_signal.json']
    for name in names:
        p=args.analysis/name
        if p.exists():
            contents=p.read_text().replace(str(ROOT)+'/','')
            (args.output/name).write_text(contents)
    for p in sorted((args.analysis/'evaluated_candidates').glob('*.json')):
        value=json.loads(p.read_text());report=value['report']
        compact={'candidate':value['candidate'],'candidate_sha256':report['candidate_sha256'],
                 'proxy_metrics':report['proxy_metrics'],'grounded_metrics':report['grounded_metrics'],
                 'probe_results':report['probe_results'],'evaluated_at_epoch':value['evaluated_at_epoch']}
        out=args.output/'prompt_evolutions'/p.name;out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(compact,indent=2).replace(str(ROOT)+'/',''))
    for p in sorted((ROOT/'results/generated_tasks').glob('*/instruction.md')):
        out=args.output/'tasks'/p.parent.name
        out.mkdir(parents=True,exist_ok=True)
        for name in ('README.md','instruction.md','oracle.sql','alternate.sql','task.json',
                     'expected_result.json','provenance.json','validation.json','verifier.py'):
            source=p.parent/name
            if source.exists():
                (out/name).write_text(source.read_text().replace(str(ROOT)+'/',''))
    receipt=ROOT/'data/runtime/retail.json'
    if receipt.exists():shutil.copy2(receipt,args.output/'warehouse_receipt.json')
    (args.output/'README.md').write_text('''# Overnight POC review artifacts

These are compact outputs from real learner rollouts and prompt optimization.
Raw trajectories, databases, and model caches remain in the local `results/` and
`data/` directories. The HTML report one level up contains interactive inspection.

Formation metrics use frozen LLM judgments, not independent execution labels.
Generated SQL tasks have executable oracle consistency checks. Those checks do
not establish faithful recreation of each source decision or training benefit.

The source benchmark and warehouse are from Snowflake-Labs/data-eng-bench at the
revision in ../../docs/EXPERIMENT.md; upstream licensing continues to apply.
''')
    print(args.output)

if __name__=='__main__':
    main()
