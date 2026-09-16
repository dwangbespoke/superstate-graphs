"""Summarize completed learner trials without exposing request credentials."""
import argparse
import json
from pathlib import Path


def summarize(root:Path):
    rows=[]
    for p in sorted(root.glob('*/result.json')):
        r=json.loads(p.read_text());v=r.get('verifier_result') or {};e=r.get('exception_info') or {}
        task=Path(((r.get('config') or {}).get('task') or {}).get('path') or p.parent.name.split('__')[0]).name
        calls=p.parent/'agent/policy_calls.jsonl'
        rows.append({'task':task,'trial':p.parent.name,'rewards':v.get('rewards'),
                     'error_type':e.get('exception_type'),'error_message':e.get('exception_message'),
                     'policy_calls':len(calls.read_text().splitlines()) if calls.exists() else 0,
                     'usage':{k:v for k,v in (r.get('agent_result') or {}).items() if k in ('n_input_tokens','n_cache_tokens','n_output_tokens','cost_usd')}})
    running=[{'trial':p.parent.parent.name,'policy_calls':len(p.read_text().splitlines())}
             for p in root.glob('*/agent/policy_calls.jsonl') if not (p.parent.parent/'result.json').exists()]
    return {'job':str(root),'completed':len(rows),'trials':rows,'active':running}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path)
    args=parser.parse_args();print(json.dumps(summarize(args.job),indent=2))
