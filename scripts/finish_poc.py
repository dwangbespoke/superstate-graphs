"""Resume the remaining bounded POC after an already-running collection finishes."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def run(*args):
    print(json.dumps({'event':'phase_start','command':list(args)}),flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,default=ROOT/'results/rollouts/pilot-v2')
    parser.add_argument('--analysis',type=Path,default=ROOT/'results/analysis_v1')
    a=parser.parse_args()
    deadline=time.time()+7200
    while time.time()<deadline:
        result=json.loads((a.job/'result.json').read_text())
        if result.get('finished_at'):
            break
        time.sleep(20)
    else:
        raise TimeoutError('Collection has not completed after two hours')
    for role in ('learner','reflector'):
        ep=json.loads((ROOT/f'results/runtime/{role}.json').read_text())
        with urllib.request.urlopen(urllib.request.Request(ep['api_base']+'/models',headers={'Authorization':'Bearer '+ep['api_key']}),timeout=30) as response:
            assert ep['model'] in {m['id'] for m in json.load(response)['data']}
    run('-m','superstate_graphs.analyze','--rollouts',str(a.job),'--output',str(a.analysis),'--max-metric-calls','144')
    run('scripts/generate_examples.py','--analysis',str(a.analysis),'--count','3')
    run('scripts/build_report.py','--analysis',str(a.analysis))
    print(json.dumps({'event':'poc_artifacts_ready','analysis':str(a.analysis)}),flush=True)

if __name__=='__main__':
    main()
