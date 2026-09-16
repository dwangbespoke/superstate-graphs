"""Resume the remaining bounded POC after an already-running collection finishes."""
from pathlib import Path
import argparse
import json
import os
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
    parser.add_argument('--resume',action='store_true',help='Use existing model servers and cached extraction; do not launch duplicate servers')
    parser.add_argument('--exclude-task',action='append',default=['dbt-consolidate'])
    a=parser.parse_args()
    world_args=[part for task in sorted(set(a.exclude_task)) for part in ('--exclude-task',task)]
    deadline=time.time()+7200
    while time.time()<deadline:
        result=json.loads((a.job/'result.json').read_text())
        if result.get('finished_at'):
            break
        time.sleep(20)
    else:
        raise TimeoutError('Collection has not completed after two hours')
    if not a.resume:
        stop=ROOT/'results/runtime/reflector.stop'
        if stop.exists():stop.unlink()
        reflector_log=(ROOT/'logs/reflector-auto.log').open('w')
        reflector_process=subprocess.Popen([str(ROOT/'.venv/bin/modal'),'run','scripts/modal_models.py','--duration-seconds','7200'],cwd=ROOT,env=dict(os.environ,SG_MODEL_ROLE='reflector'),stdout=reflector_log,stderr=subprocess.STDOUT)
        print(json.dumps({'event':'reflector_launch','pid':reflector_process.pid}),flush=True)
        run('-m','superstate_graphs.analyze','--rollouts',str(a.job),'--output',str(a.analysis),'--extract-only',*world_args)
    for role in ('learner','reflector'):
        ep=json.loads((ROOT/f'results/runtime/{role}.json').read_text())
        with urllib.request.urlopen(urllib.request.Request(ep['api_base']+'/models',headers={'Authorization':'Bearer '+ep['api_key']}),timeout=30) as response:
            assert ep['model'] in {m['id'] for m in json.load(response)['data']}
    run('-m','superstate_graphs.analyze','--rollouts',str(a.job),'--output',str(a.analysis),'--max-metric-calls','144',*world_args)
    run('scripts/generate_examples.py','--analysis',str(a.analysis),'--count','3')
    run('scripts/build_report.py','--analysis',str(a.analysis))
    run('scripts/export_review_bundle.py','--analysis',str(a.analysis))
    for role in ('learner','reflector'):
        (ROOT/f'results/runtime/{role}.stop').touch()
    print(json.dumps({'event':'poc_artifacts_ready','analysis':str(a.analysis)}),flush=True)

if __name__=='__main__':
    main()
