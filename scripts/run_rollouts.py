"""Run a bounded Harbor collection with credentials kept out of configs/logs."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/generated/pilot.json')
    parser.add_argument('--name', default='pilot-v1')
    parser.add_argument('--endpoint', type=Path, default=ROOT/'results/runtime/learner.json')
    args = parser.parse_args()
    deadline = time.time()+900
    while time.time()<deadline:
        endpoint=json.loads(args.endpoint.read_text())
        request=urllib.request.Request(endpoint['api_base']+'/models', headers={'Authorization':'Bearer '+endpoint['api_key']})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status==200:
                    break
        except Exception:
            time.sleep(5)
    else:
        raise TimeoutError('Learner server unavailable after 15 minutes')
    subprocess.run([sys.executable,'-m','superstate_graphs.stage','--endpoint',str(args.endpoint)],cwd=ROOT,check=True)
    env=dict(os.environ, HOSTED_VLLM_API_KEY=endpoint['api_key'], OPENAI_API_KEY=endpoint['api_key'])
    print(json.dumps({'event':'rollouts_starting','job':args.name,'model':endpoint['model']}),flush=True)
    subprocess.run([str(ROOT/'.venv/bin/harbor'),'run','-c',args.config,'--job-name',args.name],env=env,cwd=ROOT,check=True)

if __name__=='__main__':
    main()
