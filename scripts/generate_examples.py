"""Instantiate a few cross-task graph paths and execute their SQL tasks."""
from pathlib import Path
import argparse
import json

from superstate_graphs.analyze import CachedClient, JUDGE_PROMPT, json_response, save
from superstate_graphs.task_constructor import construct_task

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis',type=Path,default=ROOT/'results/analysis_v1')
    parser.add_argument('--db',type=Path,default=ROOT/'data/runtime/retail.duckdb')
    parser.add_argument('--output',type=Path,default=ROOT/'results/generated_tasks')
    parser.add_argument('--count',type=int,default=3)
    args=parser.parse_args()
    paths=json.loads((args.analysis/'selected_paths.json').read_text())['paths']
    histories={h['history_id']:h for h in json.loads((args.analysis/'histories.json').read_text())}
    client=CachedClient(ROOT/'results/runtime/reflector.json',args.output/'cache','constructor')
    outcomes=[]
    seen=set()
    for path in paths:
        identity=(path['target_superstate'],tuple(path['source_task_ids']))
        if identity in seen:
            continue
        seen.add(identity)
        left,right=path['transitions'][:2]
        payload={'prefix_A':json.loads(histories[left['target_history_id']]['prefix']),
                 'prefix_B':json.loads(histories[right['source_history_id']]['prefix']),
                 'observed_segment_after_B':right['effects']}
        judgment=json_response(client.call(JUDGE_PROMPT+json.dumps(payload),max_tokens=2400))
        save(args.output/'junction_judgments'/f"{path['path_id']}.json",{'path_id':path['path_id'],'judgment':judgment,'tier':'LLM compatibility proxy, not execution proof'})
        if any(judgment[key]['label']=='contradicted' for key in ('decision','splice')):
            outcomes.append({'path_id':path['path_id'],'status':'skipped_contradicted_junction'})
            continue
        path['junction_judgment']=judgment
        target=args.output/path['path_id']
        try:
            result=construct_task(path,lambda prompt:json_response(client.call(prompt,thinking=False,max_tokens=10000,temperature=.4)),args.db,target,max_repairs=2)
            outcomes.append({'path_id':path['path_id'],'output':str(target),'status':result.get('status'),'result':result})
        except Exception as error:
            outcomes.append({'path_id':path['path_id'],'status':'error','error':str(error)})
        save(args.output/'summary.json',{'attempted':outcomes,'client':client.stats()})
        print(json.dumps({'event':'task_attempted','path_id':path['path_id'],'status':outcomes[-1]['status']}),flush=True)
        if sum((args.output/o['path_id']/'instruction.md').exists() for o in outcomes)>=args.count:
            break
    save(args.output/'summary.json',{'attempted':outcomes,'client':client.stats()})

if __name__=='__main__':
    main()
