"""Build a portable local experiment report from recorded artifacts."""
from pathlib import Path
import argparse
import html
import json

ROOT=Path(__file__).resolve().parents[1]


def build(analysis:Path,output:Path):
    read=lambda name:json.loads((analysis/name).read_text())
    summary=read('summary.json')
    candidate=read('selected_candidate.json')
    data={'summary':summary,'codebook':json.loads(candidate['codebook']),
          'report':read('selected_report.json'),'histories':read('histories.json'),
          'outcomes':read('pooled_outcomes.json'),'paths':read('selected_paths.json')}
    data['revisions']=[]
    for p in sorted((analysis/'evaluated_candidates').glob('*.json')):
        c=json.loads(p.read_text())
        data['revisions'].append({'candidate':c['candidate'],'metrics':c['report']['proxy_metrics'],
                                  'hash':c['report']['candidate_sha256'],'time':c['evaluated_at_epoch']})
    data['revisions'].sort(key=lambda r:r['time'])
    data['tasks']=[]
    for p in sorted((ROOT/'results/generated_tasks').glob('*/instruction.md')):
        data['tasks'].append({'name':p.parent.name,'instruction':p.read_text(),
                              'format':'Read-only SQL workflow',
                              'validation':json.loads((p.parent/'validation.json').read_text()) if (p.parent/'validation.json').exists() else {},
                              'path':str(p.parent)})
    for p in sorted((ROOT/'results/generated_dbt_tasks').glob('*/instruction.md')):
        receipt=p.parent/'runtime_validation.json'
        if not receipt.exists():continue
        validation=json.loads(receipt.read_text())
        if validation.get('status')!='validated':continue
        data['tasks'].append({'name':p.parent.name,'instruction':p.read_text(),
                              'format':'Mutable dbt project',
                              'validation':validation,'path':str(p.parent)})
    template=(ROOT/'reports/template.html').read_text()
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(template.replace('__REPORT_DATA__',payload))
    print(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis',type=Path,default=ROOT/'results/analysis_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/overnight-poc.html')
    a=parser.parse_args();build(a.analysis,a.output)
