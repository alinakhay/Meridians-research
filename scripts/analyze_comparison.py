"""Paired development deltas and cluster bootstrap intervals. No significance claim."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def paired_interval(rows_a,rows_b,examples,source,seed=2026,replicates=2000):
    a={x['id']:x['score']['exact_match'] for x in rows_a if x['score']['source']==source}
    b={x['id']:x['score']['exact_match'] for x in rows_b if x['score']['source']==source}
    if set(a)!=set(b):raise ValueError('Paired inference IDs differ')
    groups=defaultdict(list)
    for uid in sorted(a):
        x=examples[uid]
        group=x['document_id'] if source=='finqa' else x['context_id']
        groups[group].append(b[uid]-a[uid])
    keys=sorted(groups);rng=random.Random(seed);deltas=[]
    for _ in range(replicates):
        sample=[]
        for _ in keys:
            sample.extend(groups[rng.choice(keys)])
        deltas.append(sum(sample)/len(sample))
    deltas.sort()
    return {'n':len(a),'clusters':len(keys),
            'cluster_type':'company_year_report_proxy' if source=='finqa' else 'table_context_uuid',
            'delta':sum(b[uid]-a[uid] for uid in a)/len(a),
            'exploratory_cluster_bootstrap_95_interval':[deltas[int(.025*replicates)],deltas[int(.975*replicates)-1]],
            'replicates':replicates,'seed':seed}


def analyze(root,dev_path,include_ablations=True):
    root=Path(root)
    examples={x['id']:x for x in map(json.loads,Path(dev_path).read_text().splitlines())}
    output={'status':'EXPLORATORY_DEVELOPMENT_RESULTS','test_sets_used':False,'summaries':{},'paired_comparisons':{}}
    records={}
    for label,folder,condition in [('B0','B0','actual'),('B1','B1/evaluation','actual'),
                                  ('N1','N1/evaluation','actual'),('N1_zero','N1/evaluation','zero_values'),
                                  ('N1_permuted','N1/evaluation','permute_values')]:
        if not include_ablations and label in {'N1_zero','N1_permuted'}:continue
        directory=root/folder
        output['summaries'][label]=json.loads((directory/(condition+'.summary.json')).read_text())
        records[label]=[json.loads(x) for x in (directory/(condition+'.jsonl')).read_text().splitlines()]
        if len(records[label])!=len(examples) or {x['id'] for x in records[label]}!=set(examples):
            raise ValueError('Incomplete or duplicate development predictions for '+label)
    for left,right in [('B0','B1'),('B1','N1'),('N1','N1_zero'),('N1','N1_permuted')]:
        if right not in records:continue
        output['paired_comparisons'][right+'_minus_'+left]={source:paired_interval(records[left],records[right],examples,source) for source in ('finqa','tatqa')}
    b1=json.loads((root/'B1/training_manifest.json').read_text())
    n1=json.loads((root/'N1/training_manifest.json').read_text())
    for key in ('steps','seed','dataset_sha256','data_order_sha256','batch_size','gradient_accumulation','learning_rate'):
        if b1[key]!=n1[key]:raise ValueError('Matched training control differs: '+key)
    output['matched_controls_verified']={key:b1[key] for key in ('steps','seed','dataset_sha256','data_order_sha256','batch_size','gradient_accumulation','learning_rate')}
    output['trainable_parameters']={'B1':b1['trainable_parameters'],'N1':n1['trainable_parameters']}
    output['limitations']=['One seed and 64 updates; exploratory direct-answer development comparison only.',
        'Subset restricted to complete short-context arithmetic questions; full-benchmark accuracy is not measured.',
        'Confidence intervals are descriptive and not corrected for multiple exploratory comparisons.',
        'TAT-QA clusters are table contexts, not verified source documents.',
        'Public-benchmark pretraining contamination cannot be excluded.',
        'Numeric corruption at inference is a mechanism diagnostic, not a separately trained parameter-matched baseline.']
    (root/'comparison_results.json').write_text(json.dumps(output,indent=2))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--dev',default='data/research_v1/dev.jsonl');a=p.parse_args()
    print(json.dumps(analyze(a.root,a.dev),indent=2))
