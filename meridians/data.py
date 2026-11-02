"""Deterministic counterfactual financial scenarios. Fully synthetic: not evidence of real-world model performance."""
from __future__ import annotations
import json, random
from pathlib import Path

METRIC_IDS = {'revenue':1,'operating_income':2,'gross_profit':3,'cash':4,'debt':5,'eps':6}
UNIT_IDS = {'USD_m':1,'percent':2,'USD':3}


def make_case(rng: random.Random, idx: int, split: str) -> dict:
    prev_rev = rng.randrange(40, 3000, 5)
    cur_rev = round(prev_rev * rng.uniform(0.77, 1.42), 2)
    prev_margin = rng.randrange(7, 31)
    delta_margin = rng.choice([-1, 1]) * rng.uniform(2.0, 11.5)
    cur_margin = round(max(1.0, min(65.0, prev_margin + delta_margin)), 2)
    prev_inc = round(prev_rev * prev_margin / 100, 3)
    cur_inc = round(cur_rev * cur_margin / 100, 3)
    expansion = cur_margin > prev_margin
    claim_expand = rng.choice([False, True])
    vocab = rng.choice([
        ('management', 'described'),('the earnings release', 'asserted'),
        ('company commentary', 'stated'),('the CFO', 'said')])
    synonyms = ('improved compared with the prior year' if claim_expand else 'deteriorated compared with the prior year')
    narrative = (f"{vocab[0].capitalize()} {vocab[1]} operating margin as having {synonyms}. "
                 f"The business sells diversified products. These comments are management's claims, not verified facts.")
    consistent = claim_expand == expansion
    label = 'supported' if consistent else 'contradicted'
    rows = [
        {'metric':'revenue', 'metric_id':1, 'period':0, 'value':round(prev_rev,2), 'unit':'USD_m','unit_id':1},
        {'metric':'revenue', 'metric_id':1, 'period':1, 'value':cur_rev, 'unit':'USD_m','unit_id':1},
        {'metric':'operating_income','metric_id':2,'period':0,'value':prev_inc,'unit':'USD_m','unit_id':1},
        {'metric':'operating_income','metric_id':2,'period':1,'value':cur_inc,'unit':'USD_m','unit_id':1},
    ]
    numerical_text = (f"Prior revenue {prev_rev:.2f} million USD; current revenue {cur_rev:.2f} million USD; "
                      f"prior operating income {prev_inc:.3f} million USD; current operating income {cur_inc:.3f} million USD.")
    question = 'Is the claim about the direction of operating margin supported or contradicted by the numbers?'
    explanation = (f"Operating margin changed from {prev_margin:.2f}% to {cur_margin:.2f}%. "
                   f"The claim is {label}.")
    return {
        'id':f'{split}-{idx:06d}', 'source':'meridians_synthetic_v1', 'split':split, 'task_type':'claim_verification',
        'narrative':narrative, 'numeric_rows':rows, 'numeric_text':numerical_text,
        'question':question, 'answer_label':label,
        'answer':f'{label}. {explanation}',
        'ground_truth': {'prior_margin':prev_margin,'current_margin':cur_margin,
                         'margin_delta_pp':round(cur_margin-prev_margin,2),
                         'claim_expansion':claim_expand,'actual_expansion':expansion}
    }


def flip_claim(case:dict)->dict:
    """Make an exact matched counterfactual: same numbers, opposite management statement."""
    import copy
    x=copy.deepcopy(case)
    text=x['narrative']
    if 'improved compared with the prior year' in text:
        text=text.replace('improved compared with the prior year','deteriorated compared with the prior year')
    elif 'deteriorated compared with the prior year' in text:
        text=text.replace('deteriorated compared with the prior year','improved compared with the prior year')
    else:raise ValueError('Unknown claim template')
    x['narrative']=text;x['id']=case['id']+'-claim-flipped';x['pair_id']=case['id']
    x['ground_truth']['claim_expansion']=not x['ground_truth']['claim_expansion']
    x['answer_label']='contradicted' if case['answer_label']=='supported' else 'supported'
    gt=x['ground_truth']
    x['answer']=(f"{x['answer_label']}. Operating margin changed from "
       f"{gt['prior_margin']:.2f}% to {gt['current_margin']:.2f}%. "
       f"The claim is {x['answer_label']}.")
    return x

def create_data(out_dir: str | Path, train=2000, dev=300, test=300, seed=42):
    out = Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    summary={}
    for split, size, offset in [('train',train,0),('dev',dev,1_000_000),('test',test,2_000_000)]:
        rng=random.Random(seed+offset)
        cases=[make_case(rng,i,split) for i in range(size)]
        with (out/f'{split}.jsonl').open('w') as f:
            for x in cases:f.write(json.dumps(x)+'\n')
        counts={k:sum(x['answer_label']==k for x in cases) for k in ('supported','contradicted')}
        summary[split]={'n':size,'labels':counts}
    # Held-out matched claim flips: a model must change its answer when the
    # claim changes while exactly the same numeric evidence is held constant.
    tests=list(load_jsonl(out/'test.jsonl'))
    with (out/'counterfactual_test.jsonl').open('w') as f:
        for x in tests:
            for item in (x,flip_claim(x)):
                f.write(json.dumps(item)+'\n')
    (out/'dataset_manifest.json').write_text(json.dumps({
        'name':'Meridians Synthetic Counterfactual Development Data',
        'type':'fully synthetic', 'license':'CC0-1.0 (new generated examples only)',
        'seed':seed,'sizes':summary,'paired_counterfactual_test_n':test*2,
        'warning':'Do not claim performance on synthetic data generalises to real financial reports.'},indent=2))
    return summary


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():yield json.loads(line)
