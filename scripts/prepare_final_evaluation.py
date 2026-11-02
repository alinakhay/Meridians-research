"""Freeze input-selected official-test questions and balanced counterfactual quartets."""
import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from transformers import AutoTokenizer
from scripts.prepare_research import load_examples,token_ids
from meridians.evaluation.evidence import research_prompt,extract_numeric_rows
from meridians.evaluation.scoring import canonical_target,score_example

ROOT=Path(__file__).resolve().parents[1]

def counterfactuals(n=32,seed=1337):
    originals=[json.loads(x) for x in (ROOT/'data/synthetic/test.jsonl').read_text().splitlines()]
    originals.sort(key=lambda r:hashlib.sha256(f'{seed}:cf:{r["id"]}'.encode()).hexdigest())
    result=[]
    for index,original in enumerate(originals[:n]):
        values={(r['metric'],r['period']):Decimal(str(r['value'])) for r in original['numeric_rows']}
        r0,r1=values['revenue',0],values['revenue',1]
        p0,p1=values['operating_income',0],values['operating_income',1]
        m0,m1=p0/r0,p1/r1
        if m0==m1:raise ValueError('Equal margins cannot form a direction-flip quartet')
        # A positive margin on the opposite side of the original prior margin.
        flipped_margin=m0/2 if m1>m0 else m0*Decimal('1.5')
        flipped_income=(flipped_margin*r1).quantize(Decimal('.001'))
        for numeric_flip in (False,True):
            for claim_flip in (False,True):
                current=flipped_income if numeric_flip else p1
                actual_up=current/r1>m0
                claim_up=bool(original['ground_truth']['claim_expansion'])^claim_flip
                label='supported' if actual_up==claim_up else 'contradicted'
                table=[['Metric (million USD)','Prior period','Current period'],
                       ['Revenue',f'{r0:.2f}',f'{r1:.2f}'],
                       ['Operating income',f'{p0:.3f}',f'{current:.3f}']]
                rows,meta=extract_numeric_rows(table,'Amounts in millions of USD.')
                row={'id':f'cf-final-{index:03d}-n{int(numeric_flip)}c{int(claim_flip)}',
                     'source':'synthetic_cf','split':'test','scenario_id':original['id'],
                     'context_id':original['id'],'document_id':None,
                     'numeric_flip':numeric_flip,'claim_flip':claim_flip,
                     'narrative':'Management states: operating margin '+('increased.' if claim_up else 'decreased.'),
                     'numeric_text':'\n'.join(' | '.join(r) for r in table),
                     'numeric_rows':rows,'numeric_metadata':meta,
                     'question':'Is the management claim supported or contradicted by the table?',
                     'gold_answer':label,'scale':'','answer_type':'claim_verification',
                     'research_format':'json_answer_v2'}
                # Recompute from the exact strings supplied to the model, not latent floats.
                display_up=Decimal(table[2][2])/Decimal(table[1][2])>Decimal(table[2][1])/Decimal(table[1][1])
                assert display_up==actual_up
                row['input_sha256']=hashlib.sha256(research_prompt(row).encode()).hexdigest()
                result.append(row)
        quartet=result[-4:]
        assert Counter(r['gold_answer'] for r in quartet)=={'supported':2,'contradicted':2}
        assert (flipped_income/r1>m0)!=(m1>m0)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True)
    p.add_argument('--per-source',type=int,default=64);p.add_argument('--scenarios',type=int,default=32)
    p.add_argument('--seed',type=int,default=1337);a=p.parse_args()
    dest=ROOT/'data/final_v1';dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists():raise RuntimeError('Final evaluation already frozen')
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    used=[json.loads(x) for split in ('train','dev') for x in (ROOT/'data/research_v1'/(split+'.jsonl')).read_text().splitlines()]
    reports={r['document_id'] for r in used if r['source']=='finqa'}
    contexts={(r['source'],r['context_id']) for r in used}
    selected=[];audit={}
    for source in ('finqa','tatqa'):
        eligible=[];excluded=Counter();allrows=list(load_examples(source,'test'))
        for row in allrows:
            row['research_format']='json_answer_v2'
            prompt=research_prompt(row);n=len(token_ids(tok,prompt));reasons=[]
            if row['answer_type']!='arithmetic':reasons.append('non_arithmetic')
            if n>1920:reasons.append('complete_prompt_exceeds_1920')
            if not 0<len(row['numeric_rows'])<=128:reasons.append('numeric_cell_limit')
            if source=='finqa' and row['document_id'] in reports:reasons.append('report_group_in_used_train_or_dev')
            if (source,row['context_id']) in contexts:reasons.append('context_in_used_train_or_dev')
            excluded.update(reasons)
            if not reasons:
                row['input_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
                row['input_tokens']=n;eligible.append(row)
        eligible.sort(key=lambda r:hashlib.sha256(f'{a.seed}:test:{r["id"]}'.encode()).hexdigest())
        if len(eligible)<a.per_source:raise ValueError('Insufficient eligible final questions')
        chosen=eligible[:a.per_source];selected.extend(chosen)
        audit[source]={'source_n':len(allrows),'eligible_n':len(eligible),'selected_n':len(chosen),
                       'excluded_nonexclusive':dict(excluded)}
    selected.extend(counterfactuals(a.scenarios,a.seed))
    selected.sort(key=lambda r:hashlib.sha256(f'{a.seed}:finalmix:{r["id"]}'.encode()).hexdigest())
    for row in selected:
        assert score_example(row,canonical_target(row))['exact_match']==1
        assert len(token_ids(tok,research_prompt(row)))<=1920
    path=dest/'evaluation.jsonl'
    with path.open('x') as f:
        for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest={'status':'FROZEN_BEFORE_TEST_PREDICTION','selection_seed':a.seed,'sources':audit,
              'examples':len(selected),'counterfactual_scenarios':a.scenarios,'counterfactual_cases':4*a.scenarios,
              'input_selection':'Complete arithmetic inputs, fixed token/cell limits and seeded ID hash; gold values and target lengths do not select examples.',
              'used_finqa_report_groups_excluded':len(reports),'train_dev_context_overlap':0,
              'prompt_version':'json_answer_v2; literal empty-string instruction fixes development-detected FinQA ambiguity for every model.',
              'counterfactual_scope':'Synthetic transfer stress test, not evidence of real-world validity. Same latent scenario has claim-only, numbers-only and joint flips; no counterfactual training.',
              'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'ids':[r['id'] for r in selected]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='ids'},indent=2))

if __name__=='__main__':main()
