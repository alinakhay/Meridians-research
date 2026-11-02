import hashlib
import json
from pathlib import Path
from collections import defaultdict
from meridians.evaluation.evidence import research_prompt
from meridians.evaluation.scoring import score_example
from scripts.prepare_final_evaluation import counterfactuals
from scripts.analyze_final import counterfactual_summary

ROOT=Path(__file__).resolve().parents[1]

def test_final_prompt_addition_does_not_change_frozen_development_inputs():
    for split in ('train','dev'):
        for row in map(json.loads,(ROOT/'data/research_v1'/(split+'.jsonl')).read_text().splitlines()):
            assert hashlib.sha256(research_prompt(row).encode()).hexdigest()==row['input_sha256']

def test_counterfactual_quartets_reject_a_claim_wording_only_shortcut():
    rows=counterfactuals(4)
    assert len({r['input_sha256'] for r in rows})==16
    groups=defaultdict(dict);predictions=[]
    for row in rows:
        groups[row['scenario_id']][(row['numeric_flip'],row['claim_flip'])]=row
        # This deliberately ignores the table, so each quartet must defeat it.
        label='supported' if 'increased' in row['narrative'] else 'contradicted'
        text=json.dumps({'answer':[label],'scale':''})
        predictions.append({'id':row['id'],'prediction':text,'score':score_example(row,text)})
    for cases in groups.values():
        for claim in (False,True):
            assert cases[False,claim]['narrative']==cases[True,claim]['narrative']
            assert cases[False,claim]['numeric_text']!=cases[True,claim]['numeric_text']
            assert cases[False,claim]['gold_answer']!=cases[True,claim]['gold_answer']
        for numeric in (False,True):
            assert cases[numeric,False]['numeric_text']==cases[numeric,True]['numeric_text']
            assert cases[numeric,False]['gold_answer']!=cases[numeric,True]['gold_answer']
    assert sum(p['score']['exact_match'] for p in predictions)==8
    result=counterfactual_summary(predictions,{r['id']:r for r in rows})
    assert result['all_four_correct']==0
    assert result['numeric_flip_pair_correct_rate']==0
