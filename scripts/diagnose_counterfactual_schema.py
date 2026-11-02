"""Post-hoc label-only diagnostic; never replaces frozen strict counterfactual scores."""
import argparse,json,re,hashlib
from collections import Counter
from pathlib import Path
from scripts.analyze_final import counterfactual_summary,mean_seed_delta_interval

def label_only(text):
    text=text.strip()
    if text.startswith('```'):
        m=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,re.DOTALL)
        if not m:return None
        text=m[1]
    try:x=json.loads(text)
    except (ValueError,TypeError):return None
    if not isinstance(x,dict) or set(x)!={'answer','scale'} or not isinstance(x['scale'],str):return None
    answer=x['answer']
    if isinstance(answer,str):answer=[answer]
    if not isinstance(answer,list) or len(answer)!=1 or not isinstance(answer[0],str):return None
    label=answer[0].strip().lower()
    return label if label in {'supported','contradicted'} else None

def diagnose(study,data):
    study=Path(study);data=Path(data)
    config=json.loads((study/'frozen_protocol.json').read_text())
    if json.loads((study/'execution_result.json').read_text())['status']!='FINAL_EVALUATION_COMPLETED':raise ValueError('Complete all frozen conditions first')
    if hashlib.sha256(data.read_bytes()).hexdigest()!=config['evaluation_sha256']:raise ValueError('Input mismatch')
    examples={x['id']:x for x in map(json.loads,data.read_text().splitlines()) if x['source']=='synthetic_cf'}
    summaries={};records={}
    for name in config['model_order']:
        allrows=[json.loads(x) for x in (study/name/'actual.jsonl').read_text().splitlines()]
        rows=[r for r in allrows if r['id'] in examples]
        if len(rows)!=128 or {r['id'] for r in rows}!=set(examples):raise ValueError('Incomplete synthetic predictions')
        rec=[];usable=0;strict=0;scales=Counter()
        for row in rows:
            label=label_only(row['prediction']);usable+=label is not None
            strict+=row['score']['exact_match']
            if label is not None:
                scales[json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',row['prediction'].strip()))['scale']]+=1
            # A new diagnostic row, preserving the original record and strict score unchanged.
            rec.append({**row,'score':{**row['score'],'exact_match':float(label==examples[row['id']]['gold_answer'])}})
        records[name]=rec
        summaries[name]={'n':128,'strict_case_accuracy':strict/128,'label_only_case_accuracy':sum(r['score']['exact_match'] for r in rec)/128,
                         'single_label_json_rate':usable/128,'scale_strings_when_single_label_present':dict(scales),
                         **counterfactual_summary([{**r,'prediction':json.dumps({'answer':[label_only(r['prediction'])],'scale':''}) if label_only(r['prediction']) is not None else r['prediction']} for r in rec],examples)}
    result={'status':'POST_HOC_TEST_SCHEMA_DIAGNOSTIC','specified_after':'Original scale-agnostic rule after B0 test-output inspection. Scalar/list tolerance was clarified after inspecting strict summaries, before computing any label-only metrics. All analyses are post-hoc.',
            'rule':'Require a complete two-key JSON object with answer a single supported/contradicted string (scalar or one-item list, matching the frozen parser), and scale any string; ignore the scale string for this diagnostic only. No gold-dependent parsing.',
            'frozen_primary_scores_changed':False,'model_inputs_or_generation_changed':False,'summaries':summaries,
            'mean_seed_fusion_minus_text_label_only':mean_seed_delta_interval(records,examples,'synthetic_cf'),
            'limitation':'Post-hoc diagnostic of serialization versus label choice, not a preregistered or independent confirmatory test and not evidence of real-world financial validity.'}
    (study/'counterfactual_schema_diagnostic.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study-root',required=True);p.add_argument('--data',default='data/final_v1/evaluation.jsonl');a=p.parse_args()
    x=diagnose(a.study_root,a.data);print(json.dumps({k:v for k,v in x.items() if k!='summaries'},indent=2))
