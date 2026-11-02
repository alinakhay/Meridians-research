"""Analyze every predetermined seed and paired scenario, without choosing a winner."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from scripts.analyze_comparison import paired_interval
from meridians.evaluation.scoring import parse_answer

def counterfactual_summary(rows,examples):
    groups=defaultdict(dict)
    for row in rows:
        x=examples[row['id']]
        if x['source']=='synthetic_cf':groups[x['scenario_id']][(x['numeric_flip'],x['claim_flip'])]=row
    all_correct=both_numeric_pairs=valid_changed=0;scenario_results={}
    for scenario,cases in groups.items():
        if set(cases)!={(False,False),(False,True),(True,False),(True,True)}:raise ValueError('Incomplete quartet')
        quartet_correct=all(r['score']['exact_match']==1 for r in cases.values())
        all_correct+=quartet_correct;scenario_pairs=0
        for claim in (False,True):
            a,b=cases[False,claim],cases[True,claim]
            pair_correct=a['score']['exact_match']==b['score']['exact_match']==1
            both_numeric_pairs+=pair_correct;scenario_pairs+=pair_correct
            pa,pb=parse_answer(a['prediction']),parse_answer(b['prediction'])
            valid_changed+=(pa is not None and pb is not None and pa.scale==pb.scale=='' and
                len(pa.answer)==len(pb.answer)==1 and
                pa.answer[0].lower() in {'supported','contradicted'} and
                pb.answer[0].lower() in {'supported','contradicted'} and
                pa.answer[0].lower()!=pb.answer[0].lower())
        scenario_results[scenario]={'all_four_correct':int(quartet_correct),'numeric_flip_pair_correct_rate':scenario_pairs/2}
    n=len(groups)
    return {'scenarios':n,'cases':4*n,'all_four_correct':all_correct,'all_four_correct_rate':all_correct/n,
            'numeric_only_pairs':2*n,'both_answers_correct_for_numeric_flip':both_numeric_pairs,
            'numeric_flip_pair_correct_rate':both_numeric_pairs/(2*n),
            'valid_answer_changed_for_numeric_flip':valid_changed,
            'valid_numeric_flip_change_rate':valid_changed/(2*n),'scenario_results':scenario_results}

def mean_seed_delta_interval(records,examples,source,seeds=(42,11,23),replicates=5000,left='B1',right='N1',score_key='exact_match'):
    ids=[uid for uid,x in examples.items() if x['source']==source]
    grouped=defaultdict(list)
    for uid in ids:
        x=examples[uid]
        cluster=x['scenario_id'] if source=='synthetic_cf' else x['document_id'] if source=='finqa' else x['context_id']
        grouped[cluster].append(uid)
    by_label={label:{r['id']:r['score'].get(score_key,r['score']['exact_match']) for r in rows} for label,rows in records.items()}
    def difference(seed,uids):
        return sum(by_label[f'{right}_seed{seed}'][u]-by_label[f'{left}_seed{seed}'][u] for u in uids)/len(uids)
    deltas=[difference(s,ids) for s in seeds]
    rng=random.Random(2026);keys=sorted(grouped);samples=[]
    for _ in range(replicates):
        sampled_ids=[u for _ in keys for u in grouped[rng.choice(keys)]]
        sampled_seeds=[rng.choice(seeds) for _ in seeds]
        samples.append(statistics.mean(difference(s,sampled_ids) for s in sampled_seeds))
    samples.sort()
    return {'n_examples':len(ids),'clusters':len(keys),'seeds':list(seeds),'seed_deltas':deltas,
            'mean_paired_delta':statistics.mean(deltas),'sample_sd_across_seed_deltas':statistics.stdev(deltas),
            'descriptive_seed_and_context_bootstrap_95_interval':[samples[int(.025*replicates)],samples[int(.975*replicates)-1]],
            'bootstrap_replicates':replicates,'bootstrap_seed':2026,
            'interpretation':'Conditional on this selected data and three seeds; only three training seeds, no general population or multiple-testing guarantee.'}

def analyze(root,data):
    root=Path(root);examples={x['id']:x for x in map(json.loads,Path(data).read_text().splitlines())}
    records={};summaries={};cf={}
    for label in ['B0']+[f'{mode}_seed{seed}' for seed in (42,11,23) for mode in ('B1','N1','P1')]:
        folder=root/label
        rows=[json.loads(x) for x in (folder/'actual.jsonl').read_text().splitlines()]
        if len(rows)!=len(examples) or {x['id'] for x in rows}!=set(examples):raise ValueError('Incomplete predictions: '+label)
        records[label]=rows;summaries[label]=json.loads((folder/'actual.summary.json').read_text())
        cf[label]=counterfactual_summary(rows,examples)
    result={'status':'FROZEN_RESTRICTED_TEST_RESULTS','summaries':summaries,'counterfactuals':cf,
            'mean_seed_fusion_minus_text':{s:mean_seed_delta_interval(records,examples,s) for s in ('finqa','tatqa','synthetic_cf')},
            'secondary_prefix_comparisons':{right+'_minus_'+left:{s:mean_seed_delta_interval(records,examples,s,left=left,right=right) for s in ('finqa','tatqa','synthetic_cf')} for left,right in [('B1','P1'),('P1','N1')]},
            'secondary_finqa_unit_canonical':mean_seed_delta_interval(records,examples,'finqa',score_key='exact_match_unit_canonical'),
            'generation_limit_hits':{label:{source:sum(r['output_tokens']>=64 for r in rows if r['score']['source']==source) for source in ('finqa','tatqa','synthetic_cf')} for label,rows in records.items()},
            'selection':'All three predetermined training seeds; no test-based checkpoint/seed selection.',
            'limitations':['Restricted arithmetic subset, not full official benchmark.',
                           'Three seeds trained on the same 256 questions for one epoch.',
                           'TAT-QA context clusters do not establish source-document disjointness.',
                           'Synthetic counterfactual labels are deterministic; human realism/coverage is not established.',
                           'Public-data pretraining contamination is unknown.']}
    scenario_examples={uid:{'id':uid,'source':'synthetic_cf','scenario_id':uid} for uid in cf['B0']['scenario_results']}
    result['counterfactual_paired_scenario_metrics']={}
    for metric in ('all_four_correct','numeric_flip_pair_correct_rate'):
        scenario_records={label:[{'id':uid,'score':{'exact_match':float(values[metric])}} for uid,values in item['scenario_results'].items()] for label,item in cf.items()}
        result['counterfactual_paired_scenario_metrics'][metric]=mean_seed_delta_interval(scenario_records,scenario_examples,'synthetic_cf')
    (root/'final_results.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.root,a.data),indent=2))
