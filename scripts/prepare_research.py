"""Freeze an input-complete, development-only arithmetic comparison.

Selection uses source, split, input token/cell bounds, task type and seeded ID
hashes. No gold value, model score, gold evidence index, or final-test record is
used to select contexts. Exclusions are recorded rather than silently truncated.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from meridians.evaluation.evidence import extract_numeric_rows, research_prompt
from meridians.evaluation.scoring import canonical_target, score_example

ROOT=Path(__file__).resolve().parents[1]


def load_examples(source, split):
    raw=json.loads((ROOT/'data/public'/source/(split+'.json')).read_text())
    if source=='finqa':
        for row in raw:
            qa=row['qa']
            table=row['table']
            narrative='\n'.join(row.get('pre_text',[])+row.get('post_text',[]))
            numeric,meta=extract_numeric_rows(table,narrative)
            uid=row['id']
            yield {'id':'finqa-'+split+'-'+uid,'benchmark_id':uid,'source':source,'split':split,
                   'document_id':uid.rsplit('/',1)[0],'context_id':uid.rsplit('-',1)[0],
                   'narrative':narrative,'numeric_text':'\n'.join(' | '.join(map(str,r)) for r in table),
                   'numeric_rows':numeric,'numeric_metadata':meta,'question':qa['question'],
                   'gold_answer':qa['exe_ans'],'answer_type':'arithmetic' if isinstance(qa['exe_ans'],(int,float)) else 'boolean',
                   'scale':'','reference_program':qa['program'],'research_format':'json_answer_v1'}
    else:
        for context in raw:
            table=context['table']['table']
            narrative='\n'.join(p['text'] for p in sorted(context.get('paragraphs',[]),key=lambda p:p.get('order',0)))
            numeric,meta=extract_numeric_rows(table,narrative)
            for qa in context['questions']:
                yield {'id':'tatqa-'+split+'-'+qa['uid'],'benchmark_id':qa['uid'],'source':source,'split':split,
                       'document_id':None,'context_id':context['table']['uid'],
                       'narrative':narrative,'numeric_text':'\n'.join(' | '.join(map(str,r)) for r in table),
                       'numeric_rows':numeric,'numeric_metadata':meta,'question':qa['question'],
                       'gold_answer':qa['answer'],'answer_type':qa['answer_type'],'scale':qa['scale'],
                       'research_format':'json_answer_v1'}


def token_ids(tok,prompt):
    return tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                                    add_generation_prompt=True,enable_thinking=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--max-prompt-tokens',type=int,default=1920)
    p.add_argument('--max-answer-tokens',type=int,default=128)
    p.add_argument('--max-numeric-rows',type=int,default=128)
    p.add_argument('--train-per-source',type=int,default=128)
    p.add_argument('--dev-per-source',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    output=ROOT/'data/research_v1'
    output.mkdir(exist_ok=True)
    audit={'protocol':'full_context_arithmetic_dev_pilot_v1','seed':a.seed,
           'max_prompt_tokens':a.max_prompt_tokens,'max_answer_tokens':a.max_answer_tokens,
           'max_numeric_rows':a.max_numeric_rows,'datasets':{},'files':{},
           'test_split_used':False,'selection_policy':'Seeded ID hash within source; arithmetic QA with complete inputs fitting fixed limits; no retrieval and no silent truncation.'}
    selections={'train':[],'dev':[]}
    for source in ('finqa','tatqa'):
        for split in ('dev','train'):
            rows=list(load_examples(source,split))
            excluded=Counter();eligible=[];lengths=[];cell_counts=[];self_score_failures=[]
            for row in rows:
                # Official scorer roundtrip validates gold serialization; it is NOT model performance.
                score=score_example(row,canonical_target(row))
                if score['exact_match']!=1:
                    self_score_failures.append(row['id'])
                n=len(token_ids(tok,research_prompt(row)))
                na=len(tok(canonical_target(row)+tok.eos_token,add_special_tokens=False)['input_ids'])
                lengths.append(n);cell_counts.append(len(row['numeric_rows']))
                row['input_tokens']=n;row['target_tokens']=na
                # FinQA categorical yes/no is kept for later, not mixed into this arithmetic pilot.
                reasons=[]
                if row['answer_type']!='arithmetic':reasons.append('non_arithmetic')
                if n>a.max_prompt_tokens:reasons.append('prompt_over_limit')
                if na>a.max_answer_tokens:reasons.append('answer_over_limit')
                if len(row['numeric_rows'])>a.max_numeric_rows:reasons.append('numeric_cells_over_limit')
                if not row['numeric_rows']:reasons.append('no_structured_numeric_cells')
                for reason in reasons:excluded[reason]+=1
                if not reasons:
                    row['input_sha256']=hashlib.sha256(research_prompt(row).encode()).hexdigest()
                    eligible.append(row)
            # Exclude FinQA training report groups present in any official dev record.
            if source=='finqa' and split=='train':
                dev_groups={r['document_id'] for r in load_examples(source,'dev')}
                filtered=[r for r in eligible if r['document_id'] not in dev_groups]
                excluded['training_report_group_in_dev']=len(eligible)-len(filtered)
                eligible=filtered
            nselect=a.train_per_source if split=='train' else a.dev_per_source
            eligible.sort(key=lambda r:hashlib.sha256(f'{a.seed}:{r["id"]}'.encode()).hexdigest())
            if len(eligible)<nselect:
                raise RuntimeError(f'Insufficient eligible cases: {source}/{split} {len(eligible)} < {nselect}')
            selected=eligible[:nselect]
            selections[split]+=selected
            ordered=sorted(lengths)
            audit['datasets'][source+'_'+split]={'source_n':len(rows),'eligible_n':len(eligible),'selected_n':len(selected),
                'exclusion_counts_nonexclusive':dict(excluded),'max_input_tokens':max(lengths),
                'median_input_tokens':ordered[len(ordered)//2],
                'p95_input_tokens':ordered[int(.95*(len(ordered)-1))],
                'max_numeric_cells':max(cell_counts),'gold_serialization_self_check_failures':self_score_failures}
            if self_score_failures:
                raise RuntimeError(f'Gold serialization/scoring mismatch: {source}/{split}: {self_score_failures[:5]}')
    for split,rows in selections.items():
        rows.sort(key=lambda r:hashlib.sha256(f'{a.seed}:mix:{r["id"]}'.encode()).hexdigest())
        dest=output/(split+'.jsonl')
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        audit['files'][split]={'n':len(rows),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                               'ids':[r['id'] for r in rows]}
    train_groups={r['document_id'] for r in selections['train'] if r['source']=='finqa'}
    dev_groups={r['document_id'] for r in selections['dev'] if r['source']=='finqa'}
    assert not train_groups&dev_groups
    train_contexts={(r['source'],r['context_id']) for r in selections['train']}
    dev_contexts={(r['source'],r['context_id']) for r in selections['dev']}
    assert not train_contexts&dev_contexts
    audit['finqa_train_dev_report_group_overlap']=0
    audit['train_dev_context_overlap']=0
    audit['limitations']=['Input-complete short-context arithmetic subset, not full official benchmark. Exclusion bias must be reported.',
        'FinQA report groups are company/year proxies. TAT-QA source-document IDs are unavailable; only context IDs are disjoint.',
        'Public data may have appeared in backbone pretraining; adaptation split controls do not establish pretraining decontamination.',
        'One development seed; no confirmatory test or significance claim.']
    (output/'manifest.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='files'},indent=2))

if __name__=='__main__':main()
