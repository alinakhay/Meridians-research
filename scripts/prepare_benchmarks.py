"""Convert benchmark files to common (text, numeric evidence, answer) examples.
Important: TAT-QA contains multi-item answers. Exact-match scoring must use task-specific evaluation.
"""
import argparse,hashlib,json,re
from pathlib import Path

_number=re.compile(r'(?<![a-zA-Z])[-+]?\(?\$?\d[\d,]*(?:\.\d+)?\)?%?')

def to_float(s):
    try:
        t=str(s).replace(',','').replace('$','').strip()
        if not _number.search(t):return None
        m=_number.search(t); t=m.group(0)
        neg=t.startswith('(') and t.endswith(')')
        t=t.strip('() %')
        f=float(t);return -f if neg else f
    except (TypeError,ValueError):return None

def numeric_rows(table,max_rows=48):
    rows=[]
    if not isinstance(table,list):return rows
    for ri,line in enumerate(table):
        if not isinstance(line,(list,tuple)):continue
        metric=str(line[0]) if line else ''
        # Stable hash: not built-in Python hash (which varies by process).
        mid=int.from_bytes(hashlib.sha256(metric.encode()).digest()[:2],'big')%511 + 1
        for ci,cell in enumerate(line[1:]):
            v=to_float(cell)
            if v is not None:
                rows.append({'metric':metric[:64],'metric_id':mid,'period':ci,
                             'value':v,'unit':'unspecified','unit_id':0})
            if len(rows)>=max_rows:return rows
    return rows

def prepare_finqa(file,split):
    for i,x in enumerate(json.loads(Path(file).read_text())):
        q=x.get('qa',{})
        a=q.get('exe_ans')
        if a is None or a=='':a=q.get('answer')
        if a is None or a=='':continue
        table=x.get('table') or []
        benchmark_id=str(x.get('id',i))
        yield {'id':'finqa-'+split+'-'+str(x.get('id',i)), 'source':'finqa', 'split':split,
               'benchmark_id':benchmark_id,'document_id':benchmark_id.rsplit('/',1)[0],
               'task_type':'financial_qa','narrative':' '.join(x.get('pre_text',[])+x.get('post_text',[])),
               'numeric_text':'\n'.join(' | '.join(map(str,r)) for r in table),
               'numeric_rows':numeric_rows(table), 'question':q.get('question',''), 'answer':str(a),
               'gold_answer':a,'reference_program':q.get('program','')}

def prepare_tatqa(file,split):
    for i,x in enumerate(json.loads(Path(file).read_text())):
        table=x.get('table',{}).get('table',[])
        paragraphs=sorted(x.get('paragraphs',[]),key=lambda y:y.get('order',0))
        narrative=' '.join(str(z.get('text','')) for z in paragraphs)
        text='\n'.join(' | '.join(map(str,r)) for r in table)
        for j,qa in enumerate(x.get('questions',[])):
            if 'answer' not in qa:continue
            answer=qa['answer']
            if isinstance(answer,list):answer='; '.join(map(str,answer))
            yield {'id':f'tatqa-{split}-{i}-{j}','source':'tatqa','split':split,
                   'benchmark_id':qa.get('uid',f'{i}-{j}'),
                   'context_id':x.get('table',{}).get('uid',str(i)), 'document_id':None,
                   'task_type':'financial_qa','narrative':narrative,'numeric_text':text,
                   'numeric_rows':numeric_rows(table),'question':qa.get('question',''),
                   'answer':str(answer),'gold_answer':qa['answer'],'answer_type':qa.get('answer_type',''),
                   'scale':qa.get('scale','')}

def convert(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    counts={}
    for src in ('finqa','tatqa'):
        for split in ('train','dev','test'):
            infile=root/src/(split+'.json')
            if not infile.exists():continue
            fn=prepare_finqa if src=='finqa' else prepare_tatqa
            outfile=output/f'{src}_{split}.jsonl';n=0
            with outfile.open('w') as f:
                for x in fn(infile,split):f.write(json.dumps(x)+'\n');n+=1
            counts[f'{src}_{split}']=n
    (output/'manifest.json').write_text(json.dumps({'counts':counts,
       'description':'Derived from upstream datasets. Preserve original attribution and licenses.',
       'warning':'Some questions need scale conversion; no claims about evaluated accuracy until metric is implemented.'},indent=2))
    print(json.dumps(counts,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/public')
    p.add_argument('--output',default='data/benchmarks');a=p.parse_args();convert(a.input,a.output)
