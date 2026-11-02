"""Reconstruct the released synthetic test suite from its 32 scenario seeds.

Data: CC0-1.0. Original implementation: MIT (see CODE_LICENSE).
No third-party corpus, model weights or network access is required.
"""
from pathlib import Path
from decimal import Decimal
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parent

def prompt(narrative,table,question):
    return ('Decide whether the management claim is supported or contradicted by the financial numbers. '
            'Operating margin is operating income divided by revenue. An increase means the current '
            'margin exceeds the prior margin; a decrease means it is lower. '
            'Return exactly one JSON object: {"answer":["supported"],"scale":""} or '
            '{"answer":["contradicted"],"scale":""}. Return no explanation.\n\n'
            'Management claim:\n'+narrative+'\n\nFinancial table:\n'+table+
            '\n\nQuestion: '+question+'\nAnswer:')

def generate():
    result=[]
    for line in (ROOT/'provenance/scenario_seeds.jsonl').read_text().splitlines():
        s=json.loads(line)
        r0,r1,p0,p1=[Decimal(s[k]) for k in ['prior_revenue','current_revenue','prior_operating_income','current_operating_income']]
        m0,m1=p0/r0,p1/r1
        assert m0!=m1 and r0>0 and r1>0
        flipped=((m0/2 if m1>m0 else m0*Decimal('1.5'))*r1).quantize(Decimal('.001'))
        for nf in (False,True):
            for cf in (False,True):
                current=flipped if nf else p1
                claim=bool(s['original_claim_increased'])^cf
                table='\n'.join(['Metric (million USD) | Prior period | Current period',f'Revenue | {r0:.2f} | {r1:.2f}',f'Operating income | {p0:.3f} | {current:.3f}'])
                narrative='Management states: operating margin '+('increased.' if claim else 'decreased.')
                question='Is the management claim supported or contradicted by the table?'
                text=prompt(narrative,table,question)
                display_r0,display_r1=Decimal(f'{r0:.2f}'),Decimal(f'{r1:.2f}')
                display_p0,display_p1=Decimal(f'{p0:.3f}'),Decimal(f'{current:.3f}')
                assert display_p1*display_r0!=display_p0*display_r1
                label='supported' if (display_p1*display_r0>display_p0*display_r1)==claim else 'contradicted'
                result.append({'id':f'cf-final-{s["index"]:03d}-n{int(nf)}c{int(cf)}','scenario_id':s['scenario_id'],
                    'variant':('both_flips' if cf else 'numeric_flip') if nf else ('claim_flip' if cf else 'original'),
                    'numeric_flip':nf,'claim_flip':cf,'narrative':narrative,'numeric_text':table,'question':question,
                    'gold_label':label,'prior_revenue':f'{r0:.2f}','current_revenue':f'{r1:.2f}',
                    'prior_operating_income':f'{p0:.3f}','current_operating_income':f'{current:.3f}',
                    'unit':'million USD','prompt':text,'input_sha256':hashlib.sha256(text.encode()).hexdigest()})
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    rows=generate();path=ROOT/'data/test.jsonl'
    if args.check:
        assert rows==[json.loads(l) for l in path.read_text().splitlines()]
        print('Verified all 128 released examples, exact labels and input hashes across 32 scenarios.')
    else:path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
