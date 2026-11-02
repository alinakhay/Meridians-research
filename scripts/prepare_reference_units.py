"""Freeze a secondary reference-unit diagnostic before final model predictions."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from meridians.evaluation.scoring import canonical_execution_reference

ROOT=Path(__file__).resolve().parents[1]

def main():
    dest=ROOT/'data/final_v1/reference_units.json'
    if dest.exists():raise RuntimeError('Reference-unit metadata already frozen')
    examples=[json.loads(x) for x in (ROOT/'data/final_v1/evaluation.jsonl').read_text().splitlines()]
    raw={r['id']:r['qa'] for r in json.loads((ROOT/'data/public/finqa/test.json').read_text())}
    refs={}
    for row in examples:
        if row['source']!='finqa':continue
        qa=raw[row['benchmark_id']]
        value,convention=canonical_execution_reference(row['gold_answer'],qa.get('answer',''))
        refs[row['id']]={'canonical_gold_answer':value,'reference_unit_convention':convention,
                         'display_annotation':qa.get('answer','')}
    result={'status':'FROZEN_SECONDARY_REFERENCE_DIAGNOSTIC_BEFORE_TEST_PREDICTION',
       'reason':'Development-detected reference-unit inconsistency; raw primary metric is retained.',
       'policy':'Normalize a reference percentage-point value to a five-decimal fraction only when the explicit percentage annotation supports points but not fractions within recorded rounding tolerance. No prediction enters this function. Ambiguous/inconsistent references retain the raw execution target.',
       'examples':refs,'convention_counts':dict(Counter(x['reference_unit_convention'] for x in refs.values()))}
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps({'file':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'n':len(refs)},indent=2))

if __name__=='__main__':main()
