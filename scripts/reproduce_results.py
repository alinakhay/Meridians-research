"""Re-score every released answer and reproduce all paired analyses without model inference."""
import argparse,hashlib,json,shutil,tempfile
from pathlib import Path
from meridians.evaluation.scoring import score_example,summarize
from meridians.evaluation.evidence import research_prompt
from scripts.analyze_final import analyze
from scripts.diagnose_counterfactual_schema import diagnose
ROOT=Path(__file__).resolve().parents[1]
def reproduce(data,reference_metadata):
 study=ROOT/'results/final';config=json.loads((study/'frozen_protocol.json').read_text())
 data=Path(data);reference_metadata=Path(reference_metadata)
 for p,key in [(data,'evaluation_sha256'),(reference_metadata,'reference_units_sha256')]:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=config[key]:raise ValueError('Frozen input hash mismatch: '+p.name)
 examples={x['id']:x for x in map(json.loads,data.read_text().splitlines())};assert len(examples)==256
 refs=json.loads(reference_metadata.read_text())['examples']
 for uid,ex in examples.items():
  if ex['source']=='finqa':ex.update(refs[uid])
  assert hashlib.sha256(research_prompt(ex).encode()).hexdigest()==ex['input_sha256']
 digests=json.loads((ROOT/'provenance/prediction_text_sha256.json').read_text());count=0
 for label in config['model_order']:
  rows=[json.loads(l) for l in (study/label/'actual.jsonl').read_text().splitlines()]
  assert len(rows)==256 and {r['id'] for r in rows}==set(examples)
  for row in rows:
   assert hashlib.sha256(row['prediction'].encode()).hexdigest()==digests[label][row['id']]
   assert score_example(examples[row['id']],row['prediction'])==row['score'];count+=1
  fresh=summarize([r['score'] for r in rows]);saved=json.loads((study/label/'actual.summary.json').read_text())
  assert fresh['n']==saved['n'] and fresh['datasets']==saved['datasets']
 with tempfile.TemporaryDirectory(prefix='meridians-results-') as tmp:
  target=Path(tmp)/'final';shutil.copytree(study,target)
  assert analyze(target,data)==json.loads((study/'final_results.json').read_text())
  assert diagnose(target,data)==json.loads((study/'counterfactual_schema_diagnostic.json').read_text())
 return {'status':'ALL_RELEASED_RESULTS_REPRODUCED','predictions_rescored':count,'conditions':len(config['model_order']),'frozen_inputs_verified':True,'original_prediction_texts_verified':True,'strict_and_secondary_analyses_reproduced':True}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/final_v1/evaluation.jsonl');p.add_argument('--reference-metadata',type=Path,default=ROOT/'data/final_v1/reference_units.json');a=p.parse_args();print(json.dumps(reproduce(a.data,a.reference_metadata),indent=2))
