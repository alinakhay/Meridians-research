"""Bounded financial-QA dev inference with official TAT-QA and FinQA answer scoring."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,GenerationConfig
from meridians.modeling import MeridiansLM
from meridians.numeric import rows_to_tensor
from meridians.evaluation.evidence import research_prompt
from meridians.evaluation.scoring import score_example,summarize
from scripts.train_llm import format_ids
from scripts.evaluate_llm import load_model


def numeric_input(example,max_rows,condition):
    features=rows_to_tensor([example],max_rows=max_rows,device='cuda')
    n=int(features['mask'][0].sum())
    if condition=='zero_values':
        features['values'].zero_()
    elif condition=='permute_values':
        seed=int(hashlib.sha256(example['id'].encode()).hexdigest()[:8],16)
        order=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).to('cuda')
        features['values'][0,:n]=features['values'][0,:n][order].clone()
    elif condition!='actual':
        raise ValueError('Unknown numerical condition')
    return features


def generate(example,model,tok,max_prompt,max_new_tokens,max_rows,condition):
    prompt=research_prompt(example)
    assert hashlib.sha256(prompt.encode()).hexdigest()==example['input_sha256']
    token_ids=format_ids(tok,prompt,max_prompt)
    ids=torch.tensor([token_ids],device='cuda')
    attention=torch.ones_like(ids)
    with torch.inference_mode():
        if model.numeric_encoder is not None:
            features=numeric_input(example,max_rows,condition)
            prefix=model.numeric_encoder(features).to(model.base_lm.get_input_embeddings().weight.dtype)
            embeds=torch.cat([prefix,model.base_lm.get_input_embeddings()(ids)],dim=1)
            mask=torch.cat([attention.new_ones((1,prefix.shape[1])),attention],dim=1)
            output=model.base_lm.generate(inputs_embeds=embeds,attention_mask=mask,max_new_tokens=max_new_tokens)
            generated=output[0]
        else:
            if condition!='actual':raise ValueError('Numerical ablations require a fusion checkpoint')
            output=model.base_lm.generate(input_ids=ids,attention_mask=attention,max_new_tokens=max_new_tokens)
            generated=output[0,ids.shape[-1]:]
    return tok.decode(generated,skip_special_tokens=True),len(token_ids),len(generated)



def verified_resume_prefix(data,prediction_file,condition):
    """Accept only a byte-preserved, scored prefix of the same frozen input order."""
    path=Path(prediction_file)
    previous=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if len(previous)>len(data) or [r['id'] for r in previous]!=[ex['id'] for ex in data[:len(previous)]]:
        raise ValueError('Resume records are not an exact input-order prefix')
    for row,example in zip(previous,data):
        if row['input_sha256']!=example['input_sha256'] or row['condition']!=condition or row['score']!=score_example(example,row['prediction']):
            raise ValueError('Resume record verification failed')
    return previous


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,help='base or a saved checkpoint directory')
    p.add_argument('--base',default='Qwen/Qwen3-8B')
    p.add_argument('--revision',required=True)
    p.add_argument('--data',default='data/research_v1/dev.jsonl')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--conditions',default='actual')
    p.add_argument('--max-prompt-tokens',type=int,default=1920)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--max-numeric-rows',type=int,default=128)
    p.add_argument('--evaluation-protocol',help='Required frozen protocol for final_v1 test inference')
    p.add_argument('--reference-metadata',help='Frozen secondary reference-unit metadata; never enters prompts')
    p.add_argument('--loader-script',help='Frozen standalone loader used to verify the exact Hub release path')
    p.add_argument('--resume-predictions',action='store_true',help='Append only missing frozen final examples after an audited operational interruption')
    a=p.parse_args()
    if a.resume_predictions and not a.evaluation_protocol:raise ValueError('Resumption requires the frozen final protocol')
    if not torch.cuda.is_available():raise RuntimeError('This 8B inference recipe requires the authorized CUDA worker')
    data=[json.loads(x) for x in Path(a.data).read_text().splitlines()]
    if not data:raise ValueError('Empty evaluation file')
    if a.evaluation_protocol:
        protocol=json.loads(Path(a.evaluation_protocol).read_text())
        if protocol['status']!='FROZEN_BEFORE_TEST_PREDICTION' or any(x['split']!='test' for x in data):
            raise ValueError('Invalid final-evaluation protocol')
        if hashlib.sha256(Path(a.data).read_bytes()).hexdigest()!=protocol['evaluation_sha256']:
            raise ValueError('Final data changed after freeze')
        if a.max_new_tokens!=protocol['max_answer_tokens'] or a.max_prompt_tokens!=protocol['max_prompt_tokens']:
            raise ValueError('Final decoding limits changed')
    elif any(x['split']!='dev' for x in data):
        raise ValueError('Test inference requires an explicit frozen final protocol')
    if a.reference_metadata:
        if not a.evaluation_protocol:raise ValueError('Reference diagnostic requires the final protocol')
        if hashlib.sha256(Path(a.reference_metadata).read_bytes()).hexdigest()!=protocol['reference_units_sha256']:
            raise ValueError('Reference-unit metadata changed')
        refs=json.loads(Path(a.reference_metadata).read_text())['examples']
        if set(refs)!={x['id'] for x in data if x['source']=='finqa'}:raise ValueError('Incomplete reference metadata')
        for example in data:
            if example['source']=='finqa':example.update(refs[example['id']])
    if a.model=='base':
        tok=AutoTokenizer.from_pretrained(a.base,revision=a.revision)
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
               bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        lm=AutoModelForCausalLM.from_pretrained(a.base,revision=a.revision,
               torch_dtype=dtype,quantization_config=quant,device_map={'':0})
        model=MeridiansLM(lm).eval()
    else:
        loader=load_model
        if a.loader_script:
            import importlib.util
            if not a.evaluation_protocol or hashlib.sha256(Path(a.loader_script).read_bytes()).hexdigest()!=protocol['release_loader_sha256']:
                raise ValueError('Release loader differs from the frozen protocol')
            spec=importlib.util.spec_from_file_location('meridians_release_loader',a.loader_script)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            loader=module.load_model
        model,tok,cfg=loader(Path(a.model),'cuda')
        if cfg['base_model_revision']!=a.revision:raise ValueError('Backbone revision mismatch')
        if cfg.get('max_numeric_rows',48)!=a.max_numeric_rows:raise ValueError('Numeric-row limit mismatch')
    model.base_lm.config.use_cache=True
    model.base_lm.generation_config=GenerationConfig(do_sample=False,eos_token_id=tok.eos_token_id,
                   pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id,use_cache=True)
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    summaries={}
    for condition in a.conditions.split(','):
        scores=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        prediction_file=output/(condition+'.jsonl');previous=[]
        preserved_sha=None
        if a.resume_predictions and prediction_file.exists():
            preserved_sha=hashlib.sha256(prediction_file.read_bytes()).hexdigest()
            previous=verified_resume_prefix(data,prediction_file,condition)
            scores.extend(row['score'] for row in previous)
        with prediction_file.open('a' if a.resume_predictions else 'x') as f:
            for i,example in enumerate(data[len(previous):],start=len(previous)):
                begin=time.monotonic()
                text,ninput,noutput=generate(example,model,tok,a.max_prompt_tokens,a.max_new_tokens,a.max_numeric_rows,condition)
                score=score_example(example,text);scores.append(score)
                record={'id':example['id'],'condition':condition,'prediction':text,'input_tokens':ninput,
                        'output_tokens':noutput,'elapsed_seconds':round(time.monotonic()-begin,4),
                        'input_sha256':example['input_sha256'],'score':score}
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                if (i+1)%16==0:print(json.dumps({'condition':condition,'completed':i+1,'total':len(data)}),flush=True)
        report=summarize(scores)
        report.update(condition=condition,model=a.model,elapsed_seconds=round(time.monotonic()-start,3),
                      max_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      evaluation_file_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest())
        if previous:
            report['resumption']={'preserved_predictions':len(previous),'preserved_file_sha256':preserved_sha,'new_predictions':len(data)-len(previous),'new_inference_elapsed_seconds':report['elapsed_seconds'],'preserved_per_example_seconds':sum(r['elapsed_seconds'] for r in previous),'note':'Peak memory measures the resumed process only; elapsed adds original saved per-example durations to new inference time.'}
            report['elapsed_seconds']=round(report['elapsed_seconds']+report['resumption']['preserved_per_example_seconds'],3)
        if not a.evaluation_protocol:report['dev_file_sha256']=report['evaluation_file_sha256']
        (output/(condition+'.summary.json')).write_text(json.dumps(report,indent=2))
        summaries[condition]=report
        print(json.dumps(report,indent=2),flush=True)
    (output/'all_summaries.json').write_text(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
