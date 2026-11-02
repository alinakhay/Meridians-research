"""Evaluate a trained checkpoint: classification on the synthetic claim-verification task.

This script requires the adapter created by train_llm.py and a CUDA-enabled machine for a large checkpoint.
Public financial QA uses evaluate_research.py and its validated task-specific scorers.
"""
import argparse,json,re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from meridians.data import load_jsonl
from meridians.numeric import NumericEncoder,LearnedPrefix,rows_to_tensor
from meridians.modeling import MeridiansLM
from scripts.train_llm import build_prompt,format_ids

def parse_label(text):
    m=re.search(r'\b(supported|contradicted)\b',text.lower())
    return m.group(1) if m else None

def load_model(root,device):
    cfg=json.loads((root/'training_manifest.json').read_text())
    from transformers import BitsAndBytesConfig
    q4=cfg.get('quantization')=='4bit' and device=='cuda'
    quant_cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if q4 else None
    base=AutoModelForCausalLM.from_pretrained(cfg['base_model'],revision=cfg.get('base_model_revision'),torch_dtype=torch.float32 if device=='cpu' else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,quantization_config=quant_cfg,device_map={'':0} if q4 else None)
    base.config.use_cache=True
    adapter_dir=root if (root/'adapter_config.json').exists() else root/'lora_adapter'
    tokenizer_dir=root if (root/'tokenizer_config.json').exists() else root/'tokenizer'
    lm=PeftModel.from_pretrained(base,str(adapter_dir))
    if not q4: lm=lm.to(device)
    lm.eval()
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    encoder=None
    if cfg['mode'] in {'fusion','prefix'}:
        cls=NumericEncoder if cfg['mode']=='fusion' else LearnedPrefix
        encoder=cls(lm.get_input_embeddings().embedding_dim)
        encoder.load_state_dict(torch.load(root/'numeric_encoder.pt',map_location='cpu',weights_only=True))
        encoder.to(device).eval()
    return MeridiansLM(lm,encoder).eval(),tokenizer,cfg

def prediction(x,model,tok,device,max_new_tokens,include_numeric_text=True):
    prompt=build_prompt(x,include_numeric_text=include_numeric_text)
    token_ids=format_ids(tok,prompt,512)
    ids=torch.tensor([token_ids],device=device)
    attn=torch.ones_like(ids)
    args={'max_new_tokens':max_new_tokens,'do_sample':False,'pad_token_id':tok.eos_token_id}
    with torch.inference_mode():
        if model.numeric_encoder:
            numbers=rows_to_tensor([x],device=device)
            prefix=model.numeric_encoder(numbers).to(model.base_lm.get_input_embeddings().weight.dtype)
            embs=torch.cat([prefix,model.base_lm.get_input_embeddings()(ids)],dim=1)
            allmask=torch.cat([torch.ones((1,prefix.shape[1]),dtype=attn.dtype,device=device),attn],dim=1)
            result=model.base_lm.generate(inputs_embeds=embs,attention_mask=allmask,**args)
            generated=result[0]
        else:
            result=model.base_lm.generate(input_ids=ids,attention_mask=attn,**args)
            generated=result[0,ids.shape[-1]:]
    return tok.decode(generated,skip_special_tokens=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--data',default='data/synthetic/test.jsonl')
    p.add_argument('--output',default='results/llm_test_predictions.jsonl')
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--max-new-tokens',type=int,default=64)
    a=p.parse_args();device='cuda' if torch.cuda.is_available() else 'cpu'
    root=Path(a.model);model,tok,cfg=load_model(root,device)
    dataset=[x for x in load_jsonl(a.data) if x.get('task_type')=='claim_verification']
    if a.limit is not None:dataset=dataset[:a.limit]
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    ncorrect=0; nparsed=0
    with output.open('w') as f:
        for x in dataset:
            text=prediction(x,model,tok,device,a.max_new_tokens,include_numeric_text=cfg.get('include_numeric_text',True))
            label=parse_label(text)
            nparsed+=label is not None
            ncorrect+=label==x['answer_label']
            f.write(json.dumps({'id':x['id'],'prediction':text,'parsed_label':label,'gold':x['answer_label']})+'\n')
    report={'n':len(dataset),'accuracy':ncorrect/len(dataset) if dataset else None,
            'parse_rate':nparsed/len(dataset) if dataset else None,
            'model':str(root),'dataset':a.data,'evaluation':'synthetic classification only'}
    output.with_suffix('.summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
