"""Meridians research adapter inference. Generated from audited source definitions.
See INFERENCE_SOURCE_COMPONENTS.json for provenance; original code is MIT.
"""
from __future__ import annotations
import argparse,json,hashlib,math,re
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from peft import PeftModel

# From meridians/numeric.py: rows_to_tensor
def rows_to_tensor(examples:list[dict], max_rows:int=48,device=None):
    b=len(examples)
    vals=torch.zeros(b,max_rows,3,dtype=torch.float32,device=device)
    metric=torch.zeros(b,max_rows,dtype=torch.long,device=device)
    period=torch.zeros(b,max_rows,dtype=torch.long,device=device)
    units=torch.zeros(b,max_rows,dtype=torch.long,device=device)
    mask=torch.zeros(b,max_rows,dtype=torch.bool,device=device)
    for i,ex in enumerate(examples):
        if len(ex.get('numeric_rows',[]))>max_rows:
            raise ValueError('Numeric evidence exceeds configured rows; silent truncation is disabled')
        for j,r in enumerate(ex.get('numeric_rows',[])):
            x=float(r['value'])
            # Stable numeric features: sign, log magnitude, bounded fractional part
            vals[i,j]=torch.tensor([math.copysign(1,x) if x else 0,math.log1p(abs(x))/12.,
                    (abs(x)%1.0)],device=device)
            metric[i,j]=max(0,min(int(r.get('metric_id',0)),511))
            period[i,j]=max(0,min(int(r.get('period',0)),63))
            units[i,j]=max(0,min(int(r.get('unit_id',0)),15))
            mask[i,j]=True
    return {'values':vals,'metric':metric,'period':period,'units':units,'mask':mask}

# From meridians/numeric.py: NumericEncoder
class NumericEncoder(nn.Module):
    """Set-aware numeric encoder producing K learnable pseudo-tokens in LLM embedding space."""
    def __init__(self, out_dim:int, hidden:int=128, num_tokens:int=4):
        super().__init__()
        self.value=nn.Sequential(nn.Linear(3,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.metric=nn.Embedding(512,hidden)
        self.period=nn.Embedding(64,hidden)
        self.unit=nn.Embedding(16,hidden)
        self.norm=nn.LayerNorm(hidden)
        self.queries=nn.Parameter(torch.randn(num_tokens,hidden)*0.02)
        self.attn=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.project=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,out_dim))
        # Match the approximate scale of pretrained token embeddings at startup.
        nn.init.normal_(self.project[-1].weight,std=0.02/math.sqrt(hidden))
        nn.init.zeros_(self.project[-1].bias)
        self.num_tokens=num_tokens

    def forward(self, numeric):
        mask=numeric['mask']; b=mask.shape[0]
        # Never attend to empty padding slots, but allow an empty numeric input.
        safe_mask=mask.clone()
        empty=~safe_mask.any(dim=1)
        if empty.any():safe_mask[empty,0]=True
        z=(self.value(numeric['values'])+self.metric(numeric['metric'])+
           self.period(numeric['period'])+self.unit(numeric['units']))
        z=self.norm(z)
        q=self.queries.unsqueeze(0).expand(b,-1,-1)
        pooled,_=self.attn(q,z,z,key_padding_mask=~safe_mask,need_weights=False)
        pooled[empty]=0
        return self.project(pooled)

# From meridians/numeric.py: LearnedPrefix
class LearnedPrefix(nn.Module):
    """A learned input prefix independent of every value, row, period and unit."""
    def __init__(self,out_dim:int,num_tokens:int=4):
        super().__init__()
        self.prefix=nn.Parameter(torch.randn(num_tokens,out_dim)*0.02)
        self.num_tokens=num_tokens

    def forward(self,numeric):
        return self.prefix.unsqueeze(0).expand(numeric['mask'].shape[0],-1,-1)

# From meridians/modeling.py: MeridiansLM
class MeridiansLM(nn.Module):
    def __init__(self,base_lm:nn.Module,numeric_encoder:NumericEncoder|None=None):
        super().__init__()
        self.base_lm=base_lm
        self.numeric_encoder=numeric_encoder

    def forward(self,input_ids,attention_mask,labels=None,numeric=None):
        if self.numeric_encoder is None:
            return self.base_lm(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        emb=self.base_lm.get_input_embeddings()(input_ids)
        prefix=self.numeric_encoder(numeric).to(emb.dtype)
        b,k,_=prefix.shape
        embs=torch.cat((prefix,emb),dim=1)
        padded_mask=torch.cat([attention_mask.new_ones((b,k)),attention_mask],dim=1)
        if labels is not None:
            pad_labels=labels.new_full((b,k),-100)
            labels=torch.cat([pad_labels,labels],dim=1)
        return self.base_lm(inputs_embeds=embs,attention_mask=padded_mask,labels=labels)

# From meridians/evaluation/evidence.py: UNIT_IDS
UNIT_IDS = {'unspecified': 0, 'USD_m': 1, 'percent': 2, 'USD': 3,
            'thousand': 4, 'million': 5, 'billion': 6, 'GBP': 7, 'EUR': 8,
            'JPY': 9, 'count': 10, 'USD_thousand': 11, 'USD_billion': 12,
            'GBP_million': 13, 'EUR_million': 14, 'ratio': 15}

# From meridians/evaluation/evidence.py: parse_cell
def parse_cell(text) -> tuple[float, bool, str] | None:
    text = str(text).strip().replace('\u2212', '-').replace('\u2013', '-')
    if not text or text in {'-', '--', 'n/a', 'N/A', 'NM', 'nm'}:
        return None
    currency = next((code for symbol, code in [('$', 'USD'), ('£', 'GBP'), ('€', 'EUR'), ('¥', 'JPY')]
                     if symbol in text), '')
    text = re.sub(r'[$£€¥\s,]', '', text)
    percent = text.endswith('%')
    if percent:
        text = text[:-1]
    negative = text.startswith('(') and text.endswith(')')
    if negative:
        text = text[1:-1]
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text):
        return None
    value = float(text)
    if negative:
        value = -value
    return (value, percent, currency) if math.isfinite(value) else None

# From meridians/evaluation/evidence.py: extract_numeric_rows
def extract_numeric_rows(table: list, narrative: str = '') -> tuple[list[dict], dict]:
    """Retain numeric cells with source coordinates, conservative units and periods.

    Row-position IDs avoid the old cross-label hash collisions. Column years are
    interpreted only in header rows; unknown chronology stays a column position.
    No automatic numeric-cell truncation takes place here.
    """
    header_indices = []
    years = {}
    for ri, row in enumerate(table[:3]):
        cells = [str(x).strip() for x in row[1:] if str(x).strip()]
        parsed = [parse_cell(x) for x in cells]
        numeric = [x[0] for x in parsed if x is not None]
        year_header = bool(cells) and all(re.search(r'\b(?:19|20)\d{2}\b', x) for x in cells)
        nonnumeric_header = ri == 0 and not numeric
        if year_header or nonnumeric_header:
            header_indices.append(ri)
            for ci, cell in enumerate(row[1:]):
                found = re.findall(r'\b(?:19|20)\d{2}\b', str(cell))
                if found:
                    years[ci] = int(found[-1])
        elif numeric:
            break
    scope = ' '.join(str(cell) for row in table[:3] for cell in row) + ' ' + narrative
    scales = set(re.findall(r'\bin\s+(thousands|millions|billions)\b', scope.lower()))
    scale = next(iter(scales))[:-1] if len(scales) == 1 else ''
    min_year = min(years.values()) if years else None
    rows = []
    for ri, row in enumerate(table):
        if ri in header_indices:
            continue
        if ri >= 511:
            raise ValueError('Table row position exceeds encoder vocabulary')
        metric = str(row[0]).strip() if row else ''
        for ci, cell in enumerate(row[1:]):
            parsed = parse_cell(cell)
            if parsed is None:
                continue
            value, percent, currency = parsed
            per_share = bool(re.search(r'per[ -]share|\beps\b', metric.lower()))
            if percent or '%' in metric:
                unit = 'percent'
            elif per_share:
                unit = currency or 'unspecified'
            elif currency == 'USD' and scale:
                unit = {'million': 'USD_m', 'thousand': 'USD_thousand', 'billion': 'USD_billion'}[scale]
            elif currency in {'GBP', 'EUR'} and scale == 'million':
                unit = currency + '_million'
            else:
                unit = scale or currency or 'unspecified'
            year = years.get(ci)
            period = year - min_year if year is not None and min_year is not None else ci
            if not 0 <= period <= 63:
                period = ci
                year = None
            rows.append({'metric': metric, 'metric_id': ri + 1, 'period': period,
                         'value': value, 'unit': unit, 'unit_id': UNIT_IDS[unit],
                         'table_row': ri, 'table_column': ci + 1,
                         'column_year': year, 'period_type': 'relative_year' if year is not None else 'column_position',
                         'raw_cell': str(cell)})
    return rows, {'header_rows': header_indices, 'column_years': years,
                  'unambiguous_table_scale': scale, 'numeric_cell_count': len(rows),
                  'schema': 'coordinate_typed_v2'}

# From meridians/evaluation/evidence.py: research_prompt
def research_prompt(example: dict) -> str:
    if example['source']=='synthetic_cf':
        return ('Decide whether the management claim is supported or contradicted by the financial numbers. '
                'Operating margin is operating income divided by revenue. An increase means the current '
                'margin exceeds the prior margin; a decrease means it is lower. '
                'Return exactly one JSON object: {"answer":["supported"],"scale":""} or '
                '{"answer":["contradicted"],"scale":""}. Return no explanation.\n\n'
                'Management claim:\n'+example['narrative']+'\n\nFinancial table:\n'+
                example['numeric_text']+'\n\nQuestion: '+example['question']+'\nAnswer:')
    rule = ('For FinQA, give the execution-result number: represent percentages as fractions '
            '(12% becomes 0.12). Keep other values in the question\'s stated units. Use an empty scale.'
            if example['source'] == 'finqa' else
            'For TAT-QA, give the answer in its displayed units and specify scale as '
            '"", "thousand", "million", "billion", or "percent". For example 12% is answer "12" with scale "percent".')
    if example.get('research_format')=='json_answer_v2' and example['source']=='finqa':
        rule=('For FinQA, give the execution-result number: represent percentages as fractions '
              '(12% becomes 0.12). Keep other values in the question\'s stated units. '
              'The scale must be the empty string, written literally as "scale":"". '
              'Do not write the word "empty". Example format: {"answer":["0.12"],"scale":""}.')
    return ('Answer the financial question using only the supplied document and table. '
            'Return exactly one JSON object with two keys: "answer" (a list of answer strings) '
            'and "scale" (a string). Return no explanation. ' + rule + '\n\n'
            'Financial document:\n' + example['narrative'] + '\n\nFinancial table:\n' +
            example['numeric_text'] + '\n\nQuestion: ' + example['question'] + '\nAnswer:')

# From scripts/train_llm.py: format_ids
def format_ids(tok, prompt, max_length):
    """Qwen3 chat tokens; disable free-running thinking for measurable short answers."""
    try:
        ids=tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=True,
                                    add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        ids=tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=True,
                                    add_generation_prompt=True)
    if len(ids)>max_length:
        raise ValueError(f'Prompt has {len(ids)} tokens, exceeding {max_length}; silent truncation is disabled')
    return ids

# From scripts/evaluate_llm.py: load_model
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

# From scripts/evaluate_research.py: numeric_input
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

# From scripts/evaluate_research.py: generate
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

# From scripts/predict_financial.py: main
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--input',required=True,help='JSON with source, narrative, table (2D cells), and question')
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('This tested four-bit inference recipe requires CUDA')
    x=json.loads(Path(a.input).read_text());source=x.get('source','finqa')
    if source not in {'finqa','tatqa'}:raise ValueError('source must be finqa or tatqa for the specified answer/scale convention')
    rows,meta=extract_numeric_rows(x['table'],x['narrative'])
    example={'id':'user-example','source':source,'research_format':'json_answer_v2',
             'narrative':x['narrative'],'question':x['question'],
             'numeric_text':'\n'.join(' | '.join(map(str,r)) for r in x['table']),
             'numeric_rows':rows,'numeric_metadata':meta}
    example['input_sha256']=hashlib.sha256(research_prompt(example).encode()).hexdigest()
    model,tok,manifest=load_model(Path(a.model),'cuda')
    model.base_lm.generation_config=GenerationConfig(do_sample=False,eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id,use_cache=True)
    text,ninput,noutput=generate(example,model,tok,1920,64,manifest['max_numeric_rows'],'actual')
    print(json.dumps({'prediction':text,'input_tokens':ninput,'output_tokens':noutput},ensure_ascii=False))

if __name__=='__main__':main()
