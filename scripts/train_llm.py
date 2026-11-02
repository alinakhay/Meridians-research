"""GPU-backed fine-tuning of real pretrained LM. The runner requires downloading external weights."""
import argparse, json, os, random, time, hashlib
from importlib.metadata import version
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from meridians.data import load_jsonl
from meridians.numeric import NumericEncoder, LearnedPrefix, rows_to_tensor
from meridians.modeling import MeridiansLM

def build_prompt(x, include_numeric_text=True):
    if x.get('research_format') == 'json_answer_v1':
        from meridians.evaluation.evidence import research_prompt
        if not include_numeric_text:
            raise ValueError('Numeric-only public QA ablation needs a separately frozen masking protocol')
        return research_prompt(x)
    numeric_field = x['numeric_text'] if include_numeric_text else '[Numerical evidence supplied through separate numeric encoder]'
    if x.get('task_type') == 'financial_qa':
        return f"Answer the financial question from the evidence, using explicit units where appropriate. If arithmetic is required, calculate carefully.\n\nFinancial document: {x['narrative']}\nFinancial table: {numeric_field}\nQuestion: {x['question']}\nAnswer:"
    return f"You are an evidence-grounded financial analyst. Decide if the management statement agrees with the supplied financial metrics. Compute operating margin as operating income divided by revenue. Do not assume a management claim is true.\n\nStatement: {x['narrative']}\nFinancial metrics: {numeric_field}\nQuestion: {x['question']}\nAnswer:"

def format_ids(tok, prompt, max_length):
    """Qwen3 chat tokens; disable free-running thinking for measurable short answers."""
    try:
        ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
    if len(ids) > max_length:
        raise ValueError(f'Prompt has {len(ids)} tokens, exceeding {max_length}; silent truncation is disabled')
    return ids

class FinDS(Dataset):

    def __init__(self, path, tok, max_tokens, include_numeric_text=True):
        self.data = list(load_jsonl(path))
        self.tok = tok
        self.max_tokens = max_tokens
        self.include_numeric_text = include_numeric_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        if x.get('research_format') == 'json_answer_v1':
            from meridians.evaluation.scoring import canonical_target
            target = canonical_target(x)
        else:
            target = x['answer']
        pref = build_prompt(x, self.include_numeric_text)
        answer = target + self.tok.eos_token
        a = self.tok(answer, add_special_tokens=False)['input_ids']
        p = format_ids(self.tok, pref, self.max_tokens - len(a))
        full = (p + a)[:self.max_tokens]
        labels = [-100] * len(p) + a
        return {'ids': full, 'labels': labels[:len(full)], 'example': x}

def collate(items, pad):
    length = max((len(x['ids']) for x in items))
    ids = []
    labels = []
    mask = []
    for x in items:
        n = length - len(x['ids'])
        ids.append(x['ids'] + [pad] * n)
        labels.append(x['labels'] + [-100] * n)
        mask.append([1] * len(x['ids']) + [0] * n)
    return (torch.tensor(ids), torch.tensor(mask), torch.tensor(labels), [x['example'] for x in items])

def run():
    launch_started = time.monotonic()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='Qwen/Qwen3-8B')
    p.add_argument('--revision', default=None)
    p.add_argument('--mode', choices=('text', 'fusion', 'prefix'), default='fusion')
    p.add_argument('--hide-numerics-in-prompt', action='store_true', help='Fusion only: feed numbers exclusively through numeric encoder')
    p.add_argument('--train', default='data/synthetic/train.jsonl')
    p.add_argument('--dev', default='data/synthetic/dev.jsonl')
    p.add_argument('--output', default='outputs/meridians-fusion')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--max-numeric-rows', type=int, default=48)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--quantization', choices=('4bit', 'none'), default='4bit')
    p.add_argument('--runtime-limit-seconds', type=int, default=3650)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.steps < 1 or a.accumulation < 1 or a.batch_size < 1:
        p.error('steps, accumulation and batch-size must be positive')
    if a.hide_numerics_in_prompt and a.mode != 'fusion':
        p.error('--hide-numerics-in-prompt requires --mode fusion')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Pretrained LM fine-tuning requires a CUDA GPU for this recipe; no weights are trained by this command.')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if a.quantization == '4bit' else None
    lm = AutoModelForCausalLM.from_pretrained(a.base, revision=a.revision, torch_dtype=dtype, quantization_config=quant_cfg, device_map={'': 0} if quant_cfg else None)
    lm.config.use_cache = False
    if quant_cfg:
        lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    peft = get_peft_model(lm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    d = peft.get_input_embeddings().embedding_dim
    enc = NumericEncoder(d) if a.mode == 'fusion' else LearnedPrefix(d) if a.mode == 'prefix' else None
    if enc is not None:
        enc = enc.to('cuda')
    model = MeridiansLM(peft, enc)
    if not quant_cfg:
        model = model.to('cuda')
    ds = FinDS(a.train, tok, a.max_tokens, include_numeric_text=not a.hide_numerics_in_prompt)
    if not len(ds):
        raise ValueError('Training dataset is empty')
    data_rng = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, generator=data_rng, collate_fn=lambda x: collate(x, tok.pad_token_id))
    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=a.lr)
    torch.manual_seed(a.seed)
    model.train()
    running = 0.0
    step = 0
    micro = 0
    started = time.monotonic()
    order_hash = hashlib.sha256()
    opt.zero_grad(set_to_none=True)
    log_path = out / 'training_steps.jsonl'
    log_path.write_text('')
    deadline = launch_started + max(60, a.runtime_limit_seconds)
    cut_short = False
    while step < a.steps and (not cut_short):
        for ids, attn, labels, examples in loader:
            if time.monotonic() + 90 >= deadline:
                print('Stopping before wall-time deadline to preserve checkpoint', flush=True)
                cut_short = True
                break
            ids, attn, labels = (ids.cuda(), attn.cuda(), labels.cuda())
            order_hash.update((json.dumps([x['id'] for x in examples]) + '\n').encode())
            numbers = rows_to_tensor(examples, max_rows=a.max_numeric_rows, device='cuda') if enc else None
            loss = model(ids, attn, labels, numbers).loss / a.accumulation
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; refusing to continue')
            loss.backward()
            running += float(loss.detach())
            micro += 1
            if micro % a.accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                record = {'step': step, 'loss': running, 'grad_norm': float(grad_norm), 'microbatches': micro}
                with log_path.open('a') as log:
                    log.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
                running = 0.0
                if step >= a.steps:
                    break
        if len(loader) == 0:
            break
    peft.save_pretrained(out / 'lora_adapter', safe_serialization=True)
    tok.save_pretrained(out / 'tokenizer')
    if enc:
        torch.save(enc.cpu().state_dict(), out / 'numeric_encoder.pt')
    (out / 'training_manifest.json').write_text(json.dumps({'status': 'TRAINED' if step == a.steps else 'PARTIAL', 'base_model': a.base, 'mode': a.mode, 'quantization': a.quantization, 'include_numeric_text': not a.hide_numerics_in_prompt, 'steps': step, 'steps_requested': a.steps, 'seed': a.seed, 'dataset': a.train, 'max_tokens': a.max_tokens, 'base_model_revision': a.revision or getattr(peft.config, '_commit_hash', None), 'dataset_sha256': hashlib.sha256(Path(a.train).read_bytes()).hexdigest(), 'data_order_sha256': order_hash.hexdigest(), 'batch_size': a.batch_size, 'gradient_accumulation': a.accumulation, 'learning_rate': a.lr, 'max_numeric_rows': a.max_numeric_rows, 'trainable_parameters': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'numeric_schema': 'constant_learned_prefix' if a.mode == 'prefix' else 'coordinate_typed_v2' if any((x.get('research_format') for x in ds.data)) else 'synthetic_v1', 'prompt_format': 'qwen3-chat-no-thinking', 'model_type': 'LoRA + learned number-independent prefix' if a.mode == 'prefix' else 'LoRA-adapted pretrained LM + trainable numerical-prefix encoder' if enc else 'text-only LoRA'}, indent=2))
    print('Saved trained adapter and manifest in', str(out), 'completed_steps', step, flush=True)
    if step != a.steps:
        raise RuntimeError(f'Partial checkpoint preserved: completed {step}/{a.steps} optimizer steps')
if __name__ == '__main__':
    run()
