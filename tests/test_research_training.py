import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import LoraConfig,get_peft_model,TaskType
from meridians.modeling import MeridiansLM
from meridians.numeric import NumericEncoder,LearnedPrefix,rows_to_tensor
from scripts.train_llm import format_ids


def test_prompt_overflow_is_explicit():
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [1,2,3,4]
    with pytest.raises(ValueError,match='silent truncation'):
        format_ids(Tokenizer(),'long prompt',3)


def test_numeric_overflow_is_explicit():
    with pytest.raises(ValueError,match='silent truncation'):
        rows_to_tensor([{'numeric_rows':[{'value':1},{'value':2}]}],max_rows=1)


@pytest.mark.parametrize('encoder_cls',[NumericEncoder,LearnedPrefix])
def test_fusion_nonreentrant_checkpointed_optimizer_step_and_reload(tmp_path,encoder_cls):
    torch.manual_seed(42)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,use_cache=False)
    base=Qwen3ForCausalLM(config)
    for p in base.parameters():p.requires_grad=False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    lm=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=4,
            target_modules=['q_proj','k_proj','v_proj','o_proj']))
    encoder=encoder_cls(32)
    model=MeridiansLM(lm,encoder).train()
    numeric=rows_to_tensor([{'numeric_rows':[{'value':120,'metric_id':2,'period':1,'unit_id':5},
                                            {'value':100,'metric_id':2,'period':0,'unit_id':5}]}],max_rows=128)
    ids=torch.tensor([[1,2,3,4,5,6]])
    labels=ids.clone();labels[:,:3]=-100
    before={name:p.detach().clone() for name,p in encoder.named_parameters()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    loss=model(ids,torch.ones_like(ids),labels,numeric).loss
    assert torch.isfinite(loss)
    loss.backward();opt.step()
    assert any(not torch.equal(before[name],p) for name,p in encoder.named_parameters())
    torch.save(encoder.state_dict(),tmp_path/'numeric.pt')
    restored=encoder_cls(32)
    restored.load_state_dict(torch.load(tmp_path/'numeric.pt',weights_only=True))
    encoder.eval();restored.eval()
    assert torch.allclose(encoder(numeric),restored(numeric))
