"""LoRA-trained decoder LM with optional numeric pseudo-token prefix."""
import torch
from torch import nn
from meridians.numeric import NumericEncoder

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
