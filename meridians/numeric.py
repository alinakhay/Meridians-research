"""Numeric token pathway: structured value, metric, time and unit embeddings."""
from __future__ import annotations
import math
import torch
from torch import nn


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


class LearnedPrefix(nn.Module):
    """A learned input prefix independent of every value, row, period and unit."""
    def __init__(self,out_dim:int,num_tokens:int=4):
        super().__init__()
        self.prefix=nn.Parameter(torch.randn(num_tokens,out_dim)*0.02)
        self.num_tokens=num_tokens

    def forward(self,numeric):
        return self.prefix.unsqueeze(0).expand(numeric['mask'].shape[0],-1,-1)
