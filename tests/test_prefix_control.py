import torch
from meridians.numeric import LearnedPrefix,rows_to_tensor

def test_control_cannot_condition_on_values_coordinates_or_units():
    torch.manual_seed(42);encoder=LearnedPrefix(32)
    a=rows_to_tensor([{'numeric_rows':[{'value':10,'metric_id':1,'period':0,'unit_id':1}]}],max_rows=128)
    b=rows_to_tensor([{'numeric_rows':[{'value':-1e9,'metric_id':42,'period':15,'unit_id':9},
                                    {'value':.001,'metric_id':2,'period':1,'unit_id':3}]}],max_rows=128)
    assert torch.equal(encoder(a),encoder(b))
    assert encoder(a).shape==(1,4,32)
    before=encoder.prefix.detach().clone()
    encoder(a).square().sum().backward()
    assert torch.isfinite(encoder.prefix.grad).all()
    with torch.no_grad():encoder.prefix-=.1*encoder.prefix.grad
    assert not torch.equal(before,encoder.prefix)
    assert torch.equal(encoder(a),encoder(b))
