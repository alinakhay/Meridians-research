import unittest,random,torch
from meridians.data import make_case,flip_claim
from meridians.numeric import NumericEncoder,rows_to_tensor
class TestData(unittest.TestCase):
 def test_claim_consistency(self):
    rng=random.Random(42)
    for i in range(200):
        x=make_case(rng,i,'train');f=x['ground_truth'];
        self.assertEqual(x['answer_label']=='supported',f['claim_expansion']==f['actual_expansion'])
        self.assertGreater(x['numeric_rows'][1]['value'],0)
 def test_claim_flip(self):
    x=make_case(random.Random(12),1,'test'); y=flip_claim(x)
    self.assertEqual(x['numeric_rows'],y['numeric_rows'])
    self.assertNotEqual(x['answer_label'],y['answer_label'])
    self.assertNotEqual(x['ground_truth']['claim_expansion'],y['ground_truth']['claim_expansion'])
 def test_encoder_backward(self):
    x=make_case(random.Random(12),1,'train')
    enc=NumericEncoder(40,32,3)
    features=rows_to_tensor([x],8)
    y=enc(features);self.assertEqual(y.shape,(1,3,40));y.mean().backward()
    self.assertIsNotNone(enc.value[0].weight.grad)
 def test_empty_numeric(self):
    enc=NumericEncoder(40,32,3)
    out=enc(rows_to_tensor([{'numeric_rows':[]}],4))
    self.assertTrue(torch.isfinite(out).all())
if __name__=='__main__':unittest.main()
