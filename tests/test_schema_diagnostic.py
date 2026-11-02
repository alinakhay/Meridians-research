import pytest
from scripts.diagnose_counterfactual_schema import label_only
from meridians.evaluation.scoring import score_example


def test_label_diagnostic_does_not_repair_the_frozen_primary_score():
    text='{"answer":["supported"],"scale":"."}'
    example={'id':'cf-check','source':'synthetic_cf','gold_answer':'supported'}
    assert label_only(text)=='supported'
    assert score_example(example,text)['exact_match']==0
    # Label extraction remains the same when the reference is reversed.
    example['gold_answer']='contradicted'
    assert label_only(text)!='contradicted'
    assert score_example(example,text)['exact_match']==0


@pytest.mark.parametrize('text',[
    'The claim is supported.',
    '{"answer":["supported","contradicted"],"scale":""}',
    '{"answer":["supported"],"scale":"","explanation":"extra"}',
    '{"answer":["supported"],"scale":null}',
    '{"answer":["supported"],"scale":""} trailing text',
])
def test_label_diagnostic_does_not_search_or_select_an_answer(text):
    assert label_only(text) is None


def test_scale_diagnostic_keeps_primary_scalar_answer_tolerance():
    text='{"answer":"supported","scale":"."}'
    assert label_only(text)=='supported'
    assert score_example({'id':'scalar','source':'synthetic_cf','gold_answer':'supported'},text)['exact_match']==0
