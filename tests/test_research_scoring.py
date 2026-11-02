import json
import pytest
from meridians.evaluation.scoring import parse_answer,score_example,canonical_target,canonical_execution_reference
from meridians.evaluation.evidence import parse_cell,extract_numeric_rows,research_prompt


def finqa(gold):
    return {'id':'f','source':'finqa','gold_answer':gold}


def test_strict_json_does_not_guess_a_number_from_reasoning():
    assert parse_answer('The answer is 5, although 3 is possible.') is None
    assert parse_answer('{"answer":["5"],"scale":""} trailing') is None
    assert parse_answer('{"answer":[true],"scale":""}') is None
    assert parse_answer('{"answer":[NaN],"scale":""}') is None
    assert parse_answer('{"answer":[""],"scale":""}') is None
    assert parse_answer('{"answer":["5"],"scale":"million","extra":5}') is None
    assert parse_answer('```json\n{"answer":["5"],"scale":""}\n```').answer==['5']


def test_finqa_zero_percent_sign_and_exact_convention():
    assert score_example(finqa(0),'0')['exact_match']==0
    assert score_example(finqa(0),'{"answer":["0"],"scale":""}')['exact_match']==1
    assert score_example(finqa(.12),'{"answer":["12%"],"scale":""}')['exact_match']==1
    assert score_example(finqa(.12),'{"answer":["12"],"scale":""}')['exact_match']==0
    assert score_example(finqa(-.12346),'{"answer":["-0.123456"],"scale":""}')['exact_match']==1
    assert score_example(finqa('yes'),'{"answer":["yes"],"scale":""}')['exact_match']==1


def test_tatqa_multispan_scale_count_zero_and_missing_predictions():
    x={'id':'t','source':'tatqa','gold_answer':['China','USA'],'answer_type':'multi-span','scale':''}
    assert score_example(x,'{"answer":["USA","China"],"scale":""}')['exact_match']==1
    x['gold_answer'].append('')
    assert score_example(x,canonical_target(x))['exact_match']==1
    x.update(gold_answer=['18.0','18.7',''],scale='percent')
    assert score_example(x,canonical_target(x))['exact_match']==1
    x.update(gold_answer=['China','USA'],scale='')
    assert score_example(x,'{"answer":["China"],"scale":""}')['exact_match']==0
    assert 0<score_example(x,'{"answer":["China"],"scale":""}')['f1']<1
    y={'id':'t','source':'tatqa','gold_answer':2,'answer_type':'arithmetic','scale':'million'}
    assert score_example(y,'{"answer":["2"],"scale":"million"}')['exact_match']==1
    assert score_example(y,'{"answer":["2"],"scale":"thousand"}')['exact_match']==0
    z={'id':'t','source':'tatqa','gold_answer':0,'answer_type':'count','scale':''}
    assert score_example(z,canonical_target(z))['exact_match']==1
    assert score_example(z,'') ['exact_match']==0


def test_table_parser_retains_sign_and_percentage_and_excludes_year_headers():
    assert parse_cell('$ (1,234.50)')==(-1234.5,False,'USD')
    assert parse_cell('−3.2%')==(-3.2,True,'')
    assert parse_cell('September 30, 2019') is None
    assert parse_cell('-') is None
    table=[['','2019','2018'],['Revenue','$ 10','12'],['Operating margin (%)','3.2%','(1.4)']]
    rows,meta=extract_numeric_rows(table,'Amounts in millions.')
    assert len(rows)==4
    assert all(x['value'] not in (2018,2019) for x in rows)
    assert rows[0]['unit']=='USD_m'
    assert rows[2]['unit']=='percent'
    assert rows[0]['period']==1 and rows[1]['period']==0
    assert meta['header_rows']==[0]


def test_gold_fields_never_enter_research_prompt():
    x={'source':'finqa','question':'How much?','narrative':'Revenue was disclosed.',
       'numeric_text':'Revenue | 10 | 11','gold_answer':'SECRET_TARGET_91827','scale':'SECRET_SCALE'}
    before=research_prompt(x)
    x.update(gold_answer='DIFFERENT_TARGET',reference_program='SECRET_PROGRAM')
    assert research_prompt(x)==before
    assert 'SECRET' not in before


def test_secondary_reference_units_are_determined_without_a_prediction():
    assert canonical_execution_reference(24.69136,'24.69%')==(.24691,'percentage_points_to_fraction')
    assert canonical_execution_reference(.13153,'13%')==(.13153,'fraction_already')
    assert canonical_execution_reference(5929.17,'73.7%')==(5929.17,'inconsistent_with_display_rounding')
    assert canonical_execution_reference(0,'0%')==(0,'ambiguous_both')
    x=finqa(24.69136);x.update(canonical_gold_answer=.24691,reference_unit_convention='percentage_points_to_fraction')
    score=score_example(x,'{"answer":["0.2469136"],"scale":""}')
    assert score['exact_match']==0 and score['exact_match_unit_canonical']==1
    assert score_example(x,'{"answer":["24.69136"],"scale":""}')['exact_match_unit_canonical']==0
