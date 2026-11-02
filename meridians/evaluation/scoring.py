"""Strict prediction parsing and source-specific scoring. No gold-dependent parsing."""
from __future__ import annotations
import json
import math
import re
from decimal import Decimal,InvalidOperation
from dataclasses import dataclass
from .tatqa.tatqa_metric import TaTQAEmAndF1

SCALES = {'', 'thousand', 'million', 'billion', 'percent'}

def canonical_execution_reference(execution_answer,display_annotation):
    """Normalize only unambiguous reference units, without seeing a prediction.

    FinQA sometimes stores a literal percentage-point result and sometimes a
    fraction. An explicit percentage annotation determines units only when the
    execution value is consistent with one convention and not the other, within
    annotation rounding plus five-decimal execution rounding. Ambiguous or
    inconsistent references are retained and flagged, never silently corrected.
    """
    raw=str(display_annotation).strip().replace(',','')
    match=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%',raw)
    if match is None:return execution_answer,'no_explicit_percentage_annotation'
    try:
        displayed=Decimal(match[1]);value=Decimal(str(execution_answer))
        if not value.is_finite():raise InvalidOperation
        places=len(match[1].split('.')[1]) if '.' in match[1] else 0
        half=Decimal(10)**(-places)/2
        fraction=abs(value*100-displayed)<=half+Decimal('.000500001')
        points=abs(value-displayed)<=half+Decimal('.000005001')
    except (InvalidOperation,ValueError):return execution_answer,'invalid_reference'
    if points and not fraction:return round(float(value/100),5),'percentage_points_to_fraction'
    if fraction and not points:return execution_answer,'fraction_already'
    return execution_answer,'ambiguous_both' if fraction and points else 'inconsistent_with_display_rounding'

@dataclass(frozen=True)
class ParsedAnswer:
    answer: list[str]
    scale: str


def parse_answer(text: str) -> ParsedAnswer | None:
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if not match:
            return None
        text = match.group(1)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or set(data) != {'answer', 'scale'}:
        return None
    if not isinstance(data['scale'], str) or data['scale'].strip().lower() not in SCALES:
        return None
    answer = data['answer'] if isinstance(data['answer'], list) else [data['answer']]
    if not 1 <= len(answer) <= 32:
        return None
    out = []
    for item in answer:
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            return None
        if isinstance(item, float) and not math.isfinite(item):
            return None
        value = str(item).strip()
        if len(value) > 4096:
            return None
        out.append(value)
    if not any(out):
        return None
    return ParsedAnswer(out, data['scale'].strip().lower())


def canonical_target(example: dict) -> str:
    """Gold target for TRAIN examples only; this must never enter an input prompt."""
    answer = example['gold_answer']
    if not isinstance(answer, list):
        answer = [answer]
    # Preserve upstream blank multi-span items: the official scorer's scale
    # handling makes deleting them change the score for one training example.
    return json.dumps({'answer': [str(x) for x in answer],
                       'scale': example.get('scale', '') if example['source'] == 'tatqa' else ''},
                      ensure_ascii=False, separators=(',', ':'))


def finqa_answer_match(prediction: ParsedAnswer | None, gold) -> bool:
    """Answer-only accuracy using FinQA's five-decimal execution-result convention.

    This is NOT official program execution or program equivalence accuracy.
    Explicit percentage notation is converted without consulting the gold value.
    No selection among multiple numbers or gold-dependent factor-of-100 matching.
    """
    if prediction is None or len(prediction.answer) != 1:
        return False
    pred = prediction.answer[0].strip().lower()
    if isinstance(gold, str) and gold.strip().lower() in {'yes', 'no'}:
        return prediction.scale == '' and pred == gold.strip().lower()
    if prediction.scale not in {'', 'percent'}:
        return False
    percent = pred.endswith('%')
    if percent:
        pred = pred[:-1].strip()
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', pred):
        return False
    try:
        value = float(pred)
        expected = float(gold)
    except (ValueError, TypeError, OverflowError):
        return False
    if percent or prediction.scale == 'percent':
        value /= 100
    return math.isfinite(value) and math.isfinite(expected) and round(value, 5) == expected


def score_example(example: dict, text: str) -> dict:
    pred = parse_answer(text)
    result = {'id': example['id'], 'source': example['source'],
              'valid_json': pred is not None, 'exact_match': 0.0, 'f1': None}
    if example['source'] == 'finqa':
        result['metric'] = 'finqa_answer_only_5dp'
        result['exact_match'] = float(finqa_answer_match(pred, example['gold_answer']))
        if 'canonical_gold_answer' in example:
            result['exact_match_unit_canonical']=float(finqa_answer_match(pred,example['canonical_gold_answer']))
            result['reference_unit_convention']=example['reference_unit_convention']
    elif example['source'] == 'tatqa':
        result['metric'] = 'official_tatqa_em_f1'
        metric = TaTQAEmAndF1()
        gold = {'answer_type': example['answer_type'], 'answer': example['gold_answer'],
                'scale': example.get('scale', '')}
        metric(gold, pred.answer if pred else [], pred.scale if pred else '')
        em, f1, scale, _ = metric.get_overall_metric()
        result.update(exact_match=float(em), f1=float(f1), scale_accuracy=float(scale))
    elif example['source']=='synthetic_cf':
        result['metric']='synthetic_claim_label_accuracy'
        result['exact_match']=float(pred is not None and pred.scale=='' and
            len(pred.answer)==1 and pred.answer[0].strip().lower()==example['gold_answer'])
    else:
        raise ValueError('Unsupported benchmark source')
    return result


def summarize(scores: list[dict]) -> dict:
    output = {'n': len(scores), 'datasets': {}}
    for source in ('finqa', 'tatqa', 'synthetic_cf'):
        subset = [x for x in scores if x['source'] == source]
        if not subset:
            continue
        output['datasets'][source] = {
            'n': len(subset), 'metric': subset[0]['metric'],
            'exact_match': sum(x['exact_match'] for x in subset) / len(subset),
            'parse_rate': sum(x['valid_json'] for x in subset) / len(subset),
        }
        if source == 'tatqa':
            output['datasets'][source]['f1'] = sum(x['f1'] for x in subset) / len(subset)
            output['datasets'][source]['scale_accuracy'] = sum(x['scale_accuracy'] for x in subset) / len(subset)
        if source=='finqa' and all('exact_match_unit_canonical' in x for x in subset):
            output['datasets'][source]['unit_canonical_execution_reference_exactness']=sum(x['exact_match_unit_canonical'] for x in subset)/len(subset)
            output['datasets'][source]['references_converted_from_percentage_points']=sum(x['reference_unit_convention']=='percentage_points_to_fraction' for x in subset)
    return output
