"""Input-only table parsing. Gold answers and annotation fields are never consulted."""
from __future__ import annotations
import math
import re

UNIT_IDS = {'unspecified': 0, 'USD_m': 1, 'percent': 2, 'USD': 3,
            'thousand': 4, 'million': 5, 'billion': 6, 'GBP': 7, 'EUR': 8,
            'JPY': 9, 'count': 10, 'USD_thousand': 11, 'USD_billion': 12,
            'GBP_million': 13, 'EUR_million': 14, 'ratio': 15}


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
