import json
import tempfile
import unittest
from pathlib import Path
from scripts.prepare_benchmarks import prepare_finqa, prepare_tatqa


class BenchmarkConversionTests(unittest.TestCase):
    def write_input(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'source.json'
        path.write_text(json.dumps(rows))
        return path

    def test_finqa_retains_zero_and_full_evidence(self):
        path = self.write_input([{'id': 'ACME/2020/page_1.pdf-0',
            'pre_text': ['a' * 3000], 'post_text': [], 'table': [],
            'qa': {'exe_ans': 0.0, 'question': 'What is the difference?'}}])
        row, = prepare_finqa(path, 'dev')
        self.assertEqual(row['answer'], '0.0')
        self.assertEqual(row['gold_answer'], 0.0)
        self.assertEqual(len(row['narrative']), 3000)

    def test_tatqa_preserves_list_scale_and_official_id(self):
        path = self.write_input([{'table': {'uid': 'table-1', 'table': []},
            'paragraphs': [], 'questions': [{'uid': 'q-1', 'answer': ['10', '20'],
            'question': 'Which values?', 'answer_type': 'multi-span', 'scale': 'million'}]}])
        row, = prepare_tatqa(path, 'dev')
        self.assertEqual(row['gold_answer'], ['10', '20'])
        self.assertEqual(row['benchmark_id'], 'q-1')
        self.assertEqual(row['scale'], 'million')
