"""核对资料装载、拒收边界与指纹稳定性。"""
import json
import tempfile
import unittest
from pathlib import Path
from analysis_data.io import finite,load,unique
from analysis_data.domain import summarize
class MaterialTests(unittest.TestCase):
    def test_primary_material(self):
        value,digest=load('data/normal.json');self.assertEqual(len(digest),64)
        first=summarize(value);second=summarize(json.loads(json.dumps(value)))
        self.assertEqual(first,second)
        self.assertTrue(first)
    def test_invalid_material(self):
        for p in Path('data').glob('invalid-*.json'):
            with self.subTest(path=str(p)):
                with self.assertRaises((ValueError,KeyError,TypeError)):summarize(load(p)[0])
    def test_finite_numbers(self):
        for x in [True,float('inf'),float('nan'),'2']:
            with self.assertRaises(ValueError):finite(x)
    def test_raw_loading(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'资料.json';p.write_text('{"a":1}',encoding='utf8')
            v,k=load(p);self.assertEqual(v,{'a':1});self.assertEqual(k,load(p)[1])
            p.write_text('{"a":NaN}',encoding='utf8')
            with self.assertRaises(ValueError):load(p)
    def test_duplicate_identity(self):
        with self.assertRaises(ValueError):unique([{'id':'same'},{'id':'same'}],'id')
