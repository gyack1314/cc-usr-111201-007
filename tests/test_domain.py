"""核对领域原语的业务数值。"""
import unittest
from analysis_data.io import load
from analysis_data.domain import summarize
class DomainTests(unittest.TestCase):
    def test_base_estimate(self):
        r=summarize(load('data/normal.json')[0]);self.assertEqual(r['base_weight_ratio'],0.25)
        self.assertEqual(len(r['psus']['H1']),2)
        self.assertEqual(len(summarize(load('data/single-psu-stratum.json')[0])['psus']['H1']),1)
