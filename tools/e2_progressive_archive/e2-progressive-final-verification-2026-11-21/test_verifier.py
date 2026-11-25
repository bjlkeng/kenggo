"""Bounded guard and saved-metric tests. No model execution or production writes."""
import unittest
from unittest.mock import patch
from copy import deepcopy
import common, verify

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.ledger={'reservations':{},'entries':[]}
        self.summary={'status':'failed','costs':{'ledger':{'state':deepcopy(self.ledger)}}}
        self.obs={'fully_stopped':True}
    def test_live_blocks(self):
        self.assertIn('live',common.gate({'fully_stopped':False},self.ledger,self.summary))
    def test_unknown_reservation_blocks(self):
        self.ledger['reservations']={'abandoned-gpu':{}}
        self.assertIn('abandoned-gpu',common.gate(self.obs,self.ledger,self.summary))
    def test_missing_summary_blocks(self):
        self.assertIn('absent',common.gate(self.obs,self.ledger,None))
    def test_stale_summary_blocks(self):
        self.summary['costs']['ledger']['state']['entries']=[{'different':True}]
        self.assertIn('settled',common.gate(self.obs,self.ledger,self.summary))
    def test_missing_stop_record_blocks(self):
        with patch('common.Path.exists',return_value=False):
            self.assertIn('stop-hook',common.gate(self.obs,self.ledger,self.summary))
    def test_failed_experiment_may_be_inspected_but_is_not_success(self):
        with patch('common.Path.exists',return_value=True):
            self.assertIsNone(common.gate(self.obs,self.ledger,self.summary))
        self.assertEqual(self.summary['status'],'failed')
    def test_numeric_drift_rejected(self):
        with self.assertRaises(ValueError):verify.close({'nll':2.0},{'nll':2.01},'NLL')
    def test_mask_drift_rejected(self):
        with self.assertRaises(ValueError):verify.close({'mask':True},{'mask':False},'mask')
    def test_source_and_pins_read_only(self):
        self.assertEqual(common.source()['head'],common.HEAD)
        p=common.pins();self.assertEqual((p['manifest_pins'],p['handoff_artifacts']),(40,19))
    def test_live_identity(self):
        o=common.observation();self.assertEqual(o['identity_issues'],[])
        self.assertFalse(o['fully_stopped'])

if __name__=='__main__':unittest.main(verbosity=2)
