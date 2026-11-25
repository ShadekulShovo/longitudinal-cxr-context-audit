import unittest
import numpy as np
import pandas as pd
import models as m
class HistoryTests(unittest.TestCase):
    def test_strict_dates_and_patient_boundaries(self):
        frame=pd.DataFrame(dict(PatientID=['a','a','a','a','b'],_day=[1,1,2,3,1]))
        e=m.history_indices(frame)
        self.assertEqual(e[0].tolist(),[])
        self.assertEqual(e[2].tolist(),[0,1])
        self.assertEqual(e[4].tolist(),[])
        y=np.zeros((5,5));y[0,0]=1
        self.assertEqual(m.history_block(2,e[2],y,frame._day.to_numpy()).shape,(35,))
        with self.assertRaises(m.ProtocolError):m.history_block(2,[2],y,frame._day.to_numpy())
    def test_donors_are_matched_other_patients(self):
        keys=np.array([0,0,1,1]);patients=np.array(['a','b','c','d'])
        donors=m.select_donors(keys,patients)
        np.testing.assert_array_equal(keys[donors],keys)
        self.assertTrue(np.all(patients[donors]!=patients))
    def test_overlap_rejected(self):
        with self.assertRaises(m.ProtocolError):m.assert_disjoint(np.array(['a','a']),np.array(['train','test']))
