import unittest
import numpy as np
import pandas as pd
from graded_history import graded_history
class GradedTests(unittest.TestCase):
    def test_original_graded_width_and_strict_history(self):
        dates={'a':pd.Timestamp('2020-01-01'),'b':pd.Timestamp('2020-01-02')}
        grades={'a':np.ones(8,int),'b':np.zeros(8,int)}
        last,summary,days=graded_history(pd.Timestamp('2020-01-03'),['a','b'],dates,grades)
        self.assertEqual(last.shape,(33,))
        self.assertEqual(summary.shape,(76,))
        np.testing.assert_array_equal(summary[49:57],(-np.ones(8)/np.array([3,4,4,4,4,4,4,4])).astype(np.float32))
        with self.assertRaises(AssertionError):graded_history(pd.Timestamp('2020-01-02'),['b'],dates,grades)
