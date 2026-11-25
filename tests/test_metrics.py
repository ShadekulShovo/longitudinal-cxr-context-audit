import ast
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, roc_curve
import metrics as m
ROOT = Path(__file__).resolve().parents[1]

class UpgradeMetricsTests(unittest.TestCase):

    def test_auc_decomposition_matches_bruteforce_with_ties(self):
        rng = np.random.default_rng(20261116)
        for _ in range(40):
            y = rng.integers(0, 2, 31)
            scores = rng.integers(0, 5, 31)
            prior = rng.integers(0, 2, 31)
            r = m.auroc_decomposition(y, scores, prior)
            wins, within, across = ([], [], [])
            for i in np.flatnonzero(y == 1):
                for j in np.flatnonzero(y == 0):
                    value = float(scores[i] > scores[j]) + 0.5 * (scores[i] == scores[j])
                    wins.append(value)
                    (within if prior[i] == prior[j] else across).append(value)
            self.assertAlmostEqual(r['AUROC'], np.mean(wins), places=12)
            self.assertAlmostEqual(r['AUROC'], roc_auc_score(y, scores), places=12)
            self.assertAlmostEqual(r['A_within'], np.mean(within), places=12)
            self.assertAlmostEqual(r['A_across'], np.mean(across), places=12)
            self.assertAlmostEqual(r['AUROC'], r['p'] * r['A_within'] + (1 - r['p']) * r['A_across'], places=12)

    def test_empty_pair_sets_and_missingness(self):
        for prior in [[0, 0], [1, 0]]:
            r = m.auroc_decomposition([0, 1], [0.2, 0.8], prior)
            self.assertEqual(r['AUROC'], 1.0)
            self.assertEqual(r['within_component'] + r['across_component'], 1.0)
        self.assertTrue(np.isnan(m.auroc_decomposition([1, 1], [0, 1], [0, 1])['AUROC']))
        r = m.auroc_decomposition([0, 1, 0], [0.2, 0.8, 0.9], [0, 0, np.nan])
        self.assertEqual(r['studies'], 2)
        self.assertEqual(r['AUROC'], 1.0)

    def test_gain_identity_and_macro_ratio(self):
        y = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        a = np.array([[0.1, 0.3], [0.8, 0.4], [0.9, 0.7], [0.2, 0.8]])
        b = np.array([[0.2, 0.7], [0.1, 0.8], [0.4, 0.6], [0.3, 0.5]])
        prior = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        r = m.gain_decomposition(y, a, b, prior)
        macro = r['macro']
        self.assertAlmostEqual(macro['delta_auroc'], macro['within_gain'] + macro['across_gain'], places=12)
        expected = np.mean([roc_auc_score(y[:, j], a[:, j]) - roc_auc_score(y[:, j], b[:, j]) for j in range(2)])
        self.assertAlmostEqual(macro['delta_auroc'], expected, places=12)
        self.assertTrue(np.isnan(m.gain_decomposition(y, a, a, prior)['macro']['change_share']))
        missing = m.gain_decomposition([0, 1, 0], [0.2, 0.8, np.nan], [0.3, 0.7, 0.9], [0, 0, 1])
        self.assertEqual(missing['per_finding'][0]['model_a']['studies'], 2)
        self.assertEqual(missing['per_finding'][0]['model_b']['studies'], 2)

    def test_youden_exact_legacy_rule_including_ties(self):
        for y, scores in [([0, 1], [0.5, 0.5]), ([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])]:
            fpr, tpr, cuts = roc_curve(y, scores)
            self.assertEqual(m.youden_threshold(y, scores), float(cuts[np.argmax(tpr - fpr)]))
        self.assertTrue(np.isinf(m.youden_threshold([0, 1], [0.5, 0.5])))
        with self.assertRaises(ValueError):
            m.youden_threshold([1, 1], [0.2, 0.8])

    def test_events_and_per1000_use_studies_not_event_denominator(self):
        y = [[1, 0], [0, 1]]
        prev = [[0, 1], [1, 0]]
        pred = [[0, 1], [1, 0]]
        rates = m.event_metrics(y, prev, pred)['pooled']
        self.assertEqual(rates['onset_events'], 2)
        self.assertEqual(rates['resolution_events'], 2)
        self.assertEqual(rates['onset_sensitivity'], 0.0)
        counts = m.per_1000(y, prev, pred)
        self.assertEqual(counts['missed_onsets_per_1000'], 1000.0)
        self.assertEqual(counts['false_persistences_per_1000'], 1000.0)
        self.assertTrue(np.isnan(m.event_metrics([0], [0], [0])['pooled']['onset_sensitivity']))

    def test_prior_and_threshold_missingness(self):
        np.testing.assert_allclose(m.prior_state([0, 1, 4, -1, np.nan]), [0, 1, 1, np.nan, np.nan], equal_nan=True)
        got = m.apply_thresholds([0.3, 0.8, 0.9], [[0.4], [0.7]], [0, 1, np.nan])
        np.testing.assert_allclose(got.ravel(), [0, 1, np.nan], equal_nan=True)
        with self.assertRaises(ValueError):
            m.pst_thresholds([0, 1], [0.2, 0.8], [0, 1])
        with self.assertRaises(ValueError):
            m.event_metrics([2], [0], [1])

    def test_calibration_and_decision_curves(self):
        self.assertEqual(m.ece([0, 1], [0, 1])[0], 0.0)
        r = m.net_benefit([1, 0, 1, 0], [0.8, 0.7, 0.6, 0.1], [0.5])[0]
        self.assertAlmostEqual(r['model'][0], 0.25)
        self.assertEqual(r['treat_all'][0], 0.0)
        with self.assertRaises(ValueError):
            m.net_benefit([0, 1], [0.2, 0.8], [1.0])
        y, p = ([], [])
        for numerator in [2, 4, 6, 8]:
            y += [1] * numerator + [0] * (10 - numerator)
            p += [numerator / 10] * 10
        result = m.calibration_slope(y, expit(2 * logit(p)))[0]
        self.assertEqual(result['status'], 'estimated')
        self.assertAlmostEqual(result['slope'], 0.5, places=6)
        self.assertAlmostEqual(result['intercept'], 0.0, places=6)
        self.assertEqual(m.calibration_slope([0, 1], [0.1, 0.9])[0]['status'], 'undefined_separated')

    def test_bootstrap_preserves_entire_patients_and_reproduces_draws(self):
        ids = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
        seen = []

        def statistic(indices):
            counts = np.bincount(indices, minlength=len(ids))
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[3], counts[4])
            self.assertEqual(counts[4], counts[5])
            seen.append(indices)
            return np.array([len(indices), (ids[indices] == 'a').mean()])
        r = m.patient_bootstrap(statistic, ids, B=40)
        self.assertEqual(r['patients'], 3)
        np.testing.assert_array_equal(r['valid_replicates'], [40, 40])
        rng = np.random.default_rng(20261115)
        groups = [np.flatnonzero(ids == k) for k in np.unique(ids)]
        for indices in seen[1:]:
            expected = np.concatenate([groups[k] for k in rng.choice(3, 3, replace=True)])
            np.testing.assert_array_equal(indices, expected)
if __name__ == '__main__':
    unittest.main()
