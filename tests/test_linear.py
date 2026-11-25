import unittest, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import linear_audit as m
import models as core

def fixture():
    rng = np.random.default_rng(18)
    z = rng.normal(size=(60, 8))
    coef = rng.normal(size=(5, 8))
    b = rng.normal(size=5)
    logits = z @ coef.T + b
    group = m.b2_groups(4, 4)
    decomp = {k: m.linear_decomposition(z, coef, b, r, group, logits) for k, r in [('train_mean', np.zeros(8)), ('validate_mean', np.ones(8) * 0.1)]}
    y = rng.integers(0, 2, (60, 5))
    prior = rng.integers(0, 2, (60, 5))
    image = rng.normal(size=(60, 5))
    st = np.zeros(5)
    edges, _ = m.distance_quartiles(y, prior, image, st)
    return m.LinearAuditStatistics(y, prior, np.repeat(np.arange(20), 3), logits, image, st, st, decomp, edges)

class LinearTests(unittest.TestCase):

    def test_exact_groups_sum_to_saved_logit_and_bad_snapshot_stops(self):
        rng = np.random.default_rng(2)
        z = rng.normal(size=(9, 40))
        coef = rng.normal(size=(5, 40))
        b = np.arange(5)
        logits = z @ coef.T + b
        v = m.linear_decomposition(z, coef, b, z[:3].mean(0), m.m4_groups(5), logits)
        np.testing.assert_allclose(v['contribution'].sum(2) + v['baseline'], logits, atol=1e-12)
        logits[0, 0] += 1e-05
        with self.assertRaisesRegex(core.ProtocolError, 'RECONSTRUCTION_FAILED'):
            m.linear_decomposition(z, coef, b, z[:3].mean(0), m.m4_groups(5), logits)

    def test_group_partition_rejects_missing_and_overlapping_features(self):
        for groups in [dict(image=[0], history=[0, 1]), dict(image=[0])]:
            with self.assertRaisesRegex(core.ProtocolError, 'GROUP_PARTITION'):
                m.linear_decomposition(np.zeros((2, 2)), np.ones((5, 2)), np.zeros(5), np.zeros(2), groups, np.zeros((2, 5)))

    def test_float32_head_coordinates_avoid_raw_scaler_rounding_error(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(20, 512)).astype(np.float32)
        mean = rng.normal(size=512)
        scale = np.ones(512) * 0.93
        z = x.copy()
        z -= mean
        z /= scale
        coef = rng.normal(size=(5, 512))
        b = np.zeros(5)
        saved = np.column_stack([z @ c for c in coef])
        self.assertGreater(np.abs((x.astype(float) - mean) / scale @ coef.T - saved).max(), 1e-09)
        v = m.linear_decomposition(z, coef, b, z.mean(0, dtype=float), m.b2_groups(256, 256), saved)
        self.assertLess(v['max_absolute_logit_error'], 1e-09)

    def test_every_cell_partition_and_net_uses_study_denominator(self):
        stat = fixture()
        w = np.ones(60)
        v = stat(w)
        self.assertTrue(np.all(sum(stat.cells.values()) == 1))
        self.assertTrue(np.all(sum(stat.events.values()) == 1))
        for event, mask in stat.events.items():
            for j in range(5):
                total = sum((v[f'X2/{event}/{name}/count/finding_{j}'] for name in stat.cells))
                self.assertEqual(total, int(mask[:, j].sum()))
                expected = 1000 * ((mask[:, j] & stat.cells['rescue'][:, j]).sum() - (mask[:, j] & stat.cells['override'][:, j]).sum()) / 60
                self.assertAlmostEqual(v[f'X2/{event}/net_rescues_per1000_studies/finding_{j}'], expected)

    def test_shares_sum_to_one_for_both_references(self):
        v = fixture()(np.ones(60))
        for ref in ['train_mean', 'validate_mean']:
            for j in range(5):
                self.assertAlmostEqual(sum((v[f'X1/{ref}/all/group_share/{g}/finding_{j}'] for g in ['current_image', 'temporal'])), 1)

    def test_quartile_empty_support_and_duplicate_edges_not_rescued(self):
        y = np.zeros((8, 5), int)
        prior = np.ones_like(y)
        scores = np.zeros_like(y, float) - 1
        e, c = m.distance_quartiles(y, prior, scores, np.zeros(5))
        self.assertTrue(np.equal(e[:, 0], e[:, 2]).all())
        e, c = m.distance_quartiles(y, prior, scores, np.full(5, -2))
        self.assertTrue(np.isnan(e).all())
        self.assertEqual(c.sum(), 0)

    def test_no_new_AUROC_or_fit_and_bootstrap_resume(self):
        stat = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'fit_linear', side_effect=AssertionError('FIT')), patch.object(core, 'binary_auc', side_effect=AssertionError('AUROC')):
            first = core.bootstrap_report(stat, Path(tmp) / 'bootstrap', 5, lambda *a, **k: None)
            second = core.bootstrap_report(stat, Path(tmp) / 'bootstrap', 5, lambda *a, **k: None, resume=True)
            self.assertEqual(first['bootstrap'], second['bootstrap'])
            self.assertEqual(set(first['metrics']), set(second['metrics']))
            self.assertFalse(any((k.startswith('AUROC/') for k in first['metrics'])))

    def test_changed_saved_logits_rejected_before_any_statistics(self):
        s = fixture()
        wrong = s.candidate.copy()
        wrong[0, 0] += 0.1
        with self.assertRaisesRegex(core.ProtocolError, 'DIFFERENT_SAVED_LOGITS'):
            m.LinearAuditStatistics(s.y, s.prior, s.patients, wrong, s.image, s.cst, s.ist, s.decomp, s.quartiles)
if __name__ == '__main__':
    unittest.main()
