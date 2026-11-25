"""Threshold, discrimination, calibration and paired resampling metrics."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_curve

def _matrix(x):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError('Expected studies by findings, or one finding')
    return a

def _matched(*values):
    arrays = [_matrix(v) for v in values]
    if any((a.shape != arrays[0].shape for a in arrays)):
        raise ValueError('Study/finding shapes differ')
    return arrays

def _binary_valid(a):
    if np.any(np.isfinite(a) & ~np.isin(a, [0, 1])):
        raise ValueError('Binary inputs must be 0, 1 or missing')
    return np.isfinite(a)

def _prob_valid(a):
    if np.any(np.isfinite(a) & ((a < 0) | (a > 1))):
        raise ValueError('Probabilities must be in [0,1]')
    return np.isfinite(a)

def _ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)

def prior_state(labels_prev):
    a = np.asarray(labels_prev, dtype=float)
    return np.where(np.isfinite(a) & (a >= 0), (a > 0).astype(float), np.nan)

def youden_threshold(y, p):
    y, p = (np.asarray(y, float).ravel(), np.asarray(p, float).ravel())
    if y.shape != p.shape:
        raise ValueError('Label/score shapes differ')
    valid = _binary_valid(y) & np.isfinite(p)
    if np.unique(y[valid]).size != 2:
        raise ValueError('Youden threshold needs both outcome classes')
    fpr, tpr, cuts = roc_curve(y[valid], p[valid])
    return float(cuts[np.argmax(tpr - fpr)])

def pst_thresholds(y, p, prior):
    y, p, prior = _matched(y, p, prior)
    _binary_valid(prior)
    return np.array([[youden_threshold(y[prior[:, j] == s, j], p[prior[:, j] == s, j]) for j in range(y.shape[1])] for s in [0, 1]])

def apply_thresholds(p, thresholds, prior=None):
    p = _matrix(p)
    cuts = np.asarray(thresholds, float)
    if np.isnan(cuts).any():
        raise ValueError('Threshold is missing')
    if cuts.ndim <= 1:
        cuts = np.broadcast_to(cuts, (p.shape[1],))
        return np.where(np.isfinite(p), p >= cuts, np.nan)
    if cuts.shape != (2, p.shape[1]) or prior is None:
        raise ValueError('PST requires two cuts per finding and prior states')
    p, prior = _matched(p, prior)
    valid = np.isfinite(p) & _binary_valid(prior)
    selected = np.where(prior == 1, cuts[1], cuts[0])
    return np.where(valid, p >= selected, np.nan)

def event_metrics(y, y_prev, yhat):
    y, prior, pred = _matched(y, y_prev, yhat)
    valid = _binary_valid(y) & _binary_valid(prior) & _binary_valid(pred)
    onset = valid & (prior == 0) & (y == 1)
    resolution = valid & (prior == 1) & (y == 0)

    def result(axis):
        on, re = (onset.sum(axis=axis), resolution.sum(axis=axis))
        return dict(onset_events=on, resolution_events=re, detected_onsets=(onset & (pred == 1)).sum(axis=axis), cleared_resolutions=(resolution & (pred == 0)).sum(axis=axis), onset_sensitivity=_ratio((onset & (pred == 1)).sum(axis=axis), on), resolution_specificity=_ratio((resolution & (pred == 0)).sum(axis=axis), re))
    return dict(pooled=result(None), per_finding=result(0), studies=len(y), valid_finding_events=int(valid.sum()))

def _wins(positive_scores, negative_scores):
    neg = np.sort(negative_scores)
    return float((np.searchsorted(neg, positive_scores, side='left') + 0.5 * (np.searchsorted(neg, positive_scores, side='right') - np.searchsorted(neg, positive_scores, side='left'))).sum())

def auroc_decomposition(y, s, prior):
    y, s, prior = [np.asarray(a, float).ravel() for a in [y, s, prior]]
    if not y.shape == s.shape == prior.shape:
        raise ValueError('Label/score/prior shapes differ')
    valid = _binary_valid(y) & _binary_valid(prior) & np.isfinite(s)
    y, s, prior = (y[valid], s[valid], prior[valid])
    pos, neg = (y == 1, y == 0)
    total = int(pos.sum()) * int(neg.sum())
    if total == 0:
        return dict(p=np.nan, A_within=np.nan, A_across=np.nan, AUROC=np.nan, within_component=np.nan, across_component=np.nan, pairs=0, within_pairs=0, across_pairs=0, studies=len(y))
    all_wins = _wins(s[pos], s[neg])
    within_pairs, within_wins = (0, 0.0)
    for state in [0, 1]:
        a, b = (pos & (prior == state), neg & (prior == state))
        within_pairs += int(a.sum()) * int(b.sum())
        within_wins += _wins(s[a], s[b])
    across_pairs = total - within_pairs
    return dict(p=within_pairs / total, A_within=within_wins / within_pairs if within_pairs else 0.0, A_across=(all_wins - within_wins) / across_pairs if across_pairs else 0.0, AUROC=all_wins / total, within_component=within_wins / total, across_component=(all_wins - within_wins) / total, pairs=total, within_pairs=within_pairs, across_pairs=across_pairs, studies=len(y))

def gain_decomposition(y, s_a, s_b, prior):
    y, a, b, prior = _matched(y, s_a, s_b, prior)
    rows = []
    for j in range(y.shape[1]):
        mask = np.isfinite(a[:, j]) & np.isfinite(b[:, j])
        da = auroc_decomposition(y[mask, j], a[mask, j], prior[mask, j])
        db = auroc_decomposition(y[mask, j], b[mask, j], prior[mask, j])
        delta = da['AUROC'] - db['AUROC']
        within = da['within_component'] - db['within_component']
        across = da['across_component'] - db['across_component']
        share = within / delta if np.isfinite(delta) and abs(delta) > 1e-12 else np.nan
        rows.append(dict(delta_auroc=delta, within_gain=within, across_gain=across, change_share=share, p=da['p'], model_a=da, model_b=db))
    valid = [v for v in rows if np.isfinite(v['delta_auroc'])]
    macro = {k: float(np.mean([v[k] for v in valid])) if valid else np.nan for k in ['delta_auroc', 'within_gain', 'across_gain']}
    d = macro['delta_auroc']
    macro['change_share'] = macro['within_gain'] / d if np.isfinite(d) and abs(d) > 1e-12 else np.nan
    macro['valid_findings'] = len(valid)
    return dict(per_finding=rows, macro=macro)

def per_1000(y, y_prev, yhat):
    m = event_metrics(y, y_prev, yhat)
    per, pooled = (m['per_finding'], m['pooled'])
    result = dict(studies=m['studies'], valid_finding_events=m['valid_finding_events'])
    for key, source, hit in [('missed_onsets', 'onset_events', 'detected_onsets'), ('false_persistences', 'resolution_events', 'cleared_resolutions')]:
        result[key] = int(pooled[source] - pooled[hit])
        result[key + '_per_1000'] = float(_ratio(result[key] * 1000, m['studies']))
        result[key + '_per_finding_per_1000'] = _ratio((per[source] - per[hit]) * 1000, m['studies'])
    return result

def net_benefit(y, p, thresholds):
    y, p = _matched(y, p)
    cuts = np.asarray(thresholds, float).ravel()
    if not len(cuts) or np.any(~np.isfinite(cuts) | (cuts <= 0) | (cuts >= 1)):
        raise ValueError('Decision-curve thresholds must be strictly between0and1')
    valid = _binary_valid(y) & _prob_valid(p)
    result = []
    for j in range(y.shape[1]):
        yy, pp = (y[valid[:, j], j], p[valid[:, j], j])
        decision = pp[:, None] >= cuts
        tp = (decision & (yy[:, None] == 1)).sum(0)
        fp = (decision & (yy[:, None] == 0)).sum(0)
        odds = cuts / (1 - cuts)
        result.append(dict(thresholds=cuts.copy(), studies=len(yy), model=_ratio(tp - fp * odds, len(yy)), treat_all=_ratio((yy == 1).sum() - (yy == 0).sum() * odds, len(yy)), treat_none=np.zeros(len(cuts))))
    return result

def ece(y, p, bins=15):
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('Positive integer bin count required')
    y, p = _matched(y, p)
    valid = _binary_valid(y) & _prob_valid(p)
    result = []
    for j in range(y.shape[1]):
        yy, pp = (y[valid[:, j], j], p[valid[:, j], j])
        groups = [g for g in np.array_split(np.argsort(pp, kind='stable'), bins) if len(g)]
        result.append(sum((len(g) * abs(pp[g].mean() - yy[g].mean()) for g in groups)) / len(yy) if len(yy) else np.nan)
    return np.asarray(result)

def calibration_slope(y, p, epsilon=1e-06):
    if not 0 < epsilon < 0.5:
        raise ValueError('epsilon must lie in (0,.5)')
    y, p = _matched(y, p)
    valid = _binary_valid(y) & _prob_valid(p)
    results = []
    for j in range(y.shape[1]):
        yy, pp = (y[valid[:, j], j], p[valid[:, j], j])
        row = dict(studies=len(yy), epsilon=epsilon, intercept=np.nan, slope=np.nan, status='undefined_degenerate')
        if np.unique(yy).size == 2 and np.unique(pp).size > 1:
            x = logit(np.clip(pp, epsilon, 1 - epsilon))
            separated = x[yy == 1].min() >= x[yy == 0].max() or x[yy == 0].min() >= x[yy == 1].max()
            if separated:
                row['status'] = 'undefined_separated'
            else:

                def objective(beta):
                    z = beta[0] + beta[1] * x
                    residual = expit(z) - yy
                    return (np.mean(np.logaddexp(0, z) - yy * z), np.array([residual.mean(), (residual * x).mean()]))
                fitted = minimize(objective, [0.0, 1.0], jac=True, method='BFGS', options={'gtol': 1e-09})
                if np.isfinite(fitted.x).all() and np.max(np.abs(fitted.jac)) < 1e-06:
                    row.update(intercept=float(fitted.x[0]), slope=float(fitted.x[1]), status='estimated')
                else:
                    row['status'] = 'undefined_optimizer_failure'
        results.append(row)
    return results

def patient_bootstrap(stat_fn, patient_ids, B=2000, seed=20261115):
    ids = np.asarray(patient_ids)
    if ids.ndim != 1 or len(ids) == 0 or B < 1 or (int(B) != B):
        raise ValueError('Nonempty patient vector and positive integer B required')
    if any((v is None or (isinstance(v, (float, np.floating)) and (not np.isfinite(v))) for v in ids)):
        raise ValueError('Missing patient identifier')
    unique, inverse = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    point = np.asarray(stat_fn(np.arange(len(ids))), float)
    rng, draws = (np.random.default_rng(seed), [])
    for _ in range(int(B)):
        sample = rng.choice(len(unique), size=len(unique), replace=True)
        indices = np.concatenate([groups[i] for i in sample])
        value = np.asarray(stat_fn(indices), float)
        if value.shape != point.shape:
            raise ValueError('Bootstrap statistic shape changed')
        draws.append(value)
    draws = np.asarray(draws)
    finite = np.isfinite(draws)
    lower, upper = (np.full(point.shape, np.nan), np.full(point.shape, np.nan))
    for k in np.ndindex(point.shape):
        values = draws[(slice(None),) + k]
        values = values[np.isfinite(values)]
        if len(values):
            lower[k], upper[k] = np.quantile(values, [0.025, 0.975])
    return dict(point=point, ci95=np.stack([lower, upper], axis=-1), valid_replicates=finite.sum(axis=0), replicates=int(B), seed=seed, patients=len(unique), studies=len(ids), scope='Paired whole-patient bootstrap; fixed fitted models')
