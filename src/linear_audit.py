"""Exact linear group contributions and paired decision audit."""
import numpy as np
from scipy.special import expit
from contracts import check
import models as core
EVENTS = ['onset', 'resolution', 'persistent_positive', 'persistent_negative']

def event_masks(y, prior):
    return dict(onset=(prior == 0) & (y == 1), resolution=(prior == 1) & (y == 0), persistent_positive=(prior == 1) & (y == 1), persistent_negative=(prior == 0) & (y == 0))

def linear_decomposition(z, coef, intercept, reference, groups, saved_logits):
    z = np.asarray(z)
    coef = np.asarray(coef)
    reference = np.asarray(reference)
    saved_logits = np.asarray(saved_logits)
    check(z.ndim == 2 and coef.ndim == 2 and (coef.shape[1] == z.shape[1]) and (reference.shape == (z.shape[1],)), 'X1_LINEAR_SHAPE')
    check(np.shape(intercept) == (coef.shape[0],) and saved_logits.shape == (len(z), coef.shape[0]), 'X1_LOGIT_SHAPE')
    check(all((np.isfinite(v).all() for v in [z, coef, intercept, reference, saved_logits])), 'X1_NONFINITE_INPUT')
    indices = [j for g in groups.values() for j in g]
    check(sorted(indices) == list(range(z.shape[1])) and len(indices) == len(set(indices)), 'X1_GROUP_PARTITION')
    contribution = np.stack([(z[:, ii].astype(np.float64) - reference[ii]) @ coef[:, ii].astype(np.float64).T for ii in groups.values()], axis=2)
    baseline = np.asarray(intercept, dtype=float) + reference.astype(float) @ coef.astype(float).T
    delta = np.abs(contribution.sum(2) + baseline - saved_logits)
    error = float(delta.max(initial=0))
    if error > 1e-09:
        exc = core.ProtocolError('X1_SAVED_LOGIT_RECONSTRUCTION_FAILED')
        exc.aggregate_diagnostics = dict(max_absolute_logit_error=error, per_finding_max_absolute_logit_error=delta.max(0).tolist())
        raise exc
    return dict(contribution=contribution, baseline=baseline, max_absolute_logit_error=error, groups=list(groups))

def m4_groups(image_dimension):
    d = image_dimension
    check(type(d) is int and d > 0, 'X1_IMAGE_DIMENSION')
    return dict(image=list(range(d)), last_read=list(range(d, d + 6)), summary=list(range(d + 7, d + 22)), timing_and_count=[d + 6] + list(range(d + 22, d + 35)))

def b2_groups(current_dimension, temporal_dimension):
    d = current_dimension
    t = temporal_dimension
    check(d > 0 and t > 0, 'X1_IMAGE_DIMENSION')
    return dict(current_image=list(range(d)), temporal=list(range(d, d + t)))

def distance_quartiles(y, prior, image_logits, image_st):
    y = np.asarray(y)
    prior = np.asarray(prior)
    score = np.asarray(image_logits)
    st = np.asarray(image_st)
    check(y.shape == prior.shape == score.shape and st.shape == (y.shape[1],), 'X2_VALIDATE_SHAPE')
    distance = np.abs(expit(score) - expit(st))
    eligible = (prior == 1) & (y == 0) & (score < st)
    edges = np.full((y.shape[1], 3), np.nan)
    for j in range(y.shape[1]):
        if eligible[:, j].any():
            edges[j] = np.quantile(distance[eligible[:, j], j], [0.25, 0.5, 0.75], method='linear')
    return (edges, eligible.sum(0))

def safe_ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)

def strict_macro(values):
    v = np.asarray(values, float)
    return float(v.mean()) if np.isfinite(v).all() else np.nan

class LinearAuditStatistics:

    def __init__(self, y, prior, patients, candidate_logits, image_logits, candidate_st, image_st, decompositions, quartiles):
        self.y = np.asarray(y)
        self.prior = np.asarray(prior)
        self.patients = np.asarray(patients)
        self.candidate = np.asarray(candidate_logits)
        self.image = np.asarray(image_logits)
        self.cst = np.asarray(candidate_st)
        self.ist = np.asarray(image_st)
        check(self.y.ndim == 2 and self.y.shape == self.prior.shape == self.candidate.shape == self.image.shape and (len(self.patients) == len(self.y)), 'X12_ARRAY_SHAPE')
        check(np.isin(self.y, [0, 1]).all() and np.isin(self.prior, [0, 1]).all(), 'X12_NONBINARY_TARGET')
        check(np.isfinite(self.candidate).all() and np.isfinite(self.image).all(), 'X12_NONFINITE_LOGIT')
        check(self.cst.shape == self.ist.shape == (self.y.shape[1],) and (not np.isnan(self.cst).any()) and (not np.isnan(self.ist).any()), 'X12_ST_SHAPE')
        self.decomp = decompositions
        check(set(decompositions) == {'train_mean', 'validate_mean'}, 'X1_BOTH_REFERENCES_REQUIRED')
        check(decompositions['train_mean']['groups'] == decompositions['validate_mean']['groups'] and decompositions['train_mean']['groups'][0] in ['image', 'current_image'], 'X1_GROUP_ORDER')
        for d in decompositions.values():
            check(d['contribution'].shape[:2] == self.y.shape and d['max_absolute_logit_error'] <= 1e-09, 'X1_RECONSTRUCTION_REQUIRED')
            check(float(np.max(np.abs(d['contribution'].sum(2) + d['baseline'] - self.candidate), initial=0)) <= 1e-09, 'X1_DIFFERENT_SAVED_LOGITS')
        self.events = event_masks(self.y, self.prior)
        dc = self.candidate >= self.cst
        di = self.image >= self.ist
        cr = dc == self.y
        ir = di == self.y
        self.cells = dict(rescue=~ir & cr, override=ir & ~cr, both_right=ir & cr, both_wrong=~ir & ~cr)
        self.quartiles = np.asarray(quartiles, float)
        check(self.quartiles.shape == (self.y.shape[1], 3), 'X2_QUARTILE_SHAPE')
        check(all((np.isnan(e).all() or (np.isfinite(e).all() and (np.diff(e) >= 0).all()) for e in self.quartiles)), 'X2_QUARTILE_INVALID')
        self.distance = np.abs(expit(self.image) - expit(self.ist))
        self.eligible = self.events['resolution'] & ir

    def __call__(self, weights):
        w = np.asarray(weights, float)
        check(w.shape == (len(self.y),) and np.isfinite(w).all() and (w >= 0).all(), 'X12_WEIGHTS')
        out = {}
        total = w.sum()

        def put(path, array, macro=True):
            a = np.asarray(array, float)
            if a.ndim == 0:
                out[path] = float(a)
            else:
                for j, v in enumerate(a):
                    out[path + '/finding_' + str(j)] = float(v)
                if macro:
                    out[path + '/macro'] = strict_macro(a)
        shares = {}
        scopes = {'all': np.ones_like(self.y, bool), 'prior_negative': self.prior == 0, 'prior_positive': self.prior == 1, **self.events}
        for ref, d in self.decomp.items():
            c = d['contribution']
            names = d['groups']
            means = {}
            for scope, mask in scopes.items():
                absolute = np.einsum('n,nf,nfg->fg', w, mask, np.abs(c))
                fractions = safe_ratio(absolute, absolute.sum(1, keepdims=True))
                for g, name in enumerate(names):
                    put('X1/' + ref + '/' + scope + '/group_share/' + name, fractions[:, g])
                if scope == 'all':
                    shares[ref] = fractions
            image = c[:, :, 0]
            history = c[:, :, 1:].sum(2)
            conflict = image * history < 0
            win = conflict & ((self.candidate - self.cst) * history > 0)
            image_win = conflict & ((self.candidate - self.cst) * image > 0)
            put('X1/' + ref + '/conflict/rate', safe_ratio(w @ conflict, total))
            put('X1/' + ref + '/conflict/history_wins_rate', safe_ratio(w @ win, w @ conflict))
            put('X1/' + ref + '/conflict/image_wins_rate', safe_ratio(w @ image_win, w @ conflict))
            put('X1/' + ref + '/conflict/threshold_tie_rate', safe_ratio(w @ (conflict & (self.candidate == self.cst)), w @ conflict))
            right = (self.candidate >= self.cst) == self.y
            for event in ['onset', 'resolution']:
                for name, mask in [('history_wins', win), ('image_wins', image_win)]:
                    selected = mask & self.events[event]
                    put('X1/' + ref + '/conflict/' + event + '/' + name + '/accuracy', safe_ratio(w @ (selected & right), w @ selected))
                    put('X1/' + ref + '/conflict/' + event + '/' + name + '/count', w @ selected)
        a, b = (shares['train_mean'], shares['validate_mean'])
        for i in range(a.shape[1]):
            for j in range(i + 1, a.shape[1]):
                comparable = np.isfinite(a[:, i]) & np.isfinite(a[:, j]) & np.isfinite(b[:, i]) & np.isfinite(b[:, j])
                flip = (np.sign(a[:, i] - a[:, j]) != np.sign(b[:, i] - b[:, j])).astype(float)
                flip[~comparable] = np.nan
                put('X1/reference_rank_disagreement/' + self.decomp['train_mean']['groups'][i] + '_vs_' + self.decomp['train_mean']['groups'][j], flip)
        for event, mask in self.events.items():
            count = w @ mask
            put('X2/' + event + '/event_count', count)
            for name, cell in self.cells.items():
                hit = w @ (mask & cell)
                put('X2/' + event + '/' + name + '/count', hit)
                put('X2/' + event + '/' + name + '/rate', safe_ratio(hit, count))
            net = w @ (mask & self.cells['rescue']) - w @ (mask & self.cells['override'])
            put('X2/' + event + '/net_rescues_per1000_studies', safe_ratio(1000 * net, total))
        for q in range(4):
            mask = np.zeros_like(self.y, bool)
            for j, edge in enumerate(self.quartiles):
                if np.isfinite(edge).all():
                    mask[:, j] = self.eligible[:, j] & (np.searchsorted(edge, self.distance[:, j], side='right') == q)
            denom = w @ mask
            hit = w @ (mask & self.cells['override'])
            valid = np.isfinite(self.quartiles).all(1)
            count = denom.astype(float)
            count[~valid] = np.nan
            hit = hit.astype(float)
            hit[~valid] = np.nan
            put('X2/resolution/distance_quartile_' + str(q + 1) + '/eligible_count', count)
            put('X2/resolution/distance_quartile_' + str(q + 1) + '/override_count', hit)
            put('X2/resolution/distance_quartile_' + str(q + 1) + '/override_rate', safe_ratio(hit, count))
            put('X2/resolution/distance_quartile_' + str(q + 1) + '/override_share', safe_ratio(hit, w @ (self.events['resolution'] & self.cells['override'])))
        return out
