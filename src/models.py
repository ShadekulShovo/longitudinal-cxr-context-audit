"""History features, matched replacements, linear models and resumable inference."""
import json, itertools, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import metrics
CG=[.01,.1,1.]
BOOT_SEED,SWAP_SEED=20261115,20261116
class ProtocolError(ValueError):
    pass

def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value

def write_json(path, value, mutable=False):
    path = Path(path)
    if path.exists() and (not mutable):
        raise FileExistsError('OUTPUT_EXISTS')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{time.time_ns()}.pending')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write('\n')
    if path.exists() and (not mutable):
        raise FileExistsError('OUTPUT_EXISTS')
    tmp.replace(path)

def assert_disjoint(patients, splits):
    sets = {k: set(patients[splits == k]) for k in ['train', 'validate', 'test']}
    if any((sets[a] & sets[b] for a, b in itertools.combinations(sets, 2))):
        raise ProtocolError('PATIENT_SPLIT_OVERLAP')

def history_indices(studies):
    patients = studies.PatientID.to_numpy()
    days = studies['_day'].to_numpy()
    earlier = [np.empty(0, np.int64) for _ in range(len(studies))]
    for positions in studies.groupby('PatientID', sort=False).indices.values():
        positions = np.asarray(positions)
        for i in positions:
            prior = positions[days[positions] < days[i]]
            if not np.all(patients[prior] == patients[i]) or not np.all(days[prior] < days[i]):
                raise ProtocolError('HISTORY_LEAKAGE')
            earlier[i] = prior
    return earlier

def history_block(index, earlier, y, days):
    earlier = np.asarray(earlier, int)
    if not len(earlier) or not np.all(days[earlier] < days[index]):
        raise ProtocolError('HISTORY_NOT_STRICTLY_EARLIER')
    history_days = days[earlier]
    unique = np.unique(history_days)
    last_rows = earlier[history_days == unique[-1]]
    last = y[last_rows].max(0)
    previous = y[earlier[history_days == unique[-2]]].max(0) if len(unique) > 1 else last
    gaps = []
    for j in range(y.shape[1]):
        positive = history_days[y[earlier, j] > 0]
        gaps += [np.log1p(days[index] - positive.max()) if len(positive) else 0.0, float(len(positive) > 0)]
    return np.r_[last, np.log1p(days[index] - unique[-1]), np.log1p(len(last_rows)), y[earlier].max(0), y[earlier].mean(0), last - previous, gaps, np.log1p(len(earlier)), np.log1p(days[index] - unique[0]), float(len(unique) > 1)].astype(np.float64)

def prior_labels(index, earlier, y, days):
    e = np.asarray(earlier, int)
    return y[e[days[e] == days[e].max()]].max(0)

def binary_auc(y, score, weight=None):
    return RankedAUC(y, score)(np.ones(len(y)) if weight is None else weight)

class RankedAUC:

    def __init__(self, y, score):
        y, score = (np.asarray(y), np.asarray(score))
        if y.ndim != 1 or score.shape != y.shape or (not np.isfinite(score).all()):
            raise ProtocolError('INVALID_AUC_INPUT')
        if not np.isin(y, [0, 1]).all():
            raise ProtocolError('NONBINARY_TARGET')
        self.order = np.argsort(score, kind='stable')
        self.y = y[self.order]
        ordered = score[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1] if len(y) else np.array([], int)

    def totals(self, weight):
        w = np.asarray(weight, float)[self.order]
        if len(w) == 0:
            return (0.0, 0.0, 0.0)
        pos = np.add.reduceat(w * self.y, self.starts)
        neg = np.add.reduceat(w * (1 - self.y), self.starts)
        wins = float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)))
        return (wins, float(pos.sum()), float(neg.sum()))

    def __call__(self, weight):
        wins, p, n = self.totals(weight)
        return wins / (p * n) if p > 0 and n > 0 else np.nan

def macro(values):
    a = np.asarray(values, float)
    return float(a.mean()) if len(a) == 5 and np.isfinite(a).all() else np.nan

def fit_linear(x_train, y_train, x_validate, y_validate):
    if not all((np.isfinite(x).all() for x in [x_train, x_validate])):
        raise ProtocolError('NONFINITE_DESIGN')
    if y_train.shape[1] != 5 or any((np.unique(y_train[:, j]).size != 2 for j in range(5))):
        raise ProtocolError('TRAIN_TARGET_CLASS_MISSING')
    scaler = StandardScaler().fit(x_train)
    xtr, xva = (scaler.transform(x_train), scaler.transform(x_validate))
    best, screen = (None, [])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for c in CG:
            try:
                models = [LogisticRegression(C=c, solver='lbfgs', max_iter=3000, random_state=11).fit(xtr, y_train[:, j]) for j in range(5)]
            except ConvergenceWarning:
                raise ProtocolError('LOGISTIC_FIT_DID_NOT_CONVERGE') from None
            coef = np.concatenate([v.coef_ for v in models])
            intercept = np.array([v.intercept_[0] for v in models])
            score = macro([binary_auc(y_validate[:, j], (xva @ coef.T + intercept)[:, j]) for j in range(5)])
            if not np.isfinite(score):
                raise ProtocolError('VALIDATE_TARGET_CLASS_MISSING')
            screen.append(dict(C=c, validate_macro=score))
            if best is None or score > best['validate_macro']:
                best = dict(C=c, validate_macro=score, coef=coef, intercept=intercept, mean=scaler.mean_, scale=scaler.scale_)
    best['screen'] = screen
    return best

def predict(head, design):
    return (design - head['mean']) / head['scale'] @ head['coef'].T + head['intercept']

def select_donors(keys, patients, seed=SWAP_SEED, rng=None):
    rng, selected = (np.random.default_rng(seed) if rng is None else rng, [])
    for k, patient in zip(keys, patients):
        candidates = np.flatnonzero((keys == k) & (patients != patient))
        if not len(candidates):
            raise ProtocolError('MATCHED_CROSS_PATIENT_DONOR_UNAVAILABLE')
        selected.append(int(rng.choice(candidates)))
    selected = np.asarray(selected)
    if not np.all(keys[selected] == keys) or not np.all(patients[selected] != patients):
        raise ProtocolError('DONOR_CONTRACT_FAILED')
    return selected

def matching_keys(days, earlier, rows):
    gaps = np.array([days[i] - days[earlier[i]].max() for i in rows])
    counts = np.array([len(earlier[i]) for i in rows])
    return np.digitize(gaps, [3.5, 14.5, 60.5]) * 3 + np.digitize(counts, [1.5, 4.5])

def thresholds_from_validate(y, scores, prior):
    return dict(single=np.array([metrics.youden_threshold(y[:, j], scores[:, j]) for j in range(5)]), stratified=metrics.pst_thresholds(y, scores, prior))

def bootstrap_report(statistics, checkpoint, B, status, resume=False, seed=BOOT_SEED):
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    n = len(statistics.y)
    point = statistics(np.ones(n))
    keys = sorted(point)
    unique, inverse = np.unique(statistics.patients, return_inverse=True)
    schema = dict(keys=keys, B=B, seed=seed, patients=len(unique), studies=n)
    manifest, state = (checkpoint / 'schema.json', checkpoint / 'state.json')
    path = checkpoint / 'draws.npy'
    if manifest.exists():
        if not resume or json.loads(manifest.read_text()) != schema:
            raise ProtocolError('BOOTSTRAP_CHECKPOINT_MISMATCH')
        draws = np.load(path, mmap_mode='r+')
        if draws.shape != (B, len(keys)):
            raise ProtocolError('BOOTSTRAP_ARRAY_SHAPE_CHANGED')
        done = json.loads(state.read_text())['done']
    else:
        if path.exists() or state.exists():
            raise ProtocolError('UNEXPLAINED_BOOTSTRAP_CHECKPOINT')
        draws = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(B, len(keys)))
        draws[:] = np.nan
        draws.flush()
        write_json(manifest, schema)
        done = 0
        write_json(state, dict(done=0))
    rng = np.random.default_rng(seed)
    for i in range(B):
        multiplicities = np.bincount(rng.choice(len(unique), len(unique), replace=True), minlength=len(unique))
        if i < done:
            continue
        row = statistics(multiplicities[inverse])
        if sorted(row) != keys:
            raise ProtocolError('BOOTSTRAP_STATISTIC_SCHEMA_CHANGED')
        draws[i] = [row[k] for k in keys]
        if (i + 1) % 10 == 0 or i + 1 == B:
            draws.flush()
            write_json(state, dict(done=i + 1), mutable=True)
            status('bootstrap', replicates_done=i + 1, replicates=B)
    report = {}
    for j, k in enumerate(keys):
        values = np.asarray(draws[:, j])
        values = values[np.isfinite(values)]
        ci = np.quantile(values, [0.025, 0.975]) if len(values) else [np.nan, np.nan]
        report[k] = dict(point=point[k], ci95=ci, valid_replicates=len(values), reliable=bool(len(values) >= 0.99 * B and np.isfinite(point[k])))
    return dict(metrics=report, bootstrap=schema, law='Uniform whole-patient resampling; all paired contrasts and subgroups share each draw; fixed fitted models')

from scipy.special import expit
FINDINGS=['Cardiomegaly','Edema','Effusion','Atelectasis','Opacity']
LEVELS=['1','2','4','8','all']
def event_rates(y, prior, decision, weight):
    masks = [((prior == 0) & (y == 1), decision == 1), ((prior == 1) & (y == 0), decision == 0), (y == 1, decision == 1), (y == 0, decision == 0)]
    result = {}
    for name, (event, hit) in zip(['onset_sensitivity', 'resolution_specificity', 'sensitivity', 'specificity'], masks):
        denom = weight @ event
        numer = weight @ (event & hit)
        result[name] = np.divide(numer, denom, out=np.full(5, np.nan), where=denom > 0)
        result[name + '_pooled'] = numer.sum() / denom.sum() if denom.sum() else np.nan
    return result

def cohort_counts(y, prior, patients, mask):
    return dict(studies=int(mask.sum()), patients=int(len(np.unique(patients[mask]))), positives=y[mask].sum(0), onset_events=((prior[mask] == 0) & (y[mask] == 1)).sum(0), resolution_events=((prior[mask] == 1) & (y[mask] == 0)).sum(0))

class Statistics:

    def __init__(self, y, prior, patients, scores, cuts, candidate, reference, groups, swaps=None, dose=None, counts=None):
        self.y, self.prior, self.patients = (y, prior, patients)
        self.scores, self.cuts = (scores, cuts)
        self.candidate, self.reference = (candidate, reference)
        self.groups, self.swaps, self.dose, self.counts = (groups, swaps, dose, counts)
        self.rank = {name: [RankedAUC(y[:, j], value[:, j]) for j in range(5)] for name, value in scores.items()}
        self.prob = {name: expit(value) for name, value in scores.items()}
        self.decisions = {name: {kind: metrics.apply_thresholds(value, cuts[name][kind], prior if kind == 'stratified' else None) for kind in ['single', 'stratified']} for name, value in scores.items()}
        self.extra_ranks = {}
        if swaps:
            self.extra_ranks.update({name: [RankedAUC(y[:, j], value[:, j]) for j in range(5)] for name, value in swaps.items()})
        if dose:
            self.extra_ranks.update({'dose_' + k: [RankedAUC(y[:, j], value[:, j]) for j in range(5)] for k, value in dose.items()})

    def auc(self, name, weights, prior_state=None):
        return np.array([self.rank[name][j](weights if prior_state is None else weights * (self.prior[:, j] == prior_state)) for j in range(5)])

    def decomposition(self, a, b, w):
        rows = []
        for j in range(5):
            total = []
            for name in [a, b]:
                all_wins, p, n = self.rank[name][j].totals(w)
                within_wins, within_pairs = (0.0, 0.0)
                for s in [0, 1]:
                    wins, ps, ns = self.rank[name][j].totals(w * (self.prior[:, j] == s))
                    within_wins += wins
                    within_pairs += ps * ns
                pairs = p * n
                total.append(dict(AUROC=all_wins / pairs if pairs else np.nan, p=within_pairs / pairs if pairs else np.nan, A_within=within_wins / within_pairs if within_pairs else 0.0, A_across=(all_wins - within_wins) / (pairs - within_pairs) if pairs > within_pairs else 0.0, within_component=within_wins / pairs if pairs else np.nan, across_component=(all_wins - within_wins) / pairs if pairs else np.nan))
            da, db = total
            delta = da['AUROC'] - db['AUROC']
            within = da['within_component'] - db['within_component']
            across = da['across_component'] - db['across_component']
            rows.append(dict(delta_auroc=delta, within_gain=within, across_gain=across, change_share=within / delta if np.isfinite(delta) and abs(delta) > 1e-12 else np.nan, model_a=da, model_b=db))
        d = macro([r['delta_auroc'] for r in rows])
        win = macro([r['within_gain'] for r in rows])
        return (rows, dict(delta_auroc=d, within_gain=win, across_gain=macro([r['across_gain'] for r in rows]), change_share=win / d if np.isfinite(d) and abs(d) > 1e-12 else np.nan))

    def __call__(self, weights):
        w = np.asarray(weights, float)
        values = {}

        def put(path, value):
            a = np.asarray(value)
            if a.ndim == 0:
                values[path] = float(a)
            else:
                for j, f in enumerate(FINDINGS):
                    values[path + '/' + f] = float(a[j])
        c, r = (self.candidate, self.reference)
        rates = {}
        for name in self.scores:
            auc = self.auc(name, w)
            put('AUROC/' + name + '/per_finding', auc)
            put('AUROC/' + name + '/macro', macro(auc))
            for kind, decision in self.decisions[name].items():
                rates[name, kind] = event_rates(self.y, self.prior, decision, w)
                for metric, value in rates[name, kind].items():
                    put('events/' + name + '/' + kind + '/' + metric, value)
                study_weight = w.sum()
                for event_name, mask, miss in [('missed_onsets', (self.prior == 0) & (self.y == 1), decision == 0), ('false_persistences', (self.prior == 1) & (self.y == 0), decision == 1)]:
                    per = w @ (mask & miss) * 1000 / study_weight if study_weight else np.full(5, np.nan)
                    put('per1000/' + name + '/' + kind + '/' + event_name + '/per_finding', per)
                    put('per1000/' + name + '/' + kind + '/' + event_name + '/summed', np.sum(per))
        for name in self.scores:
            if name != r:
                put('contrasts/' + name + '_minus_' + r + '/macro', macro(self.auc(name, w)) - macro(self.auc(r, w)))
        for label, aa, bb in [('ST', (c, 'single'), (r, 'single')), ('PST_vs_reference_ST', (c, 'stratified'), (r, 'single')), ('candidate_PST_vs_ST', (c, 'stratified'), (c, 'single')), ('reference_PST_vs_ST', (r, 'stratified'), (r, 'single'))]:
            for metric in rates[aa]:
                put('threshold_contrasts/' + label + '/' + metric, rates[aa][metric] - rates[bb][metric])
        if 'M0' in self.scores:
            put('contrasts/' + c + '_minus_M0/macro', macro(self.auc(c, w)) - macro(self.auc('M0', w)))
            for kind in ['single', 'stratified']:
                for metric in rates[c, kind]:
                    put('A8_vs_M0/' + kind + '/' + metric, rates[c, kind][metric] - rates['M0', kind][metric])
        changed = np.any(self.y != self.prior, axis=1)
        for key, mask in [('changed', changed), ('unchanged', ~changed)]:
            put('A8/' + key + '/macro_gain', macro(self.auc(c, w * mask)) - macro(self.auc(r, w * mask)))
        for s in [0, 1]:
            delta = self.auc(c, w, s) - self.auc(r, w, s)
            put('A8b/prior_' + str(s) + '/per_finding_gain', delta)
            put('A8b/prior_' + str(s) + '/macro_gain', macro(delta))
        rows, aggregate = self.decomposition(c, r, w)
        for key, value in aggregate.items():
            put('U2/macro/' + key, value)
        for j, f in enumerate(FINDINGS):
            for key, value in rows[j].items():
                if isinstance(value, dict):
                    for kk, vv in value.items():
                        put('U2/' + f + '/' + key + '/' + kk, vv)
                else:
                    put('U2/' + f + '/' + key, value)
        for group, mask in self.groups.items():
            ww = w * mask
            put('subsets/' + group + '/macro_gain', macro(self.auc(c, ww)) - macro(self.auc(r, ww)))
            ca = event_rates(self.y, self.prior, self.decisions[c]['stratified'], ww)
            re = event_rates(self.y, self.prior, self.decisions[r]['single'], ww)
            for metric in ['onset_sensitivity', 'resolution_specificity']:
                put('subsets/' + group + '/' + metric, ca[metric + '_pooled'] - re[metric + '_pooled'])
            if group.startswith('method:'):
                put('subsets/' + group + '/per_finding_auc_gain', self.auc(c, ww) - self.auc(r, ww))
                for metric in ['onset_sensitivity', 'resolution_specificity']:
                    put('subsets/' + group + '/' + metric + '/per_finding', ca[metric] - re[metric])
                rr, mm = self.decomposition(c, r, ww)
                for key, value in mm.items():
                    put('subsets/' + group + '/U2/macro/' + key, value)
                for j, f in enumerate(FINDINGS):
                    for key in ['delta_auroc', 'within_gain', 'across_gain', 'change_share']:
                        put('subsets/' + group + '/U2/' + f + '/' + key, rr[j][key])
        for dimension in ['sex:', 'age:', 'projection:']:
            names = [g for g in self.groups if g.startswith(dimension)]
            for a, b in itertools.combinations(names, 2):
                for metric in ['macro_gain', 'onset_sensitivity', 'resolution_specificity']:
                    put('subgroup_differences/' + a + '_minus_' + b + '/' + metric, values['subsets/' + a + '/' + metric] - values['subsets/' + b + '/' + metric])
        if self.swaps:
            h = np.abs(self.prob[c] - expit(self.swaps['history_swap']))
            im = np.abs(self.prob[c] - expit(self.swaps['image_swap']))
            H = w @ h / w.sum() if w.sum() else np.full(5, np.nan)
            I = w @ im / w.sum() if w.sum() else np.full(5, np.nan)
            put('A7/H/per_finding', H)
            put('A7/I/per_finding', I)
            put('A7/H/macro', np.mean(H))
            put('A7/I/macro', np.mean(I))
            put('A7/image_share/per_finding', np.divide(I, I + H, out=np.full(5, np.nan), where=I + H > 0))
            put('A7/image_share/macro', I.mean() / (I.mean() + H.mean()) if I.mean() + H.mean() > 0 else np.nan)
            original = self.decisions[c]['single']
            for name, arr in self.swaps.items():
                decision = metrics.apply_thresholds(arr, self.cuts[c]['single'])
                put('A7/' + name + '/decision_crossing_share', float(w @ np.mean(original != decision, axis=1) / w.sum()) if w.sum() else np.nan)
                auc = np.array([rank(w) for rank in self.extra_ranks[name]])
                put('A7/' + name + '/macro_auroc', macro(auc))
        if self.dose:
            long = self.counts >= 8
            wl = w * long
            gains, xx = ([], [])
            for k in LEVELS:
                auc = np.array([rank(wl) for rank in self.extra_ranks['dose_' + k]])
                gain = macro(auc) - macro(self.auc(r, wl))
                put('A9/' + k + '/macro_auroc', macro(auc))
                put('A9/' + k + '/gain', gain)
                gains.append(gain)
                xx.append(float(wl @ np.log2(np.maximum(self.counts, 1)) / wl.sum()) if k == 'all' and wl.sum() else np.log2(int(k)) if k != 'all' else np.nan)
            slope = float(np.polyfit(xx, gains, 1)[0]) if np.isfinite(gains).all() and np.isfinite(xx).all() else np.nan
            put('A9/slope_per_doubling', slope)
        return values
