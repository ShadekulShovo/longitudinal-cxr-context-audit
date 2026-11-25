import numpy as np
MAXG=np.array([3,4,4,4,4,4,4,4])
def ordinal(g):
    return np.concatenate([(g[j] >= np.arange(1, MAXG[j] + 1)).astype(np.float32) for j in range(8)])
def graded_history(cur, earlier, date, grade):
    assert earlier and all((date[u] < cur for u in earlier)), 'HISTORY_NOT_STRICTLY_EARLIER'
    by = {}
    for u in earlier:
        by.setdefault(date[u], []).append(grade[u])
    ds = sorted(by)
    last = np.max(by[ds[-1]], 0)
    prev = np.max(by[ds[-2]], 0) if len(ds) > 1 else last
    allg = np.array([grade[u] for u in earlier])
    gap = (cur - ds[-1]).days
    m3 = np.concatenate([ordinal(last), [np.log1p(gap), np.log1p(len(by[ds[-1]]))]])
    sp = []
    for j in range(8):
        p = [d for d in ds if np.max(by[d], 0)[j] > 0]
        sp += [np.log1p((cur - p[-1]).days) if p else 0.0, float(bool(p))]
    m4 = np.concatenate([m3, allg.max(0) / MAXG, (allg > 0).mean(0), (last - prev) / MAXG, sp, [np.log1p(len(earlier)), np.log1p((cur - ds[0]).days), float(len(ds) > 1)]])
    return (m3.astype(np.float32), m4.astype(np.float32), ds)
