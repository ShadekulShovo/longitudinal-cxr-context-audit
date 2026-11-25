"""Fixed lexical report groups and paired report-group gain contrasts."""
import re
import numpy as np
import models as core
ENGLISH=['unchanged', 'no change', 'stable', 'stably', 'compared', 'comparison', 'prior', 'previous', 'previously', 'interval', 'again', 'persistent', 'persists', 'improved', 'improving', 'worsened', 'worsening', 'increased', 'decreased', 'resolved', 'resolving', 'new']
SPANISH=['sin cambios', 'estable', 'estabilidad', 'comparacion', 'comparado', 'previo', 'previa', 'anterior', 'persiste', 'persistente', 'mejoria', 'empeoramiento', 'aumento', 'disminucion', 'resuelto', 'nuevo', 'nueva']

def comparative_report(text,language='en'):
    if not isinstance(text,str) or not text.strip():raise ValueError('Report text unavailable')
    terms=ENGLISH if language=='en' else SPANISH
    parts=[r'\s+'.join(map(re.escape,t.split())) for t in terms]
    if language=='en':parts.append(r'redemonstrat\w*')
    return bool(re.search(r'(?<!\w)(?:'+'|'.join(parts)+r')(?!\w)',text.lower()))
class RexStatistics(core.Statistics):

    def __call__(self, weights):
        result = super().__call__(weights)
        weights = np.asarray(weights, float)
        c, r = (self.candidate, self.reference)
        mask = self.groups['language:comparative']
        result['U10/comparative_share'] = float(weights @ mask / weights.sum()) if weights.sum() else np.nan
        for group in ['comparative', 'noncomparative']:
            ww = weights * self.groups['language:' + group]
            result['U10/' + group + '/macro_gain'] = result['subsets/language:' + group + '/macro_gain']
            per, aggregate = self.decomposition(c, r, ww)
            for k, v in aggregate.items():
                result['U10/' + group + '/U2/macro/' + k] = v
            for i, f in enumerate(core.FINDINGS):
                for k in ['delta_auroc', 'within_gain', 'across_gain', 'change_share']:
                    result['U10/' + group + '/U2/' + f + '/' + k] = per[i][k]
            for name in [c, r]:
                for rule in ['single', 'stratified']:
                    rates = core.event_rates(self.y, self.prior, self.decisions[name][rule], ww)
                    for key, value in rates.items():
                        path = 'U10/' + group + '/events/' + name + '/' + rule + '/' + key
                        if np.ndim(value) == 0:
                            result[path] = float(value)
                        else:
                            for i, f in enumerate(core.FINDINGS):
                                result[path + '/' + f] = float(value[i])
        result['U10/comparative_minus_noncomparative/macro_gain'] = result['U10/comparative/macro_gain'] - result['U10/noncomparative/macro_gain']
        return result
