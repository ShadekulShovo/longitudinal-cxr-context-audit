import numpy as np
from sklearn.preprocessing import StandardScaler
def scale_current(x, mean, scale):
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = (mean, scale)
    scaler.var_, scaler.n_features_in_ = (np.asarray(scale) ** 2, len(mean))
    return scaler.transform(np.asarray(x, dtype=np.float32))

def predict_linear(path, features):
    with np.load(path, allow_pickle=False) as z:
        x = scale_current(features, z['current_mean'], z['current_scale'])
        return x @ z['coef'].T + z['intercept']
