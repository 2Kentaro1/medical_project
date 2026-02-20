from sklearn.metrics import fbeta_score
import numpy as np

def find_best_threshold(y_true, y_prob, beta=7):
    best_t = 0.5
    best_f = 0

    for t in np.linspace(0.01, 0.5, 100):
        y_pred = (y_prob >= t).astype(int)
        f = fbeta_score(y_true, y_pred, beta=beta)
        if f > best_f:
            best_f = f
            best_t = t

    return best_t, best_f