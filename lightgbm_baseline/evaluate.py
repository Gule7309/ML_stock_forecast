import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def average_return(labels, returns):
    return float(np.mean((2 * np.asarray(labels) - 1) * np.asarray(returns)))


def metrics(y, returns, labels, probabilities):
    return dict(ar=average_return(labels, returns),
                accuracy=float(accuracy_score(y, labels)),
                log_loss=float(log_loss(y, probabilities, labels=[0, 1])))
