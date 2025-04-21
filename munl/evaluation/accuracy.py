import numpy as np
import sklearn.metrics as skmet
from numpy import ndarray


def compute_accuracy(targets: ndarray, predictions: ndarray) -> float:
    assert targets.ndim == 1
    assert predictions.ndim == 1
    assert targets.shape == predictions.shape
    accuracy_score = skmet.accuracy_score(targets, predictions)
    return accuracy_score
