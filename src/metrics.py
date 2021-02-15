import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def relative_prediction_error(y_true, y_pred):
    initial_sample =((y_true - y_pred) / y_true) * 100

    mean = np.mean(initial_sample, axis=0)
    sd = np.std(initial_sample, axis=0)

    sample = [x for x in initial_sample if (x > mean - 2 * sd)]
    sample = [x for x in sample if (x < mean + 2 * sd)]
    return sample, initial_sample

