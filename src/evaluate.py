"""Evaluation and metrics for MG-STGNN."""

import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def peak_rmse(y_true, y_pred, percentile=90):
    threshold = np.percentile(y_true, percentile)
    mask = y_true >= threshold
    return rmse(y_true[mask], y_pred[mask])

def macro_rmse(y_true, y_pred, country_ids):
    country_rmses = []
    for c in np.unique(country_ids):
        mask = country_ids == c
        country_rmses.append(rmse(y_true[mask], y_pred[mask]))
    return np.mean(country_rmses)

def picp(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))

def pinaw(lower, upper, y_range):
    return np.mean(upper - lower) / y_range
