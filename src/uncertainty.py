"""Uncertainty quantification: MC Dropout + conformal prediction."""

import numpy as np


def mc_dropout_predict(model, x, n_passes=50):
    """Run MC Dropout inference for uncertainty estimation."""
    # TODO: Implement MC Dropout inference
    # 1. Enable dropout at inference
    # 2. Run n_passes forward passes
    # 3. Return mean and std of predictions
    raise NotImplementedError


def conformal_calibrate(residuals, sigma, alpha=0.10):
    """Compute conformal quantiles from validation residuals."""
    scores = np.abs(residuals) / sigma
    q = np.quantile(scores, 1 - alpha)
    return q
