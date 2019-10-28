import numpy as np
# -*- coding: utf-8 -*-
"""A function to compute the cost."""


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_rmse(y, tx, w):
    return np.sqrt(2 * compute_mse(y, tx, w))