# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    #l = np.diag([lambda_]*tx.shape[1])
    l = 2 * y.shape[0] * lambda_ *  np.identity(tx.shape[1])
    a = tx.T.dot(tx) + l
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)