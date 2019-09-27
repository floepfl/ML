# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss_MSE(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    return (1/(2 * (tx.shape[0])) * np.sum((y - tx @ w.T)**2))

def compute_loss_MAE(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    return (1/(2 * (tx.shape[0]))) * np.sum(np.abs((y - tx @ w.T)))
