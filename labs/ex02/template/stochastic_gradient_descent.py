# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
def compute_stoch_gradient(y, tx, w):
    dist = (y - tx @ w.T)
    dist[dist < 0] = -1
    dist[dist > 0] = 1
    return 1/(- 2 * (tx.shape[0])) * np.sum(dist[:, np.newaxis] * tx, axis=0)

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_s, x_s in batch_iter(y, tx, batch_size):
            loss = compute_stoch_gradient(y_s, x_s, w)
            w = w - gamma * loss
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
