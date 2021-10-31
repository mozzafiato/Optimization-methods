# -*- coding: utf-8 -*-
"""Lab 3.

Gradient descent
"""

import numpy as np

def calculate_mse(b, A, x):
    """Calculate the mean squared error for vector e."""
    # computing the prediction
    y_pred = np.dot(A, x)
    # subtract the differences
    diff = np.subtract(y_pred, b)
    # return MSE
    return np.square(diff).mean() / 2

def compute_gradient(b, A, x):
    """Compute the gradient."""
    # computing the prediction
    h = np.dot(A, x)
    #  compute gradient
    grad = 2 * np.dot(A.T, (h - b)) / len(b)
    return grad

def gradient_descent(b, A, initial_x, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    objectives = []
    x = initial_x
    for n_iter in range(max_iters):
        # ***************************************************
        # compute gradient and objective function
        # ***************************************************
        grad = compute_gradient(b, A, x)
        obj = calculate_mse(b, A, x)
        # ***************************************************
        # update x by a gradient descent step
        # ***************************************************
        x = x - (gamma * grad)
        
        # store x and objective function value
        xs.append(x)
        objectives.append(obj)
        print("Gradient Descent({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))

    return objectives, xs