import numpy as np
from scipy.optimize import minimize


def cost_function(theta, _lambda, X, y):
    m = X.shape[0]
    h = X.dot(theta.reshape(-1,1))

    dif = h - y
    grad = dif.T.dot(dif)

    theta_chunk = theta[1:]
    reg = _lambda / (2 * m) * theta_chunk.T.dot(theta_chunk.reshape(-1,1))

    cost = (1 / (2 * m)) * grad + reg

    return float(cost.max())


def gd(theta, _lambda, X, y):
    m = y.size
    h = X.dot(theta.reshape(-1, 1))

    reg = (_lambda / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    grad = (1 / m) * X.T.dot(h - y) + reg

    return grad.flatten()


def compute(theta, _lambda, X, y):
    opt = {'maxiter': 1000}
    return minimize(cost_function, theta, args=(_lambda, X, y), jac=gd, options=opt).x
