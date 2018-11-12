import numpy as np
from Parte3.sigmoide import sigmoide


def cost_function_reg(theta, _lambda, X, y):
   m = y.size
   theta = theta.T

   reg = (_lambda / (2 * m)) * np.square(theta[1:]).sum()

   gx = sigmoide(X.dot(theta))
   grad0 = np.log(gx).T.dot(y)
   grad = np.log(1 - gx).T.dot(1 - y)
   J = -(1 / m) * (grad0 + grad) + reg

   return (J[0])


def gd_reglog(theta, _lambda, X, y):
   m = y.size
   h = sigmoide(X.dot(theta.reshape(-1, 1)))

   reg = (_lambda / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
   grad = (1 / m) * X.T.dot(h - y) + reg

   return grad.flatten()

