import numpy as np


def poly_features(X, p):
    X_poly = X
    offset = 1

    if (len(X_poly.shape) == 1):
        X_poly.shape = (len(X_poly), 1)
        offset = 0

    for k in range(2, p+1):
        X_poly = np.column_stack((X_poly, np.power(X_poly[:,offset].reshape(len(X_poly),1), k)))

    return X_poly

