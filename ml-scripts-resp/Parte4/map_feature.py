import numpy as np


def map_feature(X):
    mtx = X.T
    x1 = mtx[0]
    x2 = mtx[1]

    featured = np.ones(x1.shape)

    for i in range(1, 6 + 1):
        for j in range(0, i + 1):
            featured = np.r_['-1,2,0', featured, (x1 ** (i - j)) * (x2 ** j)]

    return featured
