import os
import scipy.io as spio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def importar_dados(filepath, insert_ones=True):
    path = os.getcwd() + filepath
    data = spio.loadmat(path, squeeze_me=True)

    y = np.array(data['y'])
    y.shape = (len(y), 1)
    X = np.array(data['X'])

    ytest = np.array(data['ytest'])
    ytest.shape = (len(ytest), 1)
    Xtest = np.array(data['Xtest'])

    yval = np.array(data['yval'])
    yval.shape = (len(yval), 1)
    Xval = np.array(data['Xval'])

    # Incluir o valor de 1, pois theta0 = 1
    if insert_ones:
        X = np.c_[np.ones((X.shape[0], 1)), X]
        Xtest = np.c_[np.ones((Xtest.shape[0], 1)), Xtest]
        Xval = np.c_[np.ones((Xval.shape[0], 1)), Xval]

    return X, y, Xtest, ytest, Xval, yval


def plot(X, y, theta=[],filename='target/plot5.1.png'):
    plt.scatter(X.T[1], y, color='red', marker='x')
    plt.title('Fluidez da barragem')
    plt.xlabel('Mudanca no nivel da agua')
    plt.ylabel('fluxo da barragem')

    if len(theta) > 0:
        plt.plot(X[:,1],X.dot(theta.reshape(-1,1)).flatten())

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)
    plt.show()
