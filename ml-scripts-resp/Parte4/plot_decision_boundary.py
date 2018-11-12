import os
import numpy as np

import matplotlib.pyplot as plt
from Parte4.map_feature import map_feature

def plot(data, out_img, theta=[]):
    positivo = data[data["Rejeicao"].isin([1])]
    negativo = data[data["Rejeicao"].isin([0])]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positivo['T1'], positivo['T2'], s=50, c='k', marker='+', label='Aceito')
    ax.scatter(negativo['T1'], negativo['T2'], s=50, c='y', marker='o', label='Rejeitado')
    ax.set_xlabel('Teste 1')
    ax.set_ylabel('Teste 2')

    if len(theta) > 0:
        xvals = np.linspace(-1, 1.5, 50)
        yvals = np.linspace(-1, 1.5, 50)
        zvals = np.zeros((len(xvals), len(yvals)))
        for i in range(len(xvals)):
            for j in range(len(yvals)):
                Z = np.array(xvals[i])
                Z = np.c_[Z, np.array(yvals[j])]
                Z_map = map_feature(Z)
                zvals[i][j] = np.dot(theta, Z_map.T)
        zvals = zvals.transpose()

        ctr = ax.contour(xvals, yvals, zvals, [0], colors='g')
        plt.clabel(ctr, fmt='Fronteira')

    ax.legend()

    if not os.path.exists(os.path.dirname(out_img)):
        os.makedirs(os.path.dirname(out_img))

    plt.savefig(out_img)
    plt.show()
