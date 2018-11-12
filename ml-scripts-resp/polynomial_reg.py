import warnings
import os
import numpy as np
import matplotlib.pyplot as plt

from Parte2.normalizacao import normalizar_caracteristica
from Parte5.plot_ex5data1 import importar_dados
from Parte5 import linear_reg_cost_function
from Parte6 import learning_curve as lc
from Parte7.poly_features import poly_features

def plot(X, y, theta, mu, sigma):
    filename = 'target/plot5.1.png'
    plt.scatter(X.T[1], y, color='red', marker='x')
    plt.title('Fluidez da barragem')
    plt.xlabel('Mudanca no nivel da agua')
    plt.ylabel('fluxo da barragem')


    min_x = X.min()
    max_x = X.max()
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05))
    X_poly = poly_features(x, 8)
    X_poly = X_poly - mu
#    X_poly = X_poly / sigma

    plt.plot(x, np.dot(X_poly, theta[1:]), linewidth=2)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("=== Parte 7 ===")
    print("== Mapeando as caracteristicas com grau 5 ==")
    X, _, _, _, _, _ = importar_dados('/data/ex5data1.mat')
    X_map = poly_features(X, 5)
    print(X_map)
    print("=== Done ===")

    print("== Mapeando as caracteristicas com grau 8 normalizado ==")
    X, y, _, _, Xval, yval = importar_dados('/data/ex5data1.mat', False)
    X_map = poly_features(X, 8)
    X_norm, y_norm, mu, sigma = normalizar_caracteristica(X_map, y)
    print(X_norm)
    print("=== Done ===")

    print("== Gradiente com grau 8 normalizado ==")
    theta = np.zeros(shape=(1, X_norm.shape[1]))
    _lambda = 0
    theta = linear_reg_cost_function.compute(theta, _lambda, X_norm, y_norm)
    print(theta)
    print("=== Done ===")

    print("== Visualizacao da curva da regressao linear ==")
    plot(X_norm, y_norm, theta, mu, sigma)
    print("=== Done ===")

    print("== Visualizacao da curva de aprendizado ==")
    theta = np.ones(1)
    train_error, validation_error, m = lc.learning_curve(theta, X, y, Xval, yval)
    lc.plot(train_error, validation_error, m, 'target/plot7.2.png')
    print("=== Done ===")

