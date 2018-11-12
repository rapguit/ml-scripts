import warnings
import numpy as np

from Parte5 import linear_reg_cost_function
from Parte5.plot_ex5data1 import importar_dados
from Parte5.plot_ex5data1 import plot


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("=== Parte 5 ===")
    print("== Visualizacao dos dados de treinamento ==")
    X, y, _, _, _, _ = importar_dados('/data/ex5data1.mat')
    plot(X, y)
    print("=== Done ===")


    print("== Custo J regularizado ==")
    theta = np.ones(shape=(1, X.shape[1]))
    cost = linear_reg_cost_function.cost_function(theta, 0, X, y)
    print(cost)
    print("=== Done ===")

    print("== GD regularizado ==")
    theta = np.ones(shape=(1, X.shape[1]))
    cost = linear_reg_cost_function.gd(theta, 0, X, y)
    print(cost)
    print("=== Done ===")

    print("== Visualizacao da regracao linear para lambda 0 ==")
    _lambda = 0
    theta = linear_reg_cost_function.compute(theta, _lambda, X, y)
    print("> theta")
    print(theta)
    print("> Reta da regracao linear")
    plot(X, y, theta, 'target/plot5.2.png')
    print("=== Done ===")
