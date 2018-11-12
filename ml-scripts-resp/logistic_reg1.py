import warnings
import numpy as np
import scipy.optimize as opt

from Parte3 import plot_ex2data1
from Parte3 import predizer_aprovacao
from Parte3.sigmoide import sigmoide
from Parte3.custo_reglog import custo_reglog
from Parte3.gd_reglog import gd_reglog

from Parte2.normalizacao import normalizar_caracteristica


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("=== Parte 3 ===")
    print("== Visualizacao dos dados ==")
    data, _, _ = plot_ex2data1.importarDados()
    plot_ex2data1.plot(data)
    print("=== Done ===")

    print("== Sigmoide ==")
    print("> sig 0")
    s = sigmoide(0)
    print(s)
    print("> sig pos")
    s = sigmoide(1000000)
    print(s)
    print("> sig neg")
    s = sigmoide(-9999999999)
    print(s)
    print("> sig array")
    s = sigmoide(np.array([11,2222222,-3,444,-55]))
    print(s)
    print("=== Done ===")

    print("== Custo J ==")
    _, X, y = plot_ex2data1.importarDados()
    theta = np.array([0, 0, 0], ndmin=2)
    J = custo_reglog(theta, X, y)
    print(J)
    print("=== Done ===")

    print("== Custo J normalizado ==")
    _, X, y = plot_ex2data1.importarDados(False)
    norm_X, norm_y = normalizar_caracteristica(X, y)
    theta = np.array([0, 0, 0], ndmin=2)
    J = custo_reglog(theta, norm_X, norm_y)
    print(J)
    print("=== Done ===")

    print("== GD ==")
    gd = gd_reglog(theta, norm_X, norm_y)
    print(gd)
    print("=== Done ===")

    print("== GD com a funcao de custo ==")
    _, X, y = plot_ex2data1.importarDados()
    result = opt.fmin_tnc(func = custo_reglog, x0 = theta, fprime = gd_reglog, args = (X, y))
    gd = custo_reglog(result[0], X, y)
    print(result)
    print(gd)
    print("=== Done ===")

    print("== Avaliando o modelo ==")
    print("> Predicao")
    theta_min = np.matrix(result[0])
    x1 = np.array([[1.0, 45.0, 85.0]])
    p = predizer_aprovacao.predizer(theta_min, x1)
    print(p)
    print("> Acuracia")
    acc = predizer_aprovacao.acuracia(X, y, result)
    print('{0}%'.format(acc))
    print("> Probabolidade")
    probabilidade = sigmoide(x1 * theta_min.T)
    print(probabilidade[0, 0])
    print("=== Done ===")
