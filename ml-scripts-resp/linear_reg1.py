import warnings
import numpy as np

from Parte1 import plot_ex1data1
from Parte1 import visualizar_reta
from Parte1 import visualizar_J_contour
from Parte1 import visualizar_J_surface

from Parte1.custo_reglin_uni import custo_reglin_uni
from Parte1.gd_reglin_uni import gd_reglin_uni


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("=== Parte 1 ===")

    print("== Visualizacao dos dados ==")
    plot_ex1data1.plot()
    print("=== Done ===")

    print("== GD ==")
    print("> Custo J com params 0")
    data = plot_ex1data1.importarDados(filepath="/data/ex1data1.txt", names=["Population", "Profit"])
    J = custo_reglin_uni(data.X, data.y, np.array([0, 0], ndmin=2).T)
    print(J)
    print("----------------")

    print("> Teste da funcao de custo no GD")
    data = plot_ex1data1.importarDados(filepath="/data/ex1data1.txt", names=["Population", "Profit"])
    custo_final, _ = gd_reglin_uni(data.X, data.y, 0.01, 5000)
    print(custo_final)
    print("----------------")

    print("> Descricao do codigo da listagem")
    print('''
    gd_reglin_uni -> algoritmo da regressao linear pelo Gradiente descendente. 
    4 parametros de entrada:
      x: Reprezenta as características
      y: Reprezenta os valores alvo
      alpha: reprezenta a taxa de aprendizagem, regulando o passo dentro do GD
      epochs: maximo de iteracoes
    2 parametros de saida:
      custo (J) : Custo de theta 
      theta: Valores encontrados que reduzem o custo ao minimo, regulando o angulo da reta
    ''')
    print("----------------")

    print("> Visualizacao da reta reg_linear")
    data = plot_ex1data1.importarDados(filepath="/data/ex1data1.txt", names=["Population", "Profit"])
    custo, theta = gd_reglin_uni(data.X, data.y, 0.01, 5000)
    visualizar_reta.plot(filepath="/data/ex1data1.txt", theta=theta)
    print("=== Done ===")

    print("> Predicao para 35k habitantes")
    predict_a = np.array([1, 3.5]).dot(theta)
    print(predict_a)
    print("=== Done ===")

    print("> Predicao para 70k habitantes")
    predict_a = np.array([1, 7]).dot(theta)
    print(predict_a)
    print("=== Done ===")

    print("== Visualizacao de J(theta) ==")
    J = visualizar_J_contour.plot(data, theta)
    visualizar_J_surface.plot(J)
    print("=== Done ===")


