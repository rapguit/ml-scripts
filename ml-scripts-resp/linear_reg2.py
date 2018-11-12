import warnings
import numpy as np

from Parte2.plot_ex1data2 import importarDados
from Parte2.normalizacao import normalizar_caracteristica
from Parte2.custo_reglin_multi import custo_reglin_multi
from Parte2.gd_reglin_multi import gd


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("=== Parte 2 ===")
    X, y = importarDados('/data/ex1data2.txt')

    print("== Visualizacao dos dados normalizados ==")
    norm_X, norm_y, _, _ = normalizar_caracteristica(X, y)
    print('> X:')
    print(norm_X)
    print('> y:')
    print(norm_y)
    print("=== Done ===")

    print("== Normalizacao das caracteristicas ==")
    print('''
    > O metodo normalizar_caracteristica possui 2 params de entrada:
       X: Matriz a ser normalizada
       y: Alvo, valor final de interesse
       
       Alem disso, as entradas funcionam com conjuntos de dados de variados tamanhos, 
       pois o python trabalha com conceito de vetorizacao. Isso permite que os parametros
       de entrada assumam valores escalares, vetores ou matrizes (arrays), assim como o seu
       retorno. 
    ''')
    print("=== Done ===")

    print("== GD com mais caracteristicas ==")
    print('''
    > O metodo GD com 2 mais características:
       As entradas funcionam com conjuntos de dados de variados tamanhos, assim como na 
       normalizacao por conta do suporte a vetorizacao. Isso permite que o algoritmo do
       GD para uma ou mais caracteristicas funcionem sob a mesma implementacao.
    ''')
    print("> Custo J (com normalizacao)")
    J = custo_reglin_multi(norm_X, norm_y, np.array([0,0,0], ndmin = 2).T)
    print(J)
    print("=== Done ===")

    print("> GD (com normalizacao)")
    custo_final, theta = gd(norm_X, norm_y, 0.01, 5000)
    print("> Custo")
    print(custo_final)
    print("> theta")
    print(theta)
    print("=== Done ===")
