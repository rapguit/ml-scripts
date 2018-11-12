import warnings
import numpy as np

from Parte5.plot_ex5data1 import importar_dados
from Parte6.learning_curve import learning_curve
from Parte6.learning_curve import plot


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("=== Parte 6 ===")
    print("== Visualizacao das curvas de aprendizado ==")
    X, y, _, _, Xval, yval = importar_dados('/data/ex5data1.mat')
    theta = np.ones(2)
    train_error, validation_error, m = learning_curve(theta, X, y, Xval, yval)
    plot(train_error, validation_error, m)
    print("=== Done ===")



