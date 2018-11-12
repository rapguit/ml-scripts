import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def importarDados(filepath, names):
    path = os.getcwd() + filepath  
    data = pd.read_csv(path, header=None, names=names)

    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1:].values

    # Incluir o valor de 1 em x, pois theta0 = 1
    X = np.c_[np.ones((X.shape[0], 1)), X]

    data.X = X
    data.y = y
    
    return data


def plot(data = importarDados(filepath="/data/ex1data1.txt", names=["Population","Profit"]), filename = 'target/plot1.1.png'):
    plt.scatter(data.X.T[1], data.y, color='red', marker='x')
    plt.title('Populacao da cidade x Lucro da filial')
    plt.xlabel('Populacao da cidade (10k)')
    plt.ylabel('Lucro (10k)')

    if not os.path.exists(os.path.dirname(filename)):
      os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)
    plt.show()
