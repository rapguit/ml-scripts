import os
import matplotlib.pyplot as plt

from Parte5.linear_reg_cost_function import compute
from Parte5.linear_reg_cost_function import cost_function


def learning_curve(theta, X, y, Xval, yval):
    train_error = []
    validation_error = []
    m = []

    for i in range(1, len(y)):
        X_subset = X[:i,:]
        y_subset = y[:i]

        m.append(len(y_subset))
        theta = compute(theta, 0, X_subset, y_subset)
        cost = cost_function(theta, 0, X_subset, y_subset)
        train_error.append(cost)

        val_cost = cost_function(theta, 0, Xval, yval)
        validation_error.append(val_cost)

    return train_error, validation_error, m


def plot(train_error, validation_error, m, out_img='target/plot6.1.png'):
    plt.plot(m, train_error, label='Treino')
    plt.plot(m, validation_error, label='Validacao')
    plt.legend()
    plt.title('Curva de aprendizado da regressao linear')
    plt.xlabel('Nr de exemplo no treinamento')
    plt.ylabel('Erro')
    plt.grid(True)

    if not os.path.exists(os.path.dirname(out_img)):
        os.makedirs(os.path.dirname(out_img))

    plt.savefig(out_img)
    plt.show()

