import warnings
from scipy.optimize import minimize


from Parte3 import plot_ex2data1
from Parte4.map_feature import map_feature
from Parte4.cost_function_reg import *
from Parte4.plot_decision_boundary import plot


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("=== Parte 4 ===")
    print("== Visualizacao dos dados ==")
    data, X, y = plot_ex2data1.importarDados(insertOnes=False, filepath="/data/ex2data2.txt", names=["T1","T2","Rejeicao"])
    plot(data, 'target/plot4.1.png')
    print("=== Done ===")

    print("== Featuring Map ==")
    print("> X: ")
    print(X)
    X_map = map_feature(X)
    print("> X Featured: ")
    print(X_map)
    print(">Tamanho de X Featured: ")
    print(X_map.shape)
    print("=== Done ===")

    print("== Custo J regularizacao lambda 0 ==")
    theta = np.zeros(shape=(len(y), X_map.shape[1]))
    J = cost_function_reg(theta, 0, X_map, y)
    print(J)
    print("=== Done ===")

    print("== Custo J regularizacao lambda 1  ==")
    J = cost_function_reg(theta, 1, X_map, y)
    print(J)
    print("=== Done ===")

    print("== Custo J regularizacao lambda 100  ==")
    J = cost_function_reg(theta, 100, X_map, y)
    print(J)
    print("=== Done ===")

    print("== Custo J regularizacao lambda -50  ==")
    J = cost_function_reg(theta, -50, X_map, y)
    print(J)
    print("=== Done ===")

    print("== Gradiente Descendente para lambda 1  ==")
    theta = np.zeros(X_map.shape[1])
    _lambda = 1
    opt={'maxiter': 1000}
    theta = minimize(cost_function_reg, theta, args=(_lambda, X_map, y), jac=gd_reglog, options=opt).x
    print("> theta")
    print(theta)
    print("=== Done ===")

    print("== Visualizacao da fronteira de decisao  ==")
    #data, X, y = plot_ex2data1.importarDados(filepath="/data/ex2data2.txt", names=["T1", "T2", "Rejeicao"])
    plot(data, 'target/plot4.2.png', theta)
    print("=== Done ===")


