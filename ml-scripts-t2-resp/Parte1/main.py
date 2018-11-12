import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.io
import matplotlib.pyplot as plt

from Parte2.main import save_img


def find_closest_centroids(X, centroids):
    K = np.size(centroids, 0)
    idx = np.zeros((len(X), 1), dtype=np.int8)

    for i in range(0, len(X)):
        for j in range(0, K):
            dif = X[i] - centroids[j]
            distance = euclidean_distance(dif)

            if j == 0 or distance < min_distance:
                min_distance = distance
                min_centroid = j

        idx[i] = min_centroid

    return idx


def euclidean_distance(mtx):
    return np.sqrt(np.square(mtx).sum())


def compute_centroids(X, idx, K):
    centroids = np.zeros((K,np.size(X,1)))

    for i in range(0, K):
        #extrai os indices para todos idx = K
        idx_of_K = np.where(np.in1d(idx, i))
        #computa novo centroide pela media dos x's pertencentes ao K
        centroids[i] = np.mean(X[idx_of_K])

    return centroids

def kmeans_init_centroids(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]

def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    previous_centroids = centroids

    for iter in range(max_iters):
        # Assignment of examples do centroids
        idx = find_closest_centroids(X, centroids)

        # PLot the evolution in centroids through the iterations
        if plot_progress:
            plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
            plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
            plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
            plt.plot(previous_centroids[:,0], previous_centroids[:,1], 'yo')
            plt.plot(centroids[:,0], centroids[:,1], 'bo')
            save_img(plt, '../target/plot1_2_1.png')
            plt.show()

        previous_centroids = centroids

        # Compute new centroids
        centroids = compute_centroids(X, idx, K)

    return (centroids, idx)

def main():
    # Find closest centroids
    raw_mat = scipy.io.loadmat("../data/ex7data2.mat")
    X = raw_mat.get("X")

    K = 3

    # Fixed seeds (i.e., initial centroids)
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_closest_centroids(X, initial_centroids)

    # Plot initial assignments.
    plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
    plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
    plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
    plt.title('Initial assignments')
    save_img(plt, '../target/plot1_1.png')
    plt.show()

    print('Cluster assignments for the first, second and third examples: ' + str(idx[0:3].flatten()))

    # Compute initial means
    centroids = compute_centroids(X, idx, K)

    # Now run 10 iterations of K-means on fixed seeds
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    #initial_centroids = kmeans_init_centroids(X, K)
    max_iters = 10
    centroids, idx = run_kmeans(X, initial_centroids, max_iters, plot_progress=True)
    print('Centroids after the 1st update:\n' + str(centroids))

    # Plot final clustering.
    plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
    plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
    plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
    plt.title('Final clustering')
    save_img(plt, '../target/plot1_2.png')
    plt.show()

    #Iniciando o centroides de forma aleatoria.

    initial_centroids = kmeans_init_centroids(X, K)
    max_iters = 10
    centroids, idx = run_kmeans(X, initial_centroids, max_iters, plot_progress=False)

    # Plot final clustering.
    plt.scatter(X[np.where(idx == 0), 0], X[np.where(idx == 0), 1], marker='x')
    plt.scatter(X[np.where(idx == 1), 0], X[np.where(idx == 1), 1], marker='x')
    plt.scatter(X[np.where(idx == 2), 0], X[np.where(idx == 2), 1], marker='x')
    plt.title('Aleatory Init Clustering')
    save_img(plt, '../target/plot1_3.png')
    plt.show()


if __name__ == "__main__":
    main()
