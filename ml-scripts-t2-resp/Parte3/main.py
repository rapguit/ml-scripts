import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib.mlab as mlab
from scipy.io import loadmat  
from scipy import stats

from Parte2.main import save_img

def estimate_gaussian_params(X):
	mu = np.mean(X, axis=0).T
	sigma2 = np.var(X, axis=0).T

	return (mu, sigma2)


def select_epsilon(pval, yval):  
	best_epsilon_value = 0
	best_f1_value = 0

	step_size = (pval.max() - pval.min()) / 1000

	print('step size: ' + str(step_size))

	for epsilon in np.arange(pval.min(), pval.max(), step_size):
		preds = pval < epsilon
		f1 = calculate_f1(preds, yval)

		if f1 > best_f1_value:
			best_f1_value = f1
			best_epsilon_value = epsilon

	return best_epsilon_value, best_f1_value


def calculate_f1(preds, yval):
	# checando todos os 1 que realmente deram 1 (positivos)
	tp = np.sum(np.logical_and((preds == 1), (yval == 1)))
	# checando todos os 1 quando deveriam ser 0 (falso positivos)
	fp = np.sum(np.logical_and((preds == 1), (yval == 0)))
	# checando todos os 0 quando deveriam ser 1 (falso negativos)
	fn = np.sum(np.logical_and((preds == 0), (yval == 1)))

	prec = tp / (tp + fp)
	rec = tp / (tp + fn)

	return 2 * prec * rec / (prec + rec)


def main():
	data = loadmat('../data/ex8data1.mat')
	X = data['X']  

	(mu, sigma2) = estimate_gaussian_params(X)
	print('mu: ' + str(mu))
	print('variance: ' + str(sigma2))

	# Plot dataset
	plt.scatter(X[:,0], X[:,1], marker='x')  
	plt.axis('equal')
	save_img(plt, '../target/plot3_1.png')
	plt.show()

	# Plot dataset and contour lines
	plt.scatter(X[:,0], X[:,1], marker='x')  
	x = np.arange(0, 25, .025)
	y = np.arange(0, 25, .025)
	first_axis, second_axis = np.meshgrid(x, y)
	Z = mlab.bivariate_normal(first_axis, second_axis, np.sqrt(sigma2[0]), np.sqrt(sigma2[1]), mu[0], mu[1])
	plt.contour(first_axis, second_axis, Z, 10, cmap=plt.cm.jet)
	plt.axis('equal')
	save_img(plt, '../target/plot3_2.png')
	plt.show()

	# Load validation dataset
	Xval = data['Xval']  
	yval = data['yval'].flatten()

	stddev = np.sqrt(sigma2)

	pval = np.zeros((Xval.shape[0], Xval.shape[1]))  
	pval[:,0] = stats.norm.pdf(Xval[:,0], mu[0], stddev[0])  
	pval[:,1] = stats.norm.pdf(Xval[:,1], mu[1], stddev[1])  
	print(np.prod(pval, axis=1).shape)
	epsilon, _ = select_epsilon(np.prod(pval, axis=1), yval)  
	print('Best value found for epsilon: ' + str(epsilon))

	# Computando a densidade de probabilidade 
	# de cada um dos valores do dataset em 
	# relacao a distribuicao gaussiana
	p = np.zeros((X.shape[0], X.shape[1]))  
	p[:,0] = stats.norm.pdf(X[:,0], mu[0], stddev[0])  
	p[:,1] = stats.norm.pdf(X[:,1], mu[1], stddev[1])

	# Apply model to detect abnormal examples in X
	anomalies = np.where(np.prod(p, axis=1) < epsilon)

	# Plot the dataset X again, this time highlighting the abnormal examples.
	plt.clf()
	plt.scatter(X[:,0], X[:,1], marker='x')  
	plt.scatter(X[anomalies[0],0], X[anomalies[0],1], s=50, color='r', marker='x')  
	plt.axis('equal')
	save_img(plt, '../target/plot3_3.png')
	plt.show()

if __name__ == "__main__":
	main()