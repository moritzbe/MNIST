import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D


def plot3d(X,y,title):
	fig = plt.figure(figsize=(6,6))
	fig.suptitle(title, fontsize=20)
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('red')
	ax.set_ylabel('green')
	ax.set_zlabel('blue')
	colors = ['white','black','r','g','b']

	for i in np.unique(y):
		ax.scatter(X[np.where([y==i])[1], 0], X[np.where([y==i])[1], 1], X[np.where([y==i])[1], 2], c=colors[i], marker='o')
	plt.show()

def testplot3d(X,title):
	fig = plt.figure(figsize=(6,6))
	fig.suptitle(title, fontsize=20)
	ax = fig.add_subplot(111, projection='3d')
	# rot/blau vertauscht -> in Rohdaten Rot / Blau vertauschen!
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X, marker='o')
	ax.set_xlabel('red')
	ax.set_ylabel('green')
	ax.set_zlabel('blue')
	plt.show()

def testplot3d_PCA(X, X_Color, title):
	fig = plt.figure(figsize=(6,6))
	fig.suptitle(title, fontsize=20)
	ax = fig.add_subplot(111, projection='3d')
	# rot/blau vertauscht -> in Rohdaten Rot / Blau vertauschen!
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X_Color, marker='o')
	ax.set_xlabel('red')
	ax.set_ylabel('green')
	ax.set_zlabel('blue')
	plt.show()


def plotImage(X, entry, label = None):
	img = X[entry,:].reshape((28, 28))
	plt.title('Scores are {label}'.format(label=label))
	plt.imshow(img, cmap='gray')
	plt.show()
	