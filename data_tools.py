import numpy as np
from numpy import genfromtxt
from StringIO import StringIO
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas


# This function loads the data into X and y, 
# outputs the feature names, the label dictionary, m and n
def loaddata(filename):
	data = genfromtxt(filename, delimiter=',', dtype=None)
	X = data[1:,:].astype(float)
	# y = data[1:,0].astype(int)	
	np.save("X_TEST", X, allow_pickle=True, fix_imports=True)
	# np.save("Y_TRAIN", y.T, allow_pickle=True, fix_imports=True)

def loadnumpy(filename):
	array = np.load(filename)
	return array

def addOffset(X):
	X = np.c_[np.ones((X.shape[0],1)), X]
	return X


def saveToCSV(y, alg_name="alg"):
	ImageId = []
	for i in range(1, y.shape[0]+1):
		ImageId.append(i)

	submission = pandas.DataFrame({
        "ImageId": ImageId,
        "Label": y
    })
	filename = "kaggle_mnist_" + alg_name + ".csv"
	submission.to_csv(filename, index=False)



def setOtherLabelsZero(m, value):
	a = np.array([m.shape])
	a[m == value] = 1
	return a

def plotConfusionMatrix(y, pred):
	n_labels = len(np.unique(y))
	matrix = np.zeros([n_labels+1,n_labels+1])
	matrix[0,1:] = np.unique(y)
	matrix[1:,0] = np.unique(y)
	matrix[1:,1:] = confusion_matrix(y.astype(int), pred.astype(int))
	print "The confusion matrix (Truth X Prediction):"
	print matrix
	print "---------------------------------------------------\t"

def accuracyKmeans(true, pred):
	error = 0
	for i in xrange(len(true)):
		if true[i] != pred[i]:
			error = error + 1
	return error


