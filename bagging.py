import numpy as np
from numpy import genfromtxt
from StringIO import StringIO
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas
from data_tools import *

y_svm = pandas.DataFrame.as_matrix(pandas.read_csv("kaggle_mnist_svm.csv", sep=', ', delimiter=',', header='infer', usecols = [1]).astype(int))
y_log = pandas.DataFrame.as_matrix(pandas.read_csv("kaggle_mnist_logreg.csv", sep=', ', delimiter=',', header='infer', usecols = [1]).astype(int))
y_ran = pandas.DataFrame.as_matrix(pandas.read_csv("kaggle_mnist_randForest.csv", sep=', ', delimiter=',', header='infer', usecols = [1]).astype(int))
y_knn = pandas.DataFrame.as_matrix(pandas.read_csv("kaggle_mnist_knn.csv", sep=', ', delimiter=',', header='infer', usecols = [1]).astype(int))
y_nn = pandas.DataFrame.as_matrix(pandas.read_csv("kaggle_mnist_fullyConnectedNet.csv", sep=', ', delimiter=',', header='infer', usecols = [1]).astype(int))

svm_accuracy = 92
log_accuracy = 91
ran_accuracy = 95
knn_accuracy = 96
nn_accuracy = 97

y_weighted = np.zeros([y_svm.shape[0]])
for i in xrange(y_svm.shape[0]):
	if (y_svm[i] == y_log[i]) & (y_ran[i] == y_knn[i]) & (y_log[i] == y_ran[i]):
		y_weighted[i] = y_svm[i]
	else:
		votes = np.array([[0, 0], [1, 0],[2, 0],[3, 0],[4, 0],[5, 0],[6, 0],[7, 0],[8, 0],[9, 0]]) 
		votes[y_svm[i], 1] += 1 * svm_accuracy
		votes[y_log[i], 1] += 1 * log_accuracy
		votes[y_ran[i], 1] += 1 * ran_accuracy
		votes[y_knn[i], 1] += 1 * knn_accuracy
		votes[y_nn[i], 1] += 1 * nn_accuracy
		y_weighted[i] = np.argmax(votes[:,1]).astype(int)

# Submission:
# svm, knn, svm, rf -> 96.6%
# svm, knn, svm, rf, nn -> 97.4%

saveToCSV(y_weighted.astype(int), "weighted2")