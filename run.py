from data_tools import *
from algorithms import *
from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFE

# Loading data for fast use in Python:
# loaddata("test.csv")
# loaddata("train.csv")

# Loading the data into X, y
X = loadnumpy("X_TRAIN.npy")/255.
X_test = loadnumpy("X_TEST.npy")/255.
y = loadnumpy("Y_TRAIN.npy")


m = X.shape[0]
m_test = X_test.shape[0]
n = X.shape[1]

print "The number of training samples m is", m
print "The number of test samples m is", m_test
print "The number of features is", n

#Cross Validation
cv = 7


# # Logistic Regression (Ridge)
# # achieved Train Accuracy of 91% with reg = .3
# # accuracy largely equal for reg .1 - 20
# reg = [.3]
# accuracy = 0
# X_off = addOffset(X)
# for i in reg:
# 	log_reg = logRegress(X_off, y, i)
# 	train_accuracy = np.mean(cross_val_score(log_reg, X_off, y, cv=cv))
# 	print "The train accuracy of SVM is", train_accuracy
# 	print "c2 is", i
# 	if train_accuracy >= accuracy:
# 		accuracy = np.mean(cross_val_score(log_reg, X_off, y, cv=cv))
# 		best_reg = i
# 		best_model = log_reg
# 		print "new best!"

# y_pred = best_model.predict(addOffset(X_test))
	



# SVM
# achieved Train Accuracy of 91% with c2 = .1 - 4
# best c2 = .1
# adding bias/offset
# X_off = addOffset(X)

# c2 = [0.1]
# accuracy = 0
# for i in c2:
# 	svm = supvecm(X_off,y,i)
# 	train_accuracy = np.mean(cross_val_score(svm, X_off, y, cv=cv))
# 	print "The train accuracy of SVM is", train_accuracy
# 	print "c2 is", i
# 	if train_accuracy >= accuracy:
# 		accuracy = np.mean(cross_val_score(svm, X_off, y, cv=cv))
# 		best_c2 = i
# 		best_model = svm
# 		print "new best!"
# y_pred = best_model.predict(addOffset(X_test))

# k_nn
# train accuracy with n=11 is 0.96
# neighbors = 11
# k_nn = k_nn(X, y, neighbors)
# print "The train accuracy of K-NN is", np.mean(cross_val_score(k_nn, X, y, cv=cv))
# plotConfusionMatrix(y, k_nn.predict(X))
# y_pred = k_nn.predict(X_test)

# Random Forest:
# Number of Trees K: K = 30 -> 95%, K = 100 -> 96%, K = 
# K = 2000
# rf = randForest(X, y, K)
# print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X, y, cv=cv))
# y_pred = rf.predict(X_test)

# Fully Connected Net:
# After 100 epoqs achieved accuracy of 99.98% on training, Test_accuracy is 97% - Overfit
y = y.T 
model = fullyConnectedNet(X, y, epochs = 100)
# fix random seed for reproducibility
# seed = 7
# evaluate using 10-fold cross validation
# kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
# print "The train accuracy of the Neural Net is", np.mean(cross_val_score(model, X, y, cv=kfold))
results = model.predict(X_test)
y_pred = np.zeros([results.shape[0]])
for i in xrange(results.shape[0]):
	y_pred[i] = np.argmax(results[i,:]).astype(int)

# Submission:
saveToCSV(y_pred.astype(int), "fullyConnectedNet")
