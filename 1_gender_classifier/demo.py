from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Training Data
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Training the models
clf_tree = clf_tree.fit(X, Y)
clf_svm = clf_svm.fit(X, Y)
clf_perceptron = clf_perceptron.fit(X, Y)
clf_KNN = clf_KNN.fit(X, Y)

# Testing using the same data
accuracy_tree = clf_tree.score(X, Y)
accuracy_svm = clf_svm.score(X, Y)
accuracy_perceptron = clf_perceptron.score(X, Y)
accuracy_KNN = clf_KNN.score(X, Y)

print('Accuracy for Decision tree : {}'.format(accuracy_tree))
print('Accuracy for SVM : {}'.format(accuracy_svm))
print('Accuracy for Perceptron : {}'.format(accuracy_perceptron))
print('Accuracy for KNN : {}'.format(accuracy_KNN))
