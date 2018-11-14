# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from imblearn.under_sampling import CondensedNearestNeighbour
from numpy import exp, array, random, dot
from sklearn.cross_validation import train_test_split

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn import datasets
print(__doc__)
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Species']
iris = datasets.load_iris()
X = iris.data[:, 0:4]
y=iris.target
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

#initial statistics
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
#using standard scaling on sepa-length,sepal-width ,petal -length,petal-width and encoding on 
#different species of iris
x =iris.data[:,0:4]
y =iris.target
X_normalized=normalize(x,axis=0)
x_train, x_test, y_train, y_test =train_test_split(X_normalized,y,test_size=0.20)
cnn = CondensedNearestNeighbour(return_indices=True)
X_resampled, y_resampled, idx_resampled = cnn.fit_sample(X_normalized, y)
clf=KNeighborsClassifier(n_neighbors=1)
clf.fit(X_resampled, y_resampled)

y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
target_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
print(classification_report(y_test, y_pred, target_names=target_names))

