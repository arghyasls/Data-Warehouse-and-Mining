# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:57:24 2018

@author: Student
"""

from numpy import exp, array, random, dot
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn import datasets
#downloading of dataset and setting headers

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Species']
iris = datasets.load_iris()
X = iris.data[:, 2:4]
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

#train test split 80-20
x_train, x_test, y_train, y_test =train_test_split(X_normalized,y,test_size=0.20)
clf=KNeighborsClassifier(n_neighbors=7,weights='distance')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
target_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
print(classification_report(y_test, y_pred, target_names=target_names))
myList = list(range(1,31))

# subsetting just the odd ones
neighbours=myList
# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbours:
    knn = KNeighborsClassifier(n_neighbors=k,weights='distance')
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
mse=[]
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbours[mse.index(min(mse))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbours,mse)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()