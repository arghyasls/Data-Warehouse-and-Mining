# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:54:35 2018

@author: Student
"""


import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd

dataset=pd.read_csv('car_datacat.csv')
x = dataset.iloc[:,0:6].values
y =dataset.iloc[:,6].values
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2)

clf=KNeighborsClassifier(n_neighbors=10)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
target_names=['acc','good','unacc','v-good']
print(classification_report(y_test, y_pred, target_names=target_names))
myList = list(range(1,31))

# subsetting just the odd ones
neighbours=myList
# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbours:
    knn = KNeighborsClassifier(n_neighbors=k)
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