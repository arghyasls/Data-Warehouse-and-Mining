# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:03:54 2018

@author: Student
"""

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd

dataset=pd.read_csv('car_datacat.csv')
x = dataset.iloc[:,0:6].values
y =dataset.iloc[:,6].values
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2)
cnn = CondensedNearestNeighbour(return_indices=True)
X_resampled, y_resampled, idx_resampled = cnn.fit_sample(x, y)
clf=KNeighborsClassifier(n_neighbors=1)
clf.fit(X_resampled, y_resampled)
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))