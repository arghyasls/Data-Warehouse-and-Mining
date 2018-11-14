# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:36:44 2018

@author: Student
"""
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import tree
dataset=pd.read_csv('car_datacat.csv')
x = dataset.iloc[:,0:6].values
y =dataset.iloc[:,6].values
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.25)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=7, min_samples_leaf=5)
clf_gini.fit(x_train, y_train)
y_pred = clf_gini.predict(x_test)
print(confusion_matrix(y_test, y_pred))
target_names=['acc','good','unacc','v-good']
print(classification_report(y_test, y_pred, target_names=target_names))
data_feature_names=dataset.columns.values.tolist()
data_feature_names=data_feature_names[:-1]
print(data_feature_names)
with open("clf_gini.txt", "w") as f:
    f = tree.export_graphviz(clf_gini, feature_names=data_feature_names, filled=True,
                               rounded=True, out_file=f)
