# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:37:13 2018

@author: Student
"""

import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np 
from sklearn import datasets
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(10, 7))  

iris = datasets.load_iris()
X = iris.data[:, 0:2]
y=iris.target
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

#initial statistics
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
Z = linkage(X, 'single')
print(Z[:20])
plt.title("Complete Linkage Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete'))  
plt.show()
plt.title("Single Linkage Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='single'))  
plt.show()
'''from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')  
cluster.fit_predict(X)  
ans=cluster.labels_
print(cluster.labels_)
plt.figure(figsize=(10, 7))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap=plt.cm.Set1,
            edgecolor='k')
plt.title("Clustered Result")  
plt.xlabel('Petal length')
plt.ylabel('Petal width')'''
