# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 01:03:19 2018

@author: Student
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:37:13 2018

@author: Student
"""

import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np 
from sklearn import datasets
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
X = iris.data[:, 2:4]
stscaler = StandardScaler().fit(X)
X = stscaler.transform(X)

db = DBSCAN(eps = 0.25, min_samples = 9).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
eps=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.85,0.9,0.95,1]
min_samples=[4,5,6,7,8,9,10]
for i in eps:
    for j in min_samples:
        db = DBSCAN(eps =i, min_samples = j).fit(X)
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in labels else 0)
        print(str(i)+' '+str(j)+' '+str(n_clusters_))
        print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, db.labels_))
        

