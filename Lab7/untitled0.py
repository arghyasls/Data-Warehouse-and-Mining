# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 07:57:05 2018

@author: Arghya
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm

X=np.array([[1,0],[0,1],[-1,0],[0,-1],[3,1],[6,1],[3,-1],[6,-1]])
Y=[-1,-1,-1,-1,+1,+1,+1,+1]
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,Y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(sharex='col', sharey='row', figsize=(10, 8))



Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr.contourf(xx, yy, Z, alpha=0.1)
axarr.scatter(X[:, 0], X[:, 1], c=Y)
axarr.set_title('Linear SVM')
print('Support Vectors')
print(clf.support_vectors_)
print('Weight Vector')
print(clf.coef_ )
print('Constant Terms')
print(clf.intercept_)
print('Alpha Terms ')
print((clf.dual_coef_))
plt.show()
