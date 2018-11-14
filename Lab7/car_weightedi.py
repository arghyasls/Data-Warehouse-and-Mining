# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:49:10 2018

@author: Student
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:13:55 2018

@author: Student
"""

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
dataset=pd.read_csv('car_datacat.csv')
dataset2=pd.read_csv('car_data.csv')
x = dataset.iloc[:,0:6].values
y =dataset.iloc[:,6].values
dataset2.iloc[:,6].value_counts().plot(kind='bar')

np.random.seed(42)
indices = np.random.permutation(len(dataset))
n_training_samples = 150
learnset_data = x[indices[:-n_training_samples]]
learnset_labels = y[indices[:-n_training_samples]]
testset_data = x[indices[-n_training_samples:]]
testset_labels = y[indices[-n_training_samples:]]



def distance(instance1, instance2):
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1) 
    instance2 = np.array(instance2)
    
    return np.linalg.norm(instance1 - instance2)


def get_neighbors(training_set, 
                  labels, 
                  test_instance, 
                  k, 
                  distance=distance):

    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors
y_pred=[]
def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    
    return class_counter.most_common(1)[0][0]
def vote_distance_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    distk=neighbors[number_of_neighbors-1][1]
    dist1=neighbors[0][1]
    for index in range(number_of_neighbors):
        dist = neighbors[index][1]
        label = neighbors[index][2]
        #class_counter[label] += 1 / (dist**2 + 1)
        if(dist!=dist1):
            class_counter[label]+=(distk-dist)/(distk-dist1)
        else:
            class_counter[label]+=1
            
    labels, votes = zip(*class_counter.most_common())
    print(labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)
check=[]
for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data, 
                              learnset_labels, 
                              testset_data[i], 
                              10, 
                              distance=distance)
    '''print("index: ", i, 
          ", result of vote: ", vote(neighbors), 
          ", label: ", testset_labels[i], 
          ", data: ", testset_data[i])'''
    ans,z=vote_distance_weights(neighbors)
    check.append(vote_distance_weights(neighbors))
    y_pred.append(ans)
print(confusion_matrix(testset_labels,y_pred))
target_names=['acc','good','unacc','v-good']
print(classification_report(testset_labels, y_pred, target_names=target_names))

