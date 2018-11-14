# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:01:55 2018

@author: Student
"""
#normalize between 0 and 1
import numpy as np
from scipy import stats
import pandas as pd
import statistics as s
def minmax(data):
    return [round((item-min(data))/(max(data)-min(data)),5) for item in data]
def zscore(data):
    mean = np.mean(data)
    std=np.std(data)
    b=(data-mean)/std
    return b
  
df=pd.read_csv('cancerData.csv')
data = df['TumourSize'].fillna(df['TumourSize'].mean()).tolist()#[100,58,1,100000,222]

def decimal(data):
  return [item/pow(10,len(str(max(data)))) for item in data]

print(minmax(data))
print(zscore(data))
print(decimal(data))