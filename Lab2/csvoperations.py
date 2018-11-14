# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:45:10 2018

@author: Student
"""

import pandas as pd
import numpy as np

df=pd.read_csv('cancerData.csv')

#display first five elements
print(df.head(5))

#replace Male with 0 and female with 1
gender={'Male':0,'Female':1}
df.Gender=[gender[item] for item in df.Gender ]

#replace True with 1 and False with 0
boolean={True:1,False:0}
df.HasCancer=[boolean[check] for check in df.HasCancer]

#filling missing values with average age
m=int(df['Age'].mean())
df.Age.fillna(m,inplace=True)

#replace missingvalue for tumour size with the same tumour size of mean of patients having same age
df['TumourSize']=df.groupby('Age').transform(lambda x: x.fillna(x.mean())).TumourSize

#save csv
df.to_csv('cancerDataNew.csv', encoding='utf-8', index=False)