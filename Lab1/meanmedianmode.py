# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:04:44 2018

@author: Student
"""
def mean(l):
    s=0
    count=0
    for i in l:
        s+=i
        count+=1
    return (s/count)
def mode(l):
    count={}
    mode=[]
    for i in l:
        count[i]=count.get(i,0)+1
    maxcount=0
    
    for k,v in count.items():
        if maxcount<=v:
            maxcount=v
           
    for k,v in count.items():
        if(maxcount==v):
            mode.append(k)
        
    return mode
def median(l):
    l.sort()
    if len(l)%2==0:
        return (l[int(len(l)/2)]+l[int((len(l)/2)-1)])/2
    else:
        return (l[int((len(l))/2)])

print('Enter size of list')
n=input()
n=int(n)
l=[]
print('Enter elements of list')
for i in range(0,n):
    v=int(input())
    l.append(v)
print('Original list')
for i in l:
    print(i,sep=' ',end='\n')
print('Mean='+str(mean(l)))
print('Mode='+str(mode(l)))
print('Median='+str(median(l)))