# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 01:33:28 2018

@author: Arghya
"""
import math
import matplotlib.pyplot as plt
import numpy as np
T=[('E1',8123,83,424,1370),('E2',8330,2,622,1046),('E3',3954,3080,5,2961),
   ('E4',2886,1363,1320,4431),('E5',1500,2000,500,6000),
   ('E6',4000,2000,1000,3000),('E7',9481,298,127,94),
   ('E8',4000,2000,2000,2000),('E9',7450,2483,4,63),
   ('E10',61,2483,4,7452)]
Tnew=[]
for item in T:
        a,b,c,d,e = item
        fplus1=b+d
        fplus0=c+e
        f1plus=b+c
        f0plus=d+e
        n=b+c+d+e
        Tnew.append((a,b,c,d,e,fplus1,fplus0,f1plus,f0plus,n))
#print(Tnew)
oddratio=[]
correlation=[]
kappa=[]
interest=[]
cosine=[]
piatesky_Shapiro=[]
collective_strength=[]
jaccard=[]
all_confidence=[]
av=[]
f=[]
V=[]
L=[]
G=[]
J=[]
M=[]
Lambda=[]
for item in Tnew:
    a,f11,f10,f01,f00,fplus1,fplus0,f1plus,f0plus,n = item
    correlation.append((a,((n*f11-f1plus*fplus1)/(math.sqrt(fplus1*fplus0*f1plus*f0plus)))))
    oddratio.append((a,(f11*f00)/(f01*f10)))
    kappa.append((a,(n*f11+n*f00-f1plus*fplus1-f0plus*fplus0)/(n*n-f1plus*fplus1-f0plus*fplus0)))
    interest.append((a,(n*f11)/(f1plus*fplus1)))
    cosine.append((a,f11/(math.sqrt(f1plus*fplus1))))
    piatesky_Shapiro.append((a,(f11/n)-((f1plus*fplus1)/n*n)))
    collective_strength.append((a,((f11+f00)/(f1plus+fplus1+f0plus+fplus0))*((n-f1plus-fplus1-f0plus*fplus0)/(n-f11-f00))))
    jaccard.append((a,f11/(f1plus+fplus1-f11)))
    all_confidence.append((a,min((f11/f1plus),(f11/fplus1))))
    av.append((a,(f11/f1plus)-(fplus1/n)))
    f.append((a,((f11/f1plus)-(fplus1/n))/(1-(fplus1/n))))
    V.append((a,(f1plus*fplus0)/(n*f10)))
    L.append((a,(f11+1)/(f1plus+2)))
    G.append((a,(f1plus/n)*(pow((f11/f1plus),2)+pow((f10/f1plus),2))
    -pow((fplus1/n),2)+(f0plus/n)*(pow((f01/f0plus),2)+pow((f00/f0plus),2))
    -pow((fplus0/n),2)))
    J.append((a,(f11/n)*math.log((n*f11)/(f1plus*fplus1))+(f10/n)*math.log((n*f10)/(f1plus*fplus0))))
    mf00=(f00/n)*math.log((n*f00)/(f0plus*fplus0))
    mf01=(f01/n)*math.log((n*f01)/(f0plus*fplus1))
    mf10=(f10/n)*math.log((n*f10)/(f1plus*fplus0))
    mf11=(f11/n)*math.log((n*f11)/(f1plus*fplus1))
    df00=(f0plus/n)*math.log(f0plus/n)
    df01=(f0plus/n)*math.log(f0plus/n)
    df10=(f1plus/n)*math.log(f1plus/n)
    df11=(f1plus/n)*math.log(f1plus/n)
    Lambda.append((a,(max(f01,f00)+max(f10,f11)-max(fplus0,fplus1))/(n-max(fplus0,fplus1))))
    M.append((a,sum([mf00,mf01,mf10,mf11])/-sum([df00,df01,df10,df11])))
    
sorted_by_oddratio = sorted(oddratio, key=lambda tup:tup[1],reverse=True )
sorted_by_correlation = sorted(correlation, key=lambda tup:tup[1],reverse=True )
sorted_by_kappa = sorted(kappa, key=lambda tup:tup[1],reverse=True )
sorted_by_interest=sorted(interest,key=lambda tup:tup[1],reverse=True)
sorted_by_cosine= sorted(cosine  , key=lambda tup:tup[1],reverse=True )
sorted_by_piatesky_Shapiro  = sorted(piatesky_Shapiro , key=lambda tup:tup[1],reverse=True )
sorted_by_collective_strength  = sorted(collective_strength , key=lambda tup:tup[1],reverse=True )
sorted_by_jaccard = sorted( jaccard, key=lambda tup:tup[1],reverse=True )
sorted_by_all_confidence  = sorted(all_confidence , key=lambda tup:tup[1],reverse=True )
sorted_by_L=sorted(L,key=lambda tup:tup[1],reverse=True )
sorted_by_V= sorted(V,key=lambda tup:tup[1],reverse=True )
sorted_by_f= sorted(f,key=lambda tup:tup[1],reverse=True )
sorted_by_av= sorted(av,key=lambda tup:tup[1],reverse=True )
sorted_by_G= sorted(G,key=lambda tup:tup[1],reverse=True )
sorted_by_M= sorted(M,key=lambda tup:tup[1],reverse=True )
sorted_by_J= sorted(J,key=lambda tup:tup[1],reverse=True )
sorted_by_Lambda= sorted(Lambda,key=lambda tup:tup[1],reverse=True )
print('Symmetric measures')

print('Correlation')
print(sorted_by_correlation)
print('Odds ratio')
print(sorted_by_oddratio)
print('Kappa')
print(sorted_by_kappa)
print('Interest')
print(sorted_by_interest)
print('Cosine')
print(sorted_by_cosine)
print('Piatetsky-Shapiro')
print(sorted_by_piatesky_Shapiro)
print('Collective strength')
print(sorted_by_collective_strength)
print ('Jaccard')
print(sorted_by_jaccard)
print('All confidence')
print(sorted_by_all_confidence)

print('ASymmetric measures')
print('Goodman-Kruskal')
print(sorted_by_Lambda)
print('Mutual Information')
print(sorted_by_M)
print('J-Measure')
print(sorted_by_J)
print('Gini Index')
print(sorted_by_G)
print('Laplace')
print(sorted_by_L)
print('Conviction')
print(sorted_by_V)
print('Certainty Factor')
print(sorted_by_f)
print('Added Value')
print(sorted_by_av)

y=[e[1] for e in correlation]
x=[i for i in range(1,11)]
plt.scatter(x,y)
plt.title('Scatter plot Correlation')
plt.xlabel('Event Order')
plt.ylabel('Correlation Value')
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.plot(x,y)



    