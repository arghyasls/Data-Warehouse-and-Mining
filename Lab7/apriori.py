# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 01:07:03 2018

@author: Arghya
"""

from itertools import combinations
items = [['A', 'C', 'D'], ['B', 'C', 'E'], ['A', 'B', 'C', 'E'], ['B', 'E']]
s = []
def apriori(lname, paramater):
    second = list(combinations(lname, paramater))
    print(second)

    slist = []
    for l in items:
        st = list(combinations(l, paramater))
        slist.append(st)
    print(slist)
    qlist = []
    for p in slist:
        for q in p:
            qlist.append(q)
    print(qlist)
    scounts = {}
    for i in qlist:
        if i in second:
            if i in scounts:
                scounts[i] = scounts[i] + 1
            else:
                scounts[i] = 1
    print(scounts)

    mv = 9999
    for k in scounts:
        if scounts[k] < mv:
            mv = scounts[k]
    for k in scounts.copy():
        if scounts[k] == mv:
            scounts.pop(k)
    print(scounts)

    return scounts

#first
for i in items:
    for j in i:
        s.append(j)
counts = {}
for i in s:
    if i in counts:
        counts[i] = counts[i] + 1
    else:
        counts[i] = 1
result = sorted(counts.items(), key=lambda t: t[0], reverse=False)
new_dict = {}
for k, v in result:
    new_dict[k] = v
m = 9999
for k in new_dict:
    if new_dict[k] < m:
        m = new_dict[k]
for k in new_dict.copy():
    if new_dict[k]==m:
        new_dict.pop(k)
print(new_dict)


# second
nitem = []
for k in new_dict:
    nitem.append(k)

support2=apriori(nitem,2)
print(support2)
# third
support3 = apriori(nitem,3)
print(support3)