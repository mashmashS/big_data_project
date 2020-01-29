#!/usr/bin/env python
# coding: utf-8

# In[2]:


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as shc
import pandas as pd

sd = pd.read_csv('D:\\Datasets\\DtmMinusZeros.csv')
sd=sd.head(25)
print(sd)
x= sd.iloc[1:, [1,2]].values
linked = linkage(x, 'ward')

labelList = range(1, 25)

plt.figure(figsize=(10, 9))
plt.title("comparison between article 1987_1 to article 1987_2")
dendrogram(linked,orientation='top',labels=labelList,distance_sort='ascending',show_leaf_counts=True)
plt.show()


# plt.figure(figsize=(10,7))
# plt.title("sdsfdf")
# dend=shc.dendrogram(shc.linkage(x,method="ward"))

פלט:
            word  1987_1  1987_2
0       abstract       1       1
1           also       4      13
2       although       1       1
3       american       1       3
4         amount       1       1
5         analog       1      16
6        appears       1       1
7   architecture       1       1
8         aspect       2       1
9      available       1       3
10        become       1       1
11       becomes       1       1
12        binary       3       3
13    biological       1       5
14           bit       4       1
15          case       3       4
16       circuit       2       3
17      circuits       2       1
18    collective       1       2
19       complex       1       1
20     computing       1       3
21    conference       1       5
22     connected       3       4
23  connectivity      12       2
24        define       3       1




