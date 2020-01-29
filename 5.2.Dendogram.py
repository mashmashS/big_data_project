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


# In[ ]:




