#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.cluster import KMeans

store_data = pd.read_csv('D:\\Datasets\\NIPS_1987-2015.csv')
x = store_data.iloc[:, 1:10].values
print(x)

kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)
print(kmeans.cluster_centers_)


plt.scatter(x[:,0],x[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black',marker='x')

output:
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
[0 0 0 ... 0 0 0]
[[ 0.0423744   0.10379045  0.05042017  0.07831218  0.09395673  0.05345968
   0.09243697  0.06096907  0.07777579]
 [ 0.88679245  3.10943396  1.80754717  2.01509434  4.03773585  1.13584906
   3.37358491  1.6490566   3.54716981]
 [ 6.5        15.16666667 10.66666667 14.33333333 12.83333333  7.5
   1.         23.75        3.5       ]]
<matplotlib.collections.PathCollection at 0x21010fd0dc8>
*Graph is in the main project



