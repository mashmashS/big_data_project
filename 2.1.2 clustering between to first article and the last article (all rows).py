#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.cluster import KMeans

store_data = pd.read_csv('D:\\Datasets\\NIPS_1987-2015.csv')

x = store_data.iloc[:, [1,5811]].values
print(x)

kmeans = KMeans(n_clusters=9)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)
print(kmeans.cluster_centers_)



plt.scatter(x[:,0],x[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1],color='black')


[[0 0]
 [0 0]
 [0 0]
 ...
 [0 0]
 [0 0]
 [0 0]]
[0 0 0 ... 0 0 0]
[[-6.02295991e-15 -3.57769370e-14]
 [ 1.03125000e+00  1.36250000e+01]
 [ 3.86138614e-01  5.38613861e+00]
 [ 7.50000000e-01  2.91250000e+01]
 [ 1.96666667e+01  0.00000000e+00]
 [ 7.15789474e-02  1.47578947e+00]
 [ 4.10204082e+00  9.18367347e-01]
 [ 9.91666667e+00  3.83333333e+00]
 [ 1.16996047e+00  1.77865613e-01]]
<matplotlib.collections.PathCollection at 0x2561e93c848>
*Graph at the main file project
