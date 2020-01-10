#!/usr/bin/env python
# coding: utf-8

# In[28]:


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


#plt.scatter(x[:,0],x[:,9],c=kmeans.labels_,cmap='rainbow')
#plt.scatter(kmeans.cluster_centers_[:,1] ,kmeans.cluster_centers_[:,1],color='black',marker='x')

display_factorial_planes(x, 2, pca, [(0,1)], illustrative_var = clusters, alpha = 0.8)
plt.scatter(cluster_centers_[:, 0], cluster_centers_[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)


# In[ ]:




