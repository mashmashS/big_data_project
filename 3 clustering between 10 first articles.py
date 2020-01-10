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


plt.scatter(x[:,0],x[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black',marker='x')

# In[ ]:




