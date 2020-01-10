#!/usr/bin/env python
# coding: utf-8

# In[3]:


import seaborn as sns; sns.set()
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


store_data = pd.read_csv('C:\\Datasets\\firstvslast.csv')
sd=store_data.head(100)

x = sd.iloc[:, [1,2]].values
print(x)


kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)
kmeans3.cluster_centers_

plt.scatter(x[:,0],x[:,1],c=y_kmeans3,cmap='rainbow')


# In[ ]:




