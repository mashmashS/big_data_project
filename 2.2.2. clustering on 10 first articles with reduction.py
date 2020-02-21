#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv('C:\\Datasets\\NIPS_1987-2015.csv')

# run a KMeans model with 3 clusters.
clustering_kmeans = KMeans(n_clusters=3, precompute_distances="auto", n_jobs=-1)
data['clusters'] = clustering_kmeans.fit_predict(data)


# run PCA to reduce the dimensionality to 2 dimensions
reduced_data = PCA(n_components=2).fit_transform(data)

# create a new dataframe that contains the 2 dimensions and the cluster label
results = pd.DataFrame(reduced_data,columns=['x','y'])

# plot the results with a scatterplot
plt.figure(figsize=(10, 7))
sns.scatterplot(x="x",y="y",hue=data['clusters'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()


output:
*Graph is in the main project



