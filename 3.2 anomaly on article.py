#!/usr/bin/env python
# coding: utf-8

# In[48]:



import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


#sum all of the columns
sd = pd.read_csv('D:\\Datasets\\NIPS_1987-2015.csv')
#sd=data.head(400)
# 1987_2 הצגת מילה ונתוני העמודה הרלוונטית במקרה זה המילה וכמה פעמים מופיעה במאמר 
#x = sd.iloc[:, [0,2]].values
#print(x)
#הדפסה של אותו דבר בלי העמודה הראשונה עם המילים
X = sd.iloc[:, [2]].values
#X=np.sort(X)
#print(y)
#X = X.reshape(-1, 1) 
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
reduced_data = PCA(n_components=4).fit_transform(data)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2','pca3','pca4'])

labels = kmeans.labels_
for i in range(len(X)):
    distance = abs(X[i]-kmeans.cluster_centers_[labels[i]] )
    print(X[i],labels[i],distance)
    

plt.scatter(X[:,0],X[:,0],s=50,cmap='rainbow')
    


# In[ ]:




