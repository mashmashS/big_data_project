import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


#sum all of the columns
sd = pd.read_csv('D:\\Datasets\\NIPS_1987-2015.csv')
X = sd.iloc[:, [3]].values
X = X.reshape(-1, 1) 
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

labels = kmeans.labels_
for i in range(len(X)):
    distance = abs(X[i]-kmeans.cluster_centers_[labels[i]] )
#print(X[i],labels[i],distance)
    
mean=kmeans.cluster_centers_[labels[i]]
print("mean is:",mean)

X = sd.iloc[:, [3]].values
counter=0
for i in X:
    if (abs(i-mean)>=2):
        print(i, "is unlikely!")
        counter+=1
print(counter/11463)

plt.scatter(X[:,0],X[:,0],s=50,cmap='rainbow')

output:
mean is: [0.02601438]
[6] is unlikely!
[3] is unlikely!
[4] is unlikely!
[3] is unlikely!
[3] is unlikely!
[4] is unlikely!
[6] is unlikely!
[3] is unlikely!
[6] is unlikely!
[5] is unlikely!
[7] is unlikely!
[6] is unlikely!
[4] is unlikely!
[3] is unlikely!
[4] is unlikely!
[5] is unlikely!
[3] is unlikely!
[3] is unlikely!
[3] is unlikely!
[4] is unlikely!
[3] is unlikely!
[4] is unlikely!
[3] is unlikely!
[5] is unlikely!
[5] is unlikely!
[15] is unlikely!
[12] is unlikely!
[10] is unlikely!
[4] is unlikely!
[4] is unlikely!
[4] is unlikely!
[19] is unlikely!
[7] is unlikely!
[7] is unlikely!
[4] is unlikely!
[4] is unlikely!
[6] is unlikely!
[9] is unlikely!
[3] is unlikely!
[3] is unlikely!
[4] is unlikely!
[9] is unlikely!
[5] is unlikely!
[12] is unlikely!
[28] is unlikely!
[11] is unlikely!
[3] is unlikely!
[6] is unlikely!
[4] is unlikely!
[3] is unlikely!
[4] is unlikely!
[4] is unlikely!
[13] is unlikely!
[5] is unlikely!
[4] is unlikely!
[3] is unlikely!
[3] is unlikely!
[3] is unlikely!
[3] is unlikely!
[9] is unlikely!
[12] is unlikely!
[12] is unlikely!
[6] is unlikely!
[5] is unlikely!
[13] is unlikely!
[6] is unlikely!
[3] is unlikely!
[4] is unlikely!
[45] is unlikely!
[10] is unlikely!
[7] is unlikely!
[3] is unlikely!
[11] is unlikely!
[6] is unlikely!
[4] is unlikely!
[3] is unlikely!
[6] is unlikely!
[6] is unlikely!
[4] is unlikely!
[10] is unlikely!
[9] is unlikely!
[4] is unlikely!
[3] is unlikely!
[3] is unlikely!
[3] is unlikely!
[3] is unlikely!
[4] is unlikely!
[3] is unlikely!
[3] is unlikely!
[3] is unlikely!
[4] is unlikely!
[3] is unlikely!
[3] is unlikely!
[4] is unlikely!
[3] is unlikely!
[5] is unlikely!
[4] is unlikely!
[4] is unlikely!
[4] is unlikely!
[5] is unlikely!
[4] is unlikely!
[6] is unlikely!
[14] is unlikely!
[3] is unlikely!
[5] is unlikely!
[3] is unlikely!
[5] is unlikely!
[3] is unlikely!
[6] is unlikely!
[3] is unlikely!
[6] is unlikely!
[8] is unlikely!
[10] is unlikely!
[3] is unlikely!
[5] is unlikely!
[10] is unlikely!
[10] is unlikely!
[4] is unlikely!
[3] is unlikely!
0.010381226555003053
<matplotlib.collections.PathCollection at 0x199bd41db08>
*Graph is in the main file

    
