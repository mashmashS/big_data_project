from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as shc
import pandas as pd

sd = pd.read_csv('D:\\Datasets\\DtmMinusZeros.csv')
sd=sd.head(25)
print(sd)
x= sd.iloc[:, [0,1]].values
linked = linkage(x, 'ward')
labelList = range(1, 26)
plt.figure(figsize=(10, 9))
plt.title("comparison between article 1987_1 to article 1987_2")
dendrogram(linked,orientation='top',labels=labelList,distance_sort='ascending',show_leaf_counts=True)
plt.show()
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(sd)
plt.figure(figsize=(6, 5))
plt.scatter(x[:,0], x[:,1], c=cluster.labels_, cmap='rainbow')

פלט:
    1987_1  1987_2
0        1       1
1        4      13
2        1       1
3        1       3
4        1       1
5        1      16
6        1       1
7        1       1
8        2       1
9        1       3
10       1       1
11       1       1
12       3       3
13       1       5
14       4       1
15       3       4
16       2       3
17       2       1
18       1       2
19       1       1
20       1       3
21       1       5
22       3       4
23      12       2
24       3       1


array([1, 0, 1, 3, 1, 0, 1, 1, 1, 3, 1, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3,
       3, 2, 1], dtype=int64)
