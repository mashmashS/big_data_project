import seaborn as sns; sns.set()
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


store_data = pd.read_csv('C:\\Datasets\\firstvslast.csv')
sd=store_data.head(100)

x = sd.iloc[:, [1,2]].values
print(x)


kmeans4 = KMeans(n_clusters=4)
y_kmeans4 = kmeans4.fit_predict(x)
print(y_kmeans4)
print(kmeans4.cluster_centers_)

plt.scatter(x[:,0],x[:,1],c=y_kmeans4,cmap='rainbow')
plt.scatter(kmeans4.cluster_centers_[:,0] ,kmeans4.cluster_centers_[:,1],color='black')

output:
[[ 1  1]
 [ 1  4]
 [ 4  3]
 [ 1  2]
 [ 1  1]
 [ 1  1]
 [ 3  5]
 [ 1  2]
 [ 4  2]
 [ 3  1]
 [ 1  8]
 [ 1  1]
 [ 1  1]
 [ 1  2]
 [ 1  1]
 [ 1  4]
 [12  2]
 [ 1  1]
 [ 3  3]
 [ 1  6]
 [ 3  9]
 [ 1  1]
 [ 1  1]
 [ 1  1]
 [ 4  2]
 [ 2  1]
 [ 4  1]
 [ 3  6]
 [ 2  3]
 [ 1  2]
 [ 3  1]
 [ 2  1]
 [ 2  4]
 [ 4 32]
 [ 1  1]
 [ 1  1]
 [ 9  1]
 [ 4  1]
 [ 5  1]
 [ 2  2]
 [ 3  2]
 [ 1  1]
 [ 4  3]
 [ 2  4]
 [ 7  6]
 [ 1  1]
 [13  7]
 [ 5  1]
 [ 1  2]
 [ 1  1]
 [ 5 10]
 [ 2  1]
 [ 1  2]
 [ 2  2]
 [ 1  1]
 [ 4  2]
 [ 2  1]
 [ 8  5]
 [ 3  1]
 [ 2  1]
 [ 1  1]
 [ 1  1]
 [ 1  2]
 [ 1  3]
 [ 1 14]
 [10 10]
 [ 2  5]
 [ 2  1]
 [ 1  2]
 [ 1 17]
 [ 3  6]
 [ 3  1]
 [ 1  1]
 [ 2  1]
 [ 2  2]
 [ 1  1]
 [ 1 13]
 [ 1 11]
 [ 1  2]
 [ 1  3]
 [ 1  2]
 [ 9  3]
 [ 1  2]
 [ 1  1]
 [ 1  1]
 [ 2 26]
 [ 1  1]
 [ 2  2]
 [ 1 13]
 [ 2  1]
 [ 2  1]
 [ 1  3]
 [ 3  3]
 [ 7 14]
 [ 1  1]
 [ 2 16]
 [ 3  3]
 [ 4  3]
 [ 2  2]
 [ 3  8]]
[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 2 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 2
 0 0 0 0 0 0 0 2 0 2 0 0 0 1 0 0 0 0 0 0 2 0 0 0 0 0 0 1 2 0 0 0 1 0 0 0 0
 0 0 1 1 0 0 0 2 0 0 0 3 0 0 1 0 0 0 0 1 0 1 0 0 0 1]
[[ 1.9         1.9375    ]
 [ 2.36363636 12.09090909]
 [ 9.71428571  4.85714286]
 [ 3.         29.        ]]
<matplotlib.collections.PathCollection at 0x1fc0a2c3e08>  
*Graph is in the main file of the project
