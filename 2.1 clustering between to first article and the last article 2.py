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
kmeans3.cluster_centers_

plt.scatter(x[:,0],x[:,1],c=y_kmeans4,cmap='rainbow')

