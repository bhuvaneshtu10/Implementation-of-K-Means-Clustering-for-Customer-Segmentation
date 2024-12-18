# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. .Import pandas and matplotlib.pyplot
2.Read the dataset and transform it
3.Import KMeans and fit the data in the model
4.Plot the Cluster graph

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: BHUVANESHWARAN TU  
RegisterNumber:  24009351
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
km = KMeans(n_clusters=5, init="k-means++", random_state=42)
data["cluster"] = km.fit_predict(data.iloc[:, 3:])
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")
plt.legend()
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

```

## Output:
![K Means Clustering for Customer Segmentation](sam.png)
![Screenshot 2024-12-18 210837](https://github.com/user-attachments/assets/1475c96f-d382-481e-99e6-5564d0c52b74)
![Screenshot 2024-12-18 210916](https://github.com/user-attachments/assets/e16c8cb8-e873-4ec1-8bcb-f0178f921d85)
![Screenshot 2024-12-18 210924](https://github.com/user-attachments/assets/d85d1eaa-d8a5-4eb9-a9cd-96aefdfccc1d)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
