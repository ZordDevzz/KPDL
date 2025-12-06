import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids


def kmedoids_sklearn(X, n_clusters):
    wcss_list = []
    for i in range(1, 11):
        kmedoids = KMedoids(n_clusters=i, random_state=0)
        kmedoids.fit(X)
        wcss_list.append(kmedoids.inertia_)
    return wcss_list


data = pd.read_csv("./Mall_Customers.csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]].values


# clustering data to 5 clusters using KMedoids
n_clusters = 5
wcss_kmedoids_sklearn = kmedoids_sklearn(X, n_clusters)
print(f"WCSS for {n_clusters} clusters: {wcss_kmedoids_sklearn}")
plt.plot(range(1, 11), wcss_kmedoids_sklearn)
plt.title("The Elbow Method for KMedoids")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.savefig("elbow_method_pam.png")

# plotting the clusters
kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
kmedoids.fit(X)
labels_sklearn = kmedoids.labels_
medoids_sklearn = kmedoids.cluster_centers_

plt.figure()
for i in range(n_clusters):
    plt.scatter(
        X[labels_sklearn == i, 0],
        X[labels_sklearn == i, 1],
        label=f"Cluster {i + 1}",
    )
plt.scatter(
    medoids_sklearn[:, 0],
    medoids_sklearn[:, 1],
    s=300,
    c="yellow",
    label="Medoids",
)
plt.title("Clusters of customers (KMedoids)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.savefig("clusters_pam.png")
