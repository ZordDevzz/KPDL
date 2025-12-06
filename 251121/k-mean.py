import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans_numpy(X, n_clusters, max_iters=100):
    idx = np.random.choice(len(X), n_clusters, replace=False)
    centroids = X[idx]

    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        new_centroids = np.array(
            [X[labels == i].mean(axis=0) for i in range(n_clusters)]
        )
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

def kmeans_sklearn(X, n_clusters):
    wcss_list = []
    for _ in range(1, 11):
        kmeans = KMeans(
            n_clusters=_,
            init="k-means++",
            max_iter=300,
            n_init=10,
            random_state=None,
        )
        kmeans.fit(X)
        print(kmeans.inertia_)
        wcss_list.append(kmeans.inertia_)
    return wcss_list


data = pd.read_csv("./Mall_Customers.csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]].values


# clustering data to 5 clusters using custom numpy kmeans
n_clusters = 5
# labels_numpy, centroids_numpy = kmeans_numpy(X, n_clusters)
wcss_kmeans_sklearn = kmeans_sklearn(X, n_clusters)
print(f"WCSS for {n_clusters} clusters: {wcss_kmeans_sklearn}")
plt.plot(range(1, 11), wcss_kmeans_sklearn)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.savefig("elbow_method_sklearn.png")

# plotting the clusters
kmeans = KMeans(
    n_clusters=n_clusters,
    init="k-means++",
    max_iter=300,
    n_init=10,
    random_state=None,
)
kmeans.fit(X)
labels_sklearn = kmeans.labels_
centroids_sklearn = kmeans.cluster_centers_
plt.figure()
for i in range(n_clusters):
    plt.scatter(
        X[labels_sklearn == i, 0],
        X[labels_sklearn == i, 1],
        label=f"Cluster {i + 1}",
    )
plt.scatter(
    centroids_sklearn[:, 0],
    centroids_sklearn[:, 1],
    s=300,
    c="yellow",
    label="Centroids",
)
plt.title("Clusters of customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.savefig("clusters_sklearn.png")