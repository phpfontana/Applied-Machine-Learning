# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

# loading dataset
digits = load_digits()
X = digits.data
y = digits.target
n_digits = len(np.unique(y))

# scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=n_digits, random_state=0)
kmeans.fit(X_scaled)

# K-Medoids
kmedoids = KMedoids(n_clusters=n_digits, random_state=0)
kmedoids.fit(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=3, min_samples=3)
dbscan.fit(X_scaled)

# evaluation
for estimator in [kmeans, kmedoids, dbscan]:
    print(estimator.__class__.__name__)
    print('Silhouette Score: ', silhouette_score(X_scaled, estimator.labels_))
    print('Homogeneity Score: ', homogeneity_score(y, estimator.labels_))
    print('Completeness Score: ', completeness_score(y, estimator.labels_))
    print('V-Measure Score: ', v_measure_score(y, estimator.labels_))
    print('Adjusted Rand Score: ', adjusted_rand_score(y, estimator.labels_))
    print('Adjusted Mutual Info Score: ', adjusted_mutual_info_score(y, estimator.labels_))
    print()

# visualization
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)
kmeans.fit(X_tsne)
kmedoids.fit(X_tsne)

# calculating centroids
kmeans_centroids = kmeans.cluster_centers_
kmedoids_centroids = kmedoids.cluster_centers_
labels = kmeans.predict(X_tsne)
unique_labels = np.unique(labels)

# plotting
plt.figure(figsize=(10, 10))
for i in unique_labels:
    plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1], label=i)
plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], marker='*', s=250, color='black', label='K-Means Centroids')
plt.scatter(kmedoids_centroids[:, 0], kmedoids_centroids[:, 1], marker='^', s=250, color='black', label='K-Medoids Centroids')
plt.legend()
plt.show()






