import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn import datasets
import warnings

warnings.filterwarnings("ignore")

# Iris dataset
iris = datasets.load_wine(as_frame=True)

# Features and y_true
X = iris.data
y = iris.target

# Normalizing data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating Dataframe
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled.head())

# Elbow plot
sse = {}

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)

    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), 'bx-')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

# Silhouette score
sc = {}

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    labels = kmeans.predict(X_scaled)
    sc[k] = silhouette_score(X_scaled, labels)

plt.figure()
plt.plot(list(sc.keys()), list(sc.values()), 'bx-')
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette Score")
plt.show()

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Creating DataFrame
df = X_scaled
df['cluster'] = kmeans.predict(X_scaled)
df['target'] = y

print(df.head())

print(df['cluster'].value_counts())

print(df['target'].value_counts())