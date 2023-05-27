import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from time import time
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

# TSNE and PCA
print('Computing t-SNE and PCA...')
t0 = time()
tsne = TSNE(n_components=2, init="pca", random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
print('t-SNE done! Time elapsed: {} seconds'.format(time() - t0))

t0 = time()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print('PCA done! Time elapsed: {} seconds'.format(time() - t0))

# plotting
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('t-SNE')
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, legend='full', palette=sns.color_palette("bright", n_digits))
plt.subplot(1, 2, 2)
plt.title('PCA')
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, legend='full', palette=sns.color_palette("bright", n_digits))
plt.show()

# 3D visualization
tsne = TSNE(n_components=3, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)

# plotting
plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.scatter3D(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap='tab10')
plt.show()
