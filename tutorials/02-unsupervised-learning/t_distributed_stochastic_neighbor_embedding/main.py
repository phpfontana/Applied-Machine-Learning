import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import datasets
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Iris dataset
iris = datasets.load_iris(as_frame=True)

# Features and y_true
X = iris.data
y = iris.target

# Normalizing data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating df
df = pd.DataFrame(X_scaled, columns=X.columns)
df['target'] = y

# t-SNE
for i in np.arange(10, 51, 5):
    tsne = TSNE(n_components=2, random_state=42, n_iter=500, verbose=1, perplexity=i)
    tsne_data = tsne.fit_transform(X_scaled)

    # 2D visualization
    df['C1'] = tsne_data[:, 0]
    df['C2'] = tsne_data[:, 1]

    plt.figure(figsize=(6, 6))

    sns.scatterplot(x="C1", y="C2", hue="target", data=df).set(title=f'Perplexity {i}')

    plt.show()