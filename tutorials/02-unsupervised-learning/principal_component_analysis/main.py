import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

import warnings
warnings.filterwarnings("ignore")

# Iris dataset
iris = datasets.load_iris(as_frame=True)
print(iris.DESCR)

# Features and y_true
X = iris.data
y = iris.target

# Normalizing data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating df
df = pd.DataFrame(X_scaled, columns=X.columns)
df['target'] = y

# Number of components
n_components = X_scaled.shape[1]

# Principal Component Analysis (PCA)
pca = PCA(n_components=n_components, random_state=42)
pca_data = pca.fit_transform(df[X.columns])

# Variance explained by each PC
exp_var = pca.explained_variance_ratio_

print("\nVariance explained by each PC:")

sum = 0
for i in range(0, n_components):
    var = exp_var[i]
    sum += var
    print(f"PC{i + 1}: {sum}")

# Visualizing PCs
df['PC1'] = pca_data[:, 0]
df['PC2'] = pca_data[:, 1]
df['PC3'] = pca_data[:, 2]

plt.figure(figsize=(6, 6))

sns.scatterplot(x="PC1", y="PC2", hue="target", data=df)

plt.show()

# PC DataFrame
components = ['PC1', 'PC2', 'PC3', 'PC4']
pca_df = pd.DataFrame(pca.components_, index=components, columns=X.columns)

print('\nPCA Analysis')
print(pca_df.T)