# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer

import warnings
warnings.filterwarnings("ignore")

# Breast Cancer Dataset
breast_cancer = load_breast_cancer(as_frame=True)
X = breast_cancer.data
y = breast_cancer.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Normalizing data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Random Forest classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train_scaled, y_train)

# Making y_pred
y_pred_train = random_forest.predict(X_train_scaled)
y_pred_test = random_forest.predict(X_test_scaled)

print("Train Predictions")
print(classification_report(y_true=y_train, y_pred=y_pred_train))

print("Test Predictions")
print(classification_report(y_true=y_test, y_pred=y_pred_test))

# Hyperparameter tuning
estimator = RandomForestClassifier(random_state=42,
                                   class_weight={1: 0.626374, 0: 0.373626})

# Parameter grid
param_grid = {"n_estimators": [110, 120],
              "max_depth": [6, 7],
              "min_samples_leaf": [20, 25],
              "max_features": [0.8, 0.9],
              "max_samples": [0.9, 1],
              }

# Running GridSearchCV
grid_search_cv = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              scoring='f1',
                              cv=5,
                              error_score="raise",
                              verbose=1)

grid_search_cv.fit(X_train_scaled, y_train)

print('\nResults from GridSearchCV')
print(f'Best estimator:{grid_search_cv.best_estimator_}')
print(f'F1 score: {grid_search_cv.best_score_}')
print(f'Best parameters:{grid_search_cv.best_params_}')

# Training tuned decision tree
rf_tuned = grid_search_cv.best_estimator_
rf_tuned.fit(X_train_scaled, y_train)

# Making y_pred
y_pred_train = rf_tuned.predict(X_train_scaled)
y_pred_test = rf_tuned.predict(X_test_scaled)

print("Train Predictions")
print(classification_report(y_true=y_train, y_pred=y_pred_train))

print("Test Predictions")
print(classification_report(y_true=y_test, y_pred=y_pred_test))

# Plotting feature importance
importances = rf_tuned.feature_importances_

indices = np.argsort(importances)

feature_names = list(X.columns)

plt.figure(figsize=(12, 12))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='violet', align='center')

plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()