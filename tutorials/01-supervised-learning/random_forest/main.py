# Importing libraries
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer

import warnings
warnings.filterwarnings("ignore")

# 01 Data
print("01. Data")

# Breast Cancer Dataset
breast_cancer = load_breast_cancer(as_frame=True)
print(breast_cancer.DESCR)

data = breast_cancer.frame
print(data.head())

# Features and y_true
X = breast_cancer.data
y = breast_cancer.target

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape: {y.shape}\n")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)
print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Percentage of classes in the training set:\n{y_train.value_counts(normalize=True)}")
print(f"Percentage of classes in the test set:\n{y_test.value_counts(normalize=True)}\n")

# Normalizing data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# 02 Model Building
print("02 Model Building")

# Decision tree classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train_scaled, y_train)

print('Base Random Forest Classifier')
print(random_forest)

# 03 Testing and evaluation
print("\n03 Testing and evaluation")

# Making y_pred
y_pred_train = random_forest.predict(X_train_scaled)
y_pred_test = random_forest.predict(X_test_scaled)

print("Train Predictions")
print(classification_report(y_true=y_train, y_pred=y_pred_train))

print("Test Predictions")
print(classification_report(y_true=y_test, y_pred=y_pred_test))

# 04 Hyperparameter tuning
print("\n04. Hyperparameter Tuning")

# Estimator
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
                              verbose=2)

grid_search_cv.fit(X_train_scaled, y_train)

print('\nResults from GridSearchCV')
print(f'Best estimator:\n{grid_search_cv.best_estimator_}')
print(f'\nF1 score: {grid_search_cv.best_score_}')
print(f'\nBest parameters:\n{grid_search_cv.best_params_}')

# Training tuned decision tree
rf_tuned = grid_search_cv.best_estimator_
rf_tuned.fit(X_train_scaled, y_train)

print("\nTuned Random Forest Classifier")
print(rf_tuned)

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

# plt.show()