# Importing libraries
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from time import time

import warnings

warnings.filterwarnings("ignore")

# Breast Cancer dataset
breast_cancer = load_breast_cancer(as_frame=True)
data = breast_cancer.frame

# Features and y_true
X = breast_cancer.data
y = breast_cancer.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Normalizing data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Hyperparameter tuning
estimator = RandomForestClassifier(random_state=42,
                                   class_weight={1: 0.626374, 0: 0.373626})

# Parameter grid
param_grid = {
    "n_estimators": [50, 100, 200, 350, 500],
    "max_depth": [5, 7, 9, 12],
    "min_samples_leaf": [20, 25],
    "max_features": [0.8, 0.9],
    "max_samples": [0.9, 1],
}

# GridSearchCV
grid_search_cv = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              scoring='f1',
                              cv=5,
                              error_score="raise",
                              verbose=1)
# RandomizedSearchCV
random_search_cv = RandomizedSearchCV(estimator=estimator,
                                      param_distributions=param_grid,
                                      scoring="f1",
                                      n_iter=10,
                                      cv=5,
                                      random_state=42,
                                      verbose=1)
# HalvingGridSearchCV
halving_search = HalvingGridSearchCV(estimator=estimator,
                                     param_grid=param_grid,
                                     scoring="f1",
                                     cv=5,
                                     random_state=42,
                                     verbose=1)

grid = [grid_search_cv, random_search_cv, halving_search]

for grid in grid:
    print(f"Fitting {grid.__class__.__name__}\n")
    t0 = time()
    grid.fit(X_train_scaled, y_train)
    t1 = time()

    print(f'results from {grid.__class__.__name__}')
    print(f'best estimator: {grid.best_estimator_}')
    print(f'f1 score: {grid.best_score_:.4}')
    print(f'best params: {grid.best_params_}')
    print(f'time to fit: {(t1 - t0):.4} seconds')
    print()