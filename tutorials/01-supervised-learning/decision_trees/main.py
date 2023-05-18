import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_wine
from sklearn import metrics

# filter warnings
import warnings
warnings.filterwarnings("ignore")

# Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Normalizing Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# One-Hot encoding y_true
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.fit_transform(y_test.reshape(-1, 1))

# Decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_scaled, y_train_encoded)

# Making y_pred
y_pred_train = decision_tree.predict(X_train_scaled)
y_pred_test = decision_tree.predict(X_test_scaled)

print("Train Predictions")
print(classification_report(y_true=y_train_encoded, y_pred=y_pred_train))

print("Test Predictions")
print(classification_report(y_true=y_test_encoded, y_pred=y_pred_test))

# Hyperparameter Tuning
estimator = DecisionTreeClassifier(random_state=42)

# Parameter grid
param_grid = {'max_depth': np.arange(2, 10),
              'criterion': ['gini', 'entropy'],
              'min_samples_leaf': [5, 10, 20, 25]
              }

# Running GridSearchCV
grid_search_cv = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              scoring='f1_micro',
                              cv=5,
                              error_score="raise")

grid_search_cv.fit(X_train_scaled, y_train_encoded)

print('\nResults from GridSearchCV')
print(f'Best estimator: {grid_search_cv.best_estimator_}')
print(f'F1 score: {grid_search_cv.best_score_}')
print(f'Best parameters: {grid_search_cv.best_params_}')

# Training tuned decision tree
dt_tuned = grid_search_cv.best_estimator_
dt_tuned.fit(X_train_scaled, y_train_encoded)

print("\nTuned Decision Tree Classifier")
# Making y_pred
y_pred_train = dt_tuned.predict(X_train_scaled)
y_pred_test = dt_tuned.predict(X_test_scaled)

print("Train Predictions")
print(classification_report(y_true=y_train_encoded, y_pred=y_pred_train))

print("Test Predictions")
print(classification_report(y_true=y_test_encoded, y_pred=y_pred_test))