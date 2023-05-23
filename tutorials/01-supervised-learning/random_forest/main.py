# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer

import warnings
warnings.filterwarnings("ignore")

# Breast cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# testing and evaluation
y_pred_train = random_forest.predict(X_train)
y_pred_test = random_forest.predict(X_test)

print('Train Classification Report: ')
print(classification_report(y_train, y_pred_train))

print('Test Classification Report: ')
print(classification_report(y_test, y_pred_test))

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}
# Running GridSearchCV to find the best parameters, this will take a few minutes...
grid_search = GridSearchCV(estimator=random_forest,
                           scoring="f1",
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_train, y_train)

print("Results from Grid Search")
print(f'Best estimator:{grid_search.best_estimator_}')
print(f'F1 score: {grid_search.best_score_}')
print(f'Best parameters:{grid_search.best_params_}')

# training the model with the best parameters
rf_tuned = grid_search.best_estimator_
rf_tuned.fit(X_train, y_train)

# testing and evaluation
y_pred_train = rf_tuned.predict(X_train)
y_pred_test = rf_tuned.predict(X_test)

print('Train Classification Report: ')
print(classification_report(y_train, y_pred_train))

print('Test Classification Report: ')
print(classification_report(y_test, y_pred_test))

# plotting the feature importance
feature_importances = rf_tuned.feature_importances_

indices = np.argsort(feature_importances)[::-1]

feature_names = breast_cancer.feature_names

plt.figure(figsize=(12, 12))

plt.title('Feature Importances')

plt.barh(range(len(indices)), feature_importances[indices], color='violet', align='center')

plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
