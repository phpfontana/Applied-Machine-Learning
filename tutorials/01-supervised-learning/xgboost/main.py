import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import datasets
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import xgboost as xgb

# filter warnings
import warnings

warnings.filterwarnings("ignore")

# Breast cancer dataset
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost Classifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)

# testing and evaluation
y_pred_train = xgb_classifier.predict(X_train)
y_pred_test = xgb_classifier.predict(X_test)

print('Train Classification Report: ')
print(classification_report(y_train, y_pred_train))

print('Test Classification Report: ')
print(classification_report(y_test, y_pred_test))

# Hyperparameter tuning
xgb_classifier = XGBClassifier()
parameters = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
              'max_depth': [1, 2, 3, 4, 5],
              'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
              'min_child_weight': [1, 2, 3, 4, 5],
              'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
              'colsample_bytree': [0.3, 0.4, 0.5, 0.7, 0.8],
              'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
              'reg_alpha': [0.0, 0.1, 0.2, 0.3, 0.4],
              'reg_lambda': [0.0, 0.1, 0.2, 0.3, 0.4]
              }

# Running GridSearchCV to find the best parameters, this will take a few minutes...
grid_search = GridSearchCV(estimator=xgb_classifier,
                           scoring="f1",
                           param_grid=parameters,
                           cv=5,
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_train, y_train)

print("Results from Grid Search")
print(f'Best estimator:{grid_search.best_estimator_}')
print(f'F1 score: {grid_search.best_score_}')
print(f'Best parameters:{grid_search.best_params_}')

# training with best parameters
xgb_classifier = grid_search.best_estimator_
xgb_classifier.fit(X_train, y_train)

# testing and evaluation
y_pred_train = xgb_classifier.predict(X_train)
y_pred_test = xgb_classifier.predict(X_test)

print('Train Classification Report: ')
print(classification_report(y_train, y_pred_train))

print('Test Classification Report: ')
print(classification_report(y_test, y_pred_test))
