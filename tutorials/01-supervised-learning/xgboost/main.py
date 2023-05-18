import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import datasets
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import xgboost as xgb

# Breast Cancer Dataset
breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Normalizing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Classification matrices
dtrain = xgb.DMatrix(X_train_scaled, y_train)
dtest = xgb.DMatrix(X_test_scaled, y_test)

# XGBoost classifier
xgb = XGBClassifier(n_estimators=100, objective='binary:logistic', random_state=42)
xgb.fit(X_train_scaled, y_train)

# Making y_pred
y_pred_train = xgb.predict(X_train_scaled)
y_pred_test = xgb.predict(X_test_scaled)

print("Train Predictions")
print(classification_report(y_true=y_train, y_pred=y_pred_train))

print("Test Predictions")
print(classification_report(y_true=y_test, y_pred=y_pred_test))