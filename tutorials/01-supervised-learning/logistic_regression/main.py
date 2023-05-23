import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
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

# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# testing and evaluation
y_pred_train = logistic_regression.predict(X_train)
y_pred_test = logistic_regression.predict(X_test)

print('Train Classification Report: ')
print(classification_report(y_train, y_pred_train))

print('Test Classification Report: ')
print(classification_report(y_test, y_pred_test))

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, logistic_regression.predict_proba(X_test)[:, 1])

# finding the optimal threshold
for i in np.arange(len(thresholds)):
    if precision[i] == recall[i]:
        optimal_threshold = thresholds[i]
        print(f"Precision-Recall threshold: {thresholds[i]}\n")

# evaluating the model with the optimal threshold
y_pred_test = (logistic_regression.predict_proba(X_test)[:, 1] >= optimal_threshold)
print('Test Classification Report: ')
print(classification_report(y_test, y_pred_test))

