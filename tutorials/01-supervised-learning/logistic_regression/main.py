import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import warnings

# filter warnings
import warnings
warnings.filterwarnings("ignore")

# Breast Cancer dataset
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Normalizing Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Logistic regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_scaled, y_train)

# Making y_pred
y_pred_train = logistic_regression.predict(X_train_scaled)
y_pred_test = logistic_regression.predict(X_test_scaled)

print("Train Predictions")
print(classification_report(y_true=y_train, y_pred=y_pred_train))

print("Test Predictions")
print(classification_report(y_true=y_test, y_pred=y_pred_test))

# Probas
y_train_proba = logistic_regression.predict_proba(X_train_scaled)
y_test_proba = logistic_regression.predict_proba(X_test_scaled)
precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba[:,1])

# Precision-recall optimal threshold
for i in np.arange(len(thresholds)):
    if precisions[i] == recalls[i]:
        optimal_threshold = thresholds[i]
        print(f"Precision-Recall threshold: {thresholds[i]}\n")

# Evaluating probas with optimal threshold
print("Train Probas")
print(classification_report(y_train, y_train_proba[:, 1] > optimal_threshold))

print("Test Probas")
print(classification_report(y_test, y_test_proba[:, 1] > optimal_threshold))
