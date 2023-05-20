# Importing libraries
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import make_column_selector as selector

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

import sklearn
from time import time

sklearn.set_config(display="diagram")

# Breast Cancer dataset
data = load_breast_cancer(as_frame=True)

# Features and y_true
X = data.data
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Numeric processing pipeline
num_transformer = Pipeline(
    steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('transformer', PowerTransformer(method='yeo-johnson', standardize=False),),
        ('scaler', StandardScaler())
    ]
)

# Category processing pipeline
cat_transformer = Pipeline(
    steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('impute', SimpleImputer(strategy='most_frequent'))
    ]
)

# Column preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, selector(dtype_exclude='category')),
        ('cat', cat_transformer, selector(dtype_include='category'))
    ]
)

# Model selection
classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    GaussianNB(),
    XGBClassifier()
]

results = []
train_time = []

for c in classifiers:

    # Model pipeline
    pipe = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', c)
        ])

    # Fitting models
    t0 = time()
    pipe.fit(X_train, y_train)
    t1 = time()

    # Making y_pred
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    target_names = list(data.target_names)

    # printing results
    print(f'Classifier: {c.__class__.__name__}')

    print(f'train acc: {accuracy_score(y_train, y_train_pred):.4}')
    print(f'test acc: {accuracy_score(y_test, y_test_pred):.4}')

    print(f'training time: {(t1-t0):.4} seconds')
    print()

    results.append(accuracy_score(y_test, y_test_pred))
    train_time.append(t1-t0)

best_score = np.argmax(results)
print(f'best classifier: {classifiers[best_score]}')
print(f'acc score: {(results[best_score]):.4}')

print()

best_time = np.argmax(train_time)
print(f'fastest classifier: {classifiers[best_time]}')
print(f'time to train: {(train_time[best_time]):.4}')