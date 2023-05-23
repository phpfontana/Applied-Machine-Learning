from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import datasets

# filter warnings
import warnings
warnings.filterwarnings("ignore")

# California housing dataset
california_housing = datasets.fetch_california_housing()
X = california_housing.data
y = california_housing.target

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Ridge Regression
ridge_regression = Ridge(alpha=0.5)
ridge_regression.fit(X_train, y_train)

# Lasso Regression
lasso_regression = Lasso(alpha=0.5)
lasso_regression.fit(X_train, y_train)

# SGD Regression
sgd_regression = SGDRegressor()
sgd_regression.fit(X_train, y_train)

# testing and evaluation
for model in [linear_regression, ridge_regression, lasso_regression, sgd_regression]:

    # y prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # evaluation
    print(model.__class__.__name__)
    print('Train R2 score: ', r2_score(y_train, y_pred_train))
    print('Test R2 score: ', r2_score(y_test, y_pred_test))
    print('Train RMSE: ', mean_squared_error(y_train, y_pred_train, squared=False))
    print('Test RMSE: ', mean_squared_error(y_test, y_pred_test, squared=False))
    print()
