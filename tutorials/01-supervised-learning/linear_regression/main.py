from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import datasets

# filter warnings
import warnings
warnings.filterwarnings("ignore")

# Hyper-parameters

# Name dataset
data = datasets.fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Normalizing Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train_scaled, y_train)

# Ridge regression
ridge_regression = Ridge()
ridge_regression.fit(X_train_scaled, y_train)

# Lasso regression
lasso_regression = Lasso()
lasso_regression.fit(X_train_scaled, y_train)

# SGD linear regression
sgd_linear_regression = SGDRegressor()
sgd_linear_regression.fit(X_train_scaled, y_train)

# Testing and Evaluation
for model in linear_regression, lasso_regression, ridge_regression, sgd_linear_regression:

    # Making y_pred
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # R2 score
    r2_train = r2_score(y_true=y_train, y_pred=y_pred_train)
    r2_test = r2_score(y_true=y_test, y_pred=y_pred_test)

    # MSE
    mse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    mse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)

    print(model.__class__.__name__)
    print(f"Train | R-squared : {r2_train} | MSE: {mse_train}")
    print(f"Test | R-squared : {r2_test} | MSE: {mse_test}\n")