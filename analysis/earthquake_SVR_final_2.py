print("################################\nStarting imports.")
import time

start = time.time()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.svm import SVR
import joblib

current = time.time()
print("Done with imports - ", current - start, "s\n")

# synthesize data by taking every 150000th point and turning it 
# into a single data point with features using the previous 150000 points

print("\nStarting data read")
X = pd.read_csv("rand_mdim_sample_100_X", index_col=False)
y = pd.read_csv("rand_mdim_sample_100_y", index_col=False)

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

current = time.time()
print("Done with data read\n", current - start, "s\n")
print("Starting parameter search.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print("\tNew Data split.")
print("\tTrain size = ", len(X_train))
print("\tTest size = ", len(X_test))

# set parameter grid for clf
params = {
    "C": [1e5, 1e6],
    "epsilon": [500, 1e5, 1e6],
    "kernel": ["rbf", "sigmoid"],
    "gamma": ["auto"]
}

# best model for our final test is svr according to initial testing
svr = SVR()
clf = GridSearchCV(estimator=svr, param_grid=params, scoring="neg_root_mean_squared_error")

clf.fit(X_train, y_train)
print("\tClf fit to data")

print(f"\tBest score:        {clf.best_score_} rmse")
print(f"\tBest parameters:        {clf.best_params_}")

current = time.time()
print("Done with parameter search - ", current - start, "s\n")

y_pred = clf.predict(X_test)

print(f"Test Results:        {mean_squared_error(y_test, y_pred, squared=False)} RMSE")
print(f"Test Results:        {mean_absolute_percentage_error(y_test, y_pred)} MAPE")

print("\nSaving model.")
joblib.dump(clf, "earthquake_model.pkl") 
current = time.time()
print("Done saving model - ", current - start, "s\n")

current = time.time()
print("End program - ", current - start, "s\n################################")
