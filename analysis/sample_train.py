import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("Start sampling")

data = pd.read_csv("../Data/train.csv")

cont_sample_multidim = []
sample_points = np.random.randint(150000, len(data) + 1, 1000)

for point in sample_points:
    new_point_X = []
    temp = pd.DataFrame(data.iloc[point])
    new_point_y = temp.loc['time_to_failure'].to_list()[0]
    
    for i in range(150000):
        temp = pd.DataFrame(data.iloc[point - i])
        temp = temp.loc['acoustic_data'].to_list()[0]
        new_point_X.append(temp)
    
    cont_sample_multidim.append([new_point_X, new_point_y])

X = []
y = []
for i in range(len(cont_sample_multidim)):
    X.append(cont_sample_multidim[i][0])
    y.append(cont_sample_multidim[i][1])

pd.DataFrame(X).to_csv("rand_mdim_sample_100_X", index=False)
pd.DataFrame(y).to_csv("rand_mdim_sample_100_y", index=False)

best_score = np.infty

print("Done sampling")
print("\nStart KNN")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

results = []

for k in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_test = knn.predict(X_test)
    score = root_mean_squared_error(y_pred_test, y_test)
    results.append([k, score])
    if score < best_score:
        best_score = score

print(results)

print("\nStart LR")
results = []

for i in range(5):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_test = lr.predict(X_test)
    score = root_mean_squared_error(y_pred_test, y_test)
    results.append([i, score])
    if score < best_score:
        best_score = score

print(results)

print("\nStart SVR")
results = []

for e in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
    for c in [0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 1000]:
        regr = make_pipeline(StandardScaler(), SVR(C=c, epsilon=e))
        regr.fit(X, y)
        y_pred_test = regr.predict(X_test)
        score = root_mean_squared_error(y_pred_test, y_test)
        results.append([e, k, score])
        if score < best_score:
            best_score = score

print(results)

print("best score: ", best_score)
