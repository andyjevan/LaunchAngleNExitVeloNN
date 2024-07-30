#import lines
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

#read in dataframes
LA_df = pd.read_csv('LaunchAngle.csv')
EV_df = pd.read_csv('ExitVelocity.csv')

#print(LA_df.columns)

#calculating slugging percentage
LA_sluglist = []
LA_highestslug = 0
LA_highestslugLA = -90
for index, row in LA_df.iterrows():
    slug = (row['1B'] + (row['2B'] * 2) + (row['3B'] * 3) + (row['HR'] * 4))/row['BBE']
    LA_sluglist.append(slug)
    if slug > LA_highestslug and row['BBE'] > 9:
        LA_highestslug = slug
        LA_highestslugLA = row['Launch Angle (Deg)']
LA_df['Slug'] = LA_sluglist
print(LA_highestslug, LA_highestslugLA)
EV_sluglist = []
EV_highestslug = 0
EV_highestslugEV = 0
for index, row in EV_df.iterrows():
    slug = (row['1B'] + (row['2B'] * 2) + (row['3B'] * 3) + (row['HR'] * 4))/row['BBE']
    EV_sluglist.append(slug)
    if slug > EV_highestslug and row['BBE'] > 9:
        EV_highestslug = slug
        EV_highestslugEV = row['Exit Velocity (MPH)']
print(EV_highestslug, EV_highestslugEV)
EV_df['Slug'] = EV_sluglist

#data preprocessing Launch Angle
X = np.array(LA_df[['Launch Angle (Deg)','Hits', 'BBE', 'AVG', 'wOBA']])#, '1B', '2B', '3B', 'HR'
y = np.array(LA_df[['Slug']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#Linear Regression Model Launch Angle
lr = LinearRegression().fit(X_train,y_train)
pred = lr.predict(X_test)
#calculate mse
sum = 0
for i in range (0, len(y_test)):
    diff = y_test[i] - pred[i]
    sq_diff = diff ** 2
    sum += sq_diff
LAlr_mse = sum/len(y_test)
print("Linear Regression Launch Angle MSE:", LAlr_mse)
print("Linear Regression Launch Angle R2 Score:", (r2_score(pred, y_test)))

#data preprocessing Exit VelocityA
X = np.array(EV_df[['Exit Velocity (MPH)','Hits', 'BBE', 'AVG', 'wOBA']])
y = np.array(EV_df[['Slug']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#Linear Regression Model Exit Velocity
lr = LinearRegression().fit(X_train,y_train)
pred = lr.predict(X_test)
#calculate mse
sum = 0
for i in range (0, len(y_test)):
    diff = y_test[i] - pred[i]
    sq_diff = diff ** 2
    sum += sq_diff
EVlr_mse = sum/len(y_test)
print("Linear Regression Exit Velocity MSE:", EVlr_mse)
print("Linear Regression Exit Velocity R2 Score:", (r2_score(pred, y_test)))

#data preprocessing decision tree
X = np.array(LA_df[['Launch Angle (Deg)','Hits', 'BBE', 'AVG', 'wOBA']])
y = np.array(LA_df[['Slug']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#Decision Tree Regressor Model Launch Angle
dtr = tree.DecisionTreeRegressor()
dtr = dtr.fit(X_train, y_train)
pred = dtr.predict(X_test)
#calculate mse
sum = 0
for i in range (0, len(y_test)):
    diff = y_test[i] - pred[i]
    sq_diff = diff ** 2
    sum += sq_diff
LAdtr_mse = sum/len(y_test)
print("Decision Tree Regressor Launch Angle MSE:", LAdtr_mse)
print("Decision Tree Regressor Launch Angle R2 Score:", (r2_score(pred, y_test)))

#data preprocessing decision tree
X = np.array(EV_df[['Exit Velocity (MPH)','Hits', 'BBE', 'AVG', 'wOBA']])
y = np.array(EV_df[['Slug']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#Decision Tree Regressor Model Exit Velocity
dtr = tree.DecisionTreeRegressor()
dtr = dtr.fit(X_train, y_train)
pred = dtr.predict(X_test)
#calculate mse
sum = 0
for i in range (0, len(y_test)):
    diff = y_test[i] - pred[i]
    sq_diff = diff ** 2
    sum += sq_diff
EVdtr_mse = sum/len(y_test)
print("Decision Tree Regressor Exit Velocity MSE:", EVdtr_mse)
print("Decision Tree Regressor Exit Velocity R2 Score:", (r2_score(pred, y_test)))

#data preprocessing neural network
X = np.array(LA_df[['Launch Angle (Deg)','Hits', 'BBE', 'AVG', 'wOBA']])
y = np.array(LA_df[['Slug']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#Neural Network Model Launch Angle
nn = MLPRegressor(max_iter=500).fit(X_train, y_train)
pred = nn.predict(X_test)
#calculate mse
sum = 0
for i in range (0, len(y_test)):
    diff = y_test[i] - pred[i]
    sq_diff = diff ** 2
    sum += sq_diff
LAnn_mse = sum/len(y_test)
print("Neural Network Launch Angle MSE:", LAnn_mse)
print("Neural Network Launch Angle R2 Score:", (r2_score(pred, y_test)))

#data preprocessing neural network
X = np.array(EV_df[['Exit Velocity (MPH)','Hits', 'BBE', 'AVG', 'wOBA']])
y = np.array(EV_df[['Slug']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#Neural Network Model Exit Velocity
nn = MLPRegressor(max_iter=500).fit(X_train, y_train)
pred = nn.predict(X_test)
#calculate mse
sum = 0
for i in range (0, len(y_test)):
    diff = y_test[i] - pred[i]
    sq_diff = diff ** 2
    sum += sq_diff
EVnn_mse = sum/len(y_test)
print("Neural Network Exit Velocity MSE:", EVnn_mse)
print("Neural Network Exit Velocity R2 Score:", (r2_score(pred, y_test)))