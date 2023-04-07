#importing required libraries

import pandas as pd # for data frames
import numpy as np # for array reshaping
import streamlit as st
import seaborn as sn # for plots
import matplotlib.pyplot as plt # for plots
from sklearn.model_selection import train_test_split # to do test & train data split
from sklearn.linear_model import LinearRegression # for linear regression
from sklearn import metrics # for the error metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score # for calculation the r squared value
import statsmodels.api as sm # Linear regression with statsmodels
from bokeh.transform import cumsum
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

# Creating data frame and ingesting the data from .csv file into the data frame
df = pd.read_csv(r'C:\Users\subra\Documents\4th semester\Mca 5964 Project\project process\sorted3.csv')

#shaping data

print(df.shape)

# Printing data types of each column
datatypes = df.dtypes
print(datatypes)

#Finding the Null values in the data set and summing up the data
print(df.isnull().sum())

# Descrebing the values by find the mean and count in the given data cells
print(df.describe().T)

#Displaying the data cells For a sample
print(df)

# Seaborn pairplot is used for visualizing the data of each rows with scatter and bar charts
sn.pairplot(df)

#Displaying the histogram charts for finding the relation between MRP and OutletSales
hist = df.hist(bins=5)
hist
plt.show()

# Checking for correlation between MRP and outlet sales

# Creating a correlation matrix from the data frame
corrMatrix = df.corr()

# Plotting using heatmap
sn.heatmap(corrMatrix, annot = True, cmap= 'PiYG')
plt.show()

# Correlation using kendall method
df.corr(method ='kendall')

# Before splitting we will be building two arrays x & y
# x array contains the data that we use to make predictions (MRP)
# y array contains the data that we will be predicting (Sales)
# In nutshell, y is dependent onx

x = df['ProductVisibility']
y = df['OutletSales']

# Now let us split the datasets x & y into train & test respectively.
# 0.2 represents 20% of data will be leveraged for training & 80% of data will be used for test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

# Since we have single feature we need to convert into a 2-D array with reshape (since model training requires 2-D array)
# This step is NOT required if you have more than one features for prediction

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)
x_train

# Linear Regression Model
model = LinearRegression()

# Training the model with the training data x (MRP) & y (Outletsales)
response= model.fit(x_train, y_train)

# Getting the coefficient
coeff = response.coef_

# Getting the intercept
intercept = response.intercept_

print ("The coefficient is: %d and the intercept is: %d"  %(coeff, intercept))

# Make predictions on the train data

y_pred1 = model.predict(x_train)
print(y_pred1)

# Make predictions on the test data

y_pred = model.predict(x_test)
print(y_pred)

# Evaluate the model on the testing data
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)

# Evaluate the model
from sklearn.metrics import r2_score, mean_squared_error
# calculate accuracy

print("R-squared score:", r2_score(y_train, y_pred1))
print("Mean squared error:", mean_squared_error(y_train, y_pred1))

# Make predictions on the test data

y_pred = model.predict(x_test)
print(y_pred)

# Evaluate the model
from sklearn.metrics import r2_score, mean_squared_error
# calculate accuracy

print("R-squared score:", r2_score(y_test, y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))

# predicing the responses (y, the outlet sales) based on the predictor (x, Mrp)
predictions = model.predict(x_test)
reg_line=[(model.coef_*x)+model.intercept_ for x in df['MRP']]



# We will be comparing with the predicted value with actual response (which was stored in y_test)
plt.scatter(y_test, predictions, color='y')
plt.show()











