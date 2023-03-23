#!/usr/bin/env python
# coding: utf-8

# # Support vector Regression

# Support Vector Regression (SVR) is a machine learning algorithm used for regression analysis. Unlike traditional regression techniques, SVR uses a hyperplane to identify the best possible fit for the data.
# 
# In SVR, the aim is to find a hyperplane that has the maximum margin on either side of the hyperplane. The margin is the distance between the hyperplane and the closest data points. The hyperplane is used to make predictions on new data points.
# 
# The hyperplane is determined by finding the support vectors, which are the data points closest to the hyperplane. These support vectors are used to define the hyperplane and determine the margin.
# 
# The SVR algorithm can handle non-linear relationships between the input variables and the output variable by using a kernel function to transform the input variables into a higher dimensional space. This allows the SVR algorithm to find a hyperplane in the transformed space that can capture the non-linear relationship.
# 
# SVR is a powerful algorithm for regression analysis and is commonly used in many applications, including finance, economics, and engineering.

# In[19]:


#importing required libraries

import pandas as pd # for data frames
import numpy as np # for array reshaping

import seaborn as sn # for plots
import matplotlib.pyplot as plt # for plots
from sklearn.model_selection import train_test_split # to do test & train data split
from sklearn.linear_model import LinearRegression # for linear regression
from sklearn import metrics # for the error metrics
from sklearn.metrics import r2_score # for calculation the r squared value
import statsmodels.api as sm # Linear regression with statsmodels
from sklearn.preprocessing import StandardScaler
from bokeh.transform import cumsum
from sklearn.svm import SVR
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


# Load the data
df = pd.read_csv(r'C:\Users\subra\Documents\4th semester\Mca 5964 Project\project process\sorted3.csv')


# In[21]:


#Shaping the data set
df.shape


# In[22]:


# Printing data types of each column
datatypes = df.dtypes
datatypes


# In[23]:


#Finding the Null values in the data set and summing up the data 
df.isnull().sum()


# In[24]:


# Descrebing the values by fing the mean and count in the given data cells
df.describe().T


# In[25]:


# Seaborn pairplot is used for visualizing the data of each rows with scatter and bar charts
sn.pairplot(df)


# In[26]:


#Displaying the histogram charts for finding the relation between MRP and OutletSales 
hist = df.hist(bins=5)
hist


# In[27]:


# Checking for correlation between MRP and outlet sales

# Creating a correlation matrix from the data frame
corrMatrix = df.corr()

# Plotting using heatmap
sn.heatmap(corrMatrix, annot = True, cmap= 'PiYG')


# In[28]:


# Split the data into features and target
X = df.drop("ProductVisibility", axis=1)
y = df["OutletSales"]


# In[29]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Scaling

# Support Vector Regression is based on the distance between data points and the hyperplane. If the data points are not scaled, the distance calculations may not reflect the true distance between the points. This can lead to incorrect predictions and a poorly performing model.
# 
# Scaling is an important step in SVR preprocessing to ensure the model's accuracy and performance. It ensures that each feature contributes equally to the model and that the distance calculations accurately reflect the true distance between data points.

# In[30]:


# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[31]:


# Create the SVR model
regressor = SVR(kernel='rbf')


# In[32]:


# Train the model
regressor.fit(X_train, y_train)


# In[33]:


# Predict on the test set
y_pred = regressor.predict(X_test)


# In[34]:


# Evaluate the model
from sklearn.metrics import r2_score, mean_squared_error
print("R-squared score:", r2_score(y_test, y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))


# In[35]:


df1=df.head(10)
plt.bar(y_test,y_pred, color='r')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




