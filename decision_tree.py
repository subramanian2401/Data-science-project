# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the sales data
data = pd.read_csv(r'C:\Users\subra\Documents\4th semester\Mca 5964 Project\project process\sorted3.csv')

# Create the feature matrix and target variable
X = data.drop('MRP', axis=1)
y = data['OutletSales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree regression model
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Display the results in Streamlit
st.write("Sales Data")
st.write(data)

st.write("Model Performance")
st.write("Mean Squared Error:", mse)

# Create a form to input new data for sales forecasting
st.write("Sales Forecasting")
form = st.form(key='input_form')
inputs = []

for column in X.columns:
    input_value = form.number_input(label=column, step=1, value=0)
    inputs.append(input_value)

submit_button = form.form_submit_button(label='Submit')

# Make a sales forecast based on the input data
if submit_button:
    new_data = pd.DataFrame([inputs], columns=X.columns)
    forecast = model.predict(new_data)
    st.write("Sales Forecast:", forecast[0])
