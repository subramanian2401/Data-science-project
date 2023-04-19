import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import streamlit as st
import plotly.express as px
import numpy as np


# Creating data frame and ingesting the data from .csv file into the data frame

uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Select x and y-axis variables using selectbox
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist() # Get numeric columns
    x_axis = st.sidebar.selectbox("Select x-axis variable", numeric_cols)
    y_axis = st.sidebar.selectbox("Select y-axis variable", numeric_cols)

    # Create a scatter plot
    fig = px.scatter(data, x=x_axis, y=y_axis)
    st.plotly_chart(fig)

    # Split the data into training and testing sets
    X = data.drop(y_axis, axis=1)
    y = data[y_axis]
    test_size = st.sidebar.slider("Select test size ratio", 0.1, 0.5, value=0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    # Fit a decision tree regressor to the training data
    dt = DecisionTreeRegressor(random_state=0)
    dt.fit(X_train, y_train)
    st.write(X_train.shape, X_test.shape)

    # Make predictions on the train data
    y_pred1 = dt.predict(X_train)

    # Make predictions on the test data
    y_pred = dt.predict(X_test)

    # Evaluate the model on the testing data
    mse = mean_squared_error(y_test, y_pred)
    st.write('MSE:', mse)

    # Predict on new data and display the results
    new_data = X_test.sample(n=5, random_state=0)  # Choose 5 random samples from the test data
    new_pred = dt.predict(new_data)  # Make predictions on the new data
    st.write('Predicted values for new data:')
    st.write(pd.DataFrame({'Actual': y_test[new_data.index], 'Predicted': new_pred}))

    # Evaluate the model
    from sklearn.metrics import r2_score, mean_squared_error

    # calculate accuracy on train data
    st.write("R-squared score on train data:", r2_score(y_train, y_pred1))
    st.write("Mean squared error on train data:", mean_squared_error(y_train, y_pred1))

    # Evaluate the model
    from sklearn.metrics import r2_score, mean_squared_error

    # calculate accuracy on test data
    st.write("R-squared score on test data:", r2_score(y_test, y_pred))
    st.write("Mean squared error on test data:", mean_squared_error(y_test, y_pred))

    import plotly.graph_objects as go

    # Create the histogram trace for predicted values
    trace_pred = go.Histogram(
        x=y_pred,
        name='Predicted Values',
        marker=dict(color='#636EFA'),
        opacity=0.75
    )

    # Create the histogram trace for actual values
    trace_actual = go.Histogram(
        x=y_test,
        name='Actual Values',
        marker=dict(color='#EF553B'),
        opacity=0.75
    )

    # Create the layout for the histogram
    layout = go.Layout(
        title=dict(
            text='Product Outlet Sales Summary',
            font=dict(size=20, color='#444444', family='Arial, sans-serif'),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title=dict(
                text='Outlet Sales',
                font=dict(size=15, color='#444444', family='Arial, sans-serif'),
            ),
            showgrid=True,
            zeroline=False,
            showticklabels=True,
            ticks='outside',
            tickangle=0,
            tickfont=dict(size=12, color='#444444')
        ),
        yaxis=dict(
            title=dict(
                text='Sales Count',
                font=dict(size=15, color='#444444', family='Arial, sans-serif'),
            ),
            showgrid=True,
            zeroline=False,
            showticklabels=True,
            ticks='outside',
            tickangle=0,
            tickfont=dict(size=12, color='#444444')
        ),
        bargap=0.2,
        bargroupgap=0.1,
        hovermode='closest'
    )

    # Add the traces to the figure
    fig = go.Figure(data=[trace_pred, trace_actual], layout=layout)

    # Display the figure
    st.plotly_chart(fig)

