import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add a browse file button to upload a CSV file
uploaded_file = st.file_uploader(r'C:\Users\subra\Documents\4th semester\Mca 5964 Project\project process\sorted3.csv', type="csv")

# If a file was uploaded, read the data into a Pandas DataFrame
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Add a dropdown menu to select a column to plot
    x_axis = st.selectbox("Select a column to plot on the x-axis", data.columns)
    y_axis = st.selectbox("Select a column to plot on the y-axis", data.columns)

    # Create a scatter plot
    plt.scatter(data[x_axis], data[y_axis])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot()
