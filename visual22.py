import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.markdown(
    """
    <h1 style='text-align: center;'>Sales Forcasting</h1>
    """,
    unsafe_allow_html=True
)
# Add a browse file button to upload a CSV file
uploaded_file = st.file_uploader(r'Upload File', type="csv")

# If a file was uploaded, read the data into a Pandas DataFrame
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Add a dropdown menu to select a column to plot
    x_axis = st.selectbox("Select a column to plot on the x-axis", data.columns)
    y_axis = st.selectbox("Select a column to plot on the y-axis", data.columns)

    import numpy as np



    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the data with a color map
    sc = ax.scatter(data[x_axis], data[y_axis], c='r', cmap=plt.cm.jet)

    # Add a color bar
    cbar = fig.colorbar(sc)

    # Set the chart title and labels
    ax.set_title('Sine Wave')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    #Create a scatter plot
    plt.scatter(data[x_axis], data[y_axis],color='#b3f542')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(fig)
