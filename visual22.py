import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
import seaborn as sns

st.markdown(
    """
    <h1 style='text-align: center;'>Sales Forcasting</h1>
    """,
    unsafe_allow_html=True
)
# Add a browse file button to upload a CSV file
uploaded_file = st.file_uploader(r'upload file', type="csv")

# If a file was uploaded, read the data into a Pandas DataFrame
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # pairplot=sns.pairplot(data)
    # st.pyplot(pairplot.fig)

    st.markdown(
        """
        <h1 style='text-align: center;'>Corelation Heat Chart</h1>
        """,
        unsafe_allow_html=True
    )
    #heat map
    # Create a heatmap
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='Greens')
    heatmap_fig = plt.gcf()
    st.pyplot(heatmap_fig)
    st.markdown(
        """
        <h1 style='text-align: center;'>Selecting maximum threshold </h1>
        """,
        unsafe_allow_html=True
    )
    st.write("The choice of a threshold for highly correlated columns depends on the specific problem and the nature of the data. Typically, a threshold of 0.5 or 0.7 is used to identify highly correlated features. This means that features with a correlation coefficient of 0.7 or above (or 0.8 or above) are considered highly correlated and one of them may be removed from the dataset.")

    # Select the threshold for high correlation
    threshold = st.slider("Select the threshold for high correlation", min_value=0.0, max_value=1.0, value=0.5)

    # Find the highly correlated columns
    st.write("Highly correlated columns:")
    high_corr = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                colname = corr.columns[i]
                high_corr.add(colname)
                st.write(colname)

    # Find the not highly correlated columns
    st.write("Not highly correlated columns:")
    not_high_corr = set(data.columns) - high_corr
    for colname in not_high_corr:
        st.write(colname)

    st.write("THE DATA CONSISTS OF(ROWS,COLUMNS):", data.shape)
    st.write("Describing data", data.describe())

    # Add a dropdown menu to select a column to plot
    x_axis = st.selectbox("Select a column to plot on the x-axis", data.columns)
    y_axis = st.selectbox("Select a column to plot on the y-axis", data.columns)





    st.markdown(
        """
        <h1 style='text-align: center;'>Scatter plot</h1>
        """,
        unsafe_allow_html=True
    )

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the data with a color map
    sc = ax.scatter(data[x_axis], data[y_axis], c='r', cmap=plt.cm.jet)

    # Add a color bar
    cbar = fig.colorbar(sc)

    # Set the chart title and labels
    ax.set_title('Vizualized_data')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    #Create a scatter plot
    plt.scatter(data[x_axis], data[y_axis],color='#b3f542')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(fig)




















