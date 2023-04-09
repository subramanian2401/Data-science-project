import streamlit as st
import seaborn as sns
import pandas as pd

# Load dataset
data = pd.read_csv(r'C:\Users\subra\Documents\4th semester\Mca 5964 Project\project process\sorted3.csv')

# Set up Streamlit app
st.title('Data Visualization with Seaborn')
st.write('Here are some visualizations of the dataset:')

# Scatter plot
st.subheader('Scatter Plot')
x_axis = st.selectbox('Select x-axis column', data.columns)
y_axis = st.selectbox('Select y-axis column', data.columns)
sns.scatterplot(data=data, x=x_axis, y=y_axis)
st.pyplot()

# Line plot
st.subheader('Line Plot')
x_axis = st.selectbox('Select x-axis column', data.columns)
y_axis = st.selectbox('Select y-axis column', data.columns)
sns.lineplot(data=data, x=x_axis, y=y_axis)
st.pyplot()

# Bar plot
st.subheader('Bar Plot')
x_axis = st.selectbox('Select x-axis column', data.columns)
y_axis = st.selectbox('Select y-axis column', data.columns)
sns.barplot(data=data, x=x_axis, y=y_axis)
st.pyplot()

# Heatmap
st.subheader('Heatmap')
sns.heatmap(data.corr(), annot=True)
st.pyplot()
