import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import PIL




st.write("""
# Simple Iris Flower Prediction Application testing
This application predicts the **Iris Flower** from the given data
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal width', 0.1, 2.5, 0.2)

    data = {'sepal_lenght': sepal_length,
             'sepal_width': sepal_width,
             'petal_length': petal_length,
             'petal_width': petal_width}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

st.subheader('Iris Dataset')
# a = [[12,334,651,62,2,1.45, 154,145,62,25], ['dasdf','af','afd','agwer','a','rg','ag','agr','a','ra'], [1223,34,15,445,245,73,356,123,14,77]]
# # df_iris = pd.DataFrame(iris)
# st.dataframe(np.array(a),width=500, height=600)
data_iris = pd.read_csv('data/Iris.csv')
st.dataframe(data_iris)


clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class Labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write("""
# Data Visualization 
""")
st.subheader('Line Chart')
st.line_chart(X)
st.subheader('Area Chart')
st.area_chart(X)
st.subheader('Bar Chart')
st.bar_chart(X)

st.subheader('Using matplotlib')
fig, ax = plt.subplots()
plt.title('Scatter plot')
plt.scatter(data_iris['SepalLengthCm'], data_iris['SepalWidthCm'])
st.pyplot(fig)

st.subheader('Using Altair')
chart = alt.Chart(data_iris).mark_circle().encode(x = 'SepalLengthCm', y= 'SepalWidthCm'
                                                  , tooltip= ['SepalLengthCm', 'SepalWidthCm'])
st.altair_chart(chart, use_container_width=True)

st.subheader('Using Plotly and maps')
st.map()
# st.bokeh_chart
# st.plotly_chart

st.subheader('Sample Image')
st.image('data/sample.jpg')
st.video('https://www.youtube.com/watch?v=jq0lKFb-P8k&list=PLuU3eVwK0I9PT48ZBYAHdKPFazhXg76h5&index=4&t=678s&ab_channel=HarshGupta')








st.write("""
# Testing the streamlit
Below this are just testing the codes
""")

@st.cache
def ret_time():
    time.sleep(5)
    return time.time()

if st.checkbox("1"):
    st.write(ret_time())

if st.checkbox("2"):
    st.write(ret_time())


st.write("""
# using widgets
Below this are just testing the codes
""")

if st.button('Subscribe'):
    st.write('Thanks friend')

name = st.text_input('Please Enter the Name')
st.write('Hi ', name)

address = st.text_area('Please Enter the Address')
st.write(address)

st.date_input('Enter the Date')
st.time_input('Enter the Time')

st.number_input('Numbers', min_value=3, max_value=300, value=34, step=3)

if st.checkbox('You accept the T&C', value=False):
    st.write('Thank You')

v1 = st.radio('Colours', ['r', 'b', 'g'], index=2)
v2 = st.selectbox('Colours', ['r', 'b', 'g'], index=1)

st.write(v1, v2)

v3 = st.multiselect('Colors', ['r', 'g', 'b'])
st.write(v3)

img = st.file_uploader('Upload a Image')
st.image(img)


if st.button('Select Messages'):

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.1)
        progress.progress(i+1)

    st.balloons()

    st.error('Error')
    st.success('Sucessfully completed')
    st.info('Information')
    st.exception(RuntimeError('This is an exception for Runtime Error'))
    st.warning('This is a warning')


st.subheader('Adding a Layout to Streamlit')

st.title('Registration Form')
first, last = st.beta_columns(2)
first.text_input('First Name: ')
last.text_input('Last Name: ')
