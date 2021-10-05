import streamlit as st
import os
import pandas as pd


st.set_page_config(page_title='Automl Application for Tabular Supervised Data', layout='wide')
st.write("""
# Automl Application for Tabular Supervised Data
This application predicts any Tabular Supervised learning
 based machine learning from the given data
""")

st.sidebar.header('Please Select and Upload the Dataset')

def user_selections():

    # filename = st.text_input('Enter a File Path: ')
    # try:
    #     with open(filename) as input:
    #         st.text(input.read())
    # except FileNotFoundError:
    #     st.error('File not found')

    filename = st.sidebar.file_uploader('Upload the Tabular Dataset', type=('csv'))
    df = pd.read_csv(filename)

    colls = df.columns
    target_variable = st.sidebar.multiselect('Please Select the Target Variable', colls)

    ml_types = ['Regression', 'Multi-Label Classification', 'Binary Classification']
    target_type = st.sidebar.multiselect('Please Select the Target Variable', ml_types)
    print(target_type)

    ml_reg_models = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor']
    ml_mclass_models = ['Decision Tree Classifier', 'Random Forest Classifier']
    ml_binary_models = ['Logistic Regression']

    if target_type[0] == "Regression" :
        target_model = st.sidebar.multiselect('Please select the Model', ml_reg_models)
    elif target_type[0] == str('Multi-Label Classification'):
        target_model = st.sidebar.multiselect('Please Select the Model', ml_mclass_models)
    else:
        target_model = st.sidebar.multiselect('Please Select the Model', ml_binary_models)

    if st.sidebar.button('EDA'):
        st.write('eda')
    if st.sidebar.button('Models'):
        st.write('Models')
    if st.sidebar.button('Both'):
        st.write('both')


    return df, colls, target_variable, target_type, target_model

df, colls, target_variable, target_type, target_model = user_selections()
# st.sidebar.subheader('User Input Parameters')
# st.write(target_variable)
# st.write(target_type)
# st.write(target_model)
# print(target_type)
# print(target_variable)
# print(target_model)




