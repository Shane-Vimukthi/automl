import streamlit as st
import streamlit.components.v1 as stc

from eda import prep_eda

# EDA Pkgs
import pandas as pd
import numpy as np
import neattext.functions as nfx
import os

# Plotting Pkgs
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image,ImageFilter,ImageEnhance

from utils import (
    HTML_BANNER,
    HTML_RANDOM_TEMPLATE
)


def main():
    stc.html(HTML_BANNER)
    st.title('Automated Machine Learning App for Supervised Tabular Datasets')
    st.subheader('Application User Interface')
    st.markdown("""
    	#### Description
    	+ This is a EDA Data Preprocessing Feature Engineering and Model Building of the Tabular Dataset where Supervised Learning Algorithms are considered depicting the various species built with Streamlit.

    	#### Purpose
    	+ Exploratory Data Analysis.
    	+ Data Preprocessing
    	+ Feature Engineering 
    	+ Hyperparameter Tunning
    	+ Model Building 
    	""")

    # sidebar controls and stuff
    st.sidebar.title('App Controls')
    st.sidebar.subheader('Please Select and Upload the Dataset')
    filename = st.sidebar.file_uploader('Browse the CSV file', type=('csv'))
    # df = pd.read_csv(filename)

    # To Improve speed and cache data
    @st.cache(persist=True)
    def explore_data(file):
        df = pd.read_csv(file)
        return df
    df = explore_data(filename)

    colls = df.columns
    st.sidebar.subheader('Select the Target Variable')
    target_variable = st.sidebar.selectbox('Find the Target Column by all the Columns of the Dataset given below', colls)

    # ml type selection
    ml_types = ['Regression', 'Multi-Label Classification', 'Binary Classification']
    st.sidebar.subheader('Select The Machine Learning Model Type')
    target_type = st.sidebar.selectbox('Find the Target Column type which will Represent the Machine Learning Type of your Models',ml_types)
    # print(target_type)

    # Select the models based on the target machine learning type
    ml_reg_models = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor']
    ml_mclass_models = ['Decision Tree Classifier', 'Random Forest Classifier']
    ml_binary_models = ['Logistic Regression']

    st.sidebar.subheader('Select the Machine Learning Models')
    # print(target_type[0])
    # print(target_type[1])

    #model type selection
    if target_type == "Regression":
        target_model = st.sidebar.multiselect('Please select the Model', ml_reg_models)
    elif target_type == str('Multi-Label Classification'):
        target_model = st.sidebar.multiselect('Please Select the Model', ml_mclass_models)
    else:
        target_model = st.sidebar.multiselect('Please Select the Model', ml_binary_models)
    # print(target_model)


    # task selection
    # menu = ['EDA Only', 'Models Only', 'EDA&Models']
    # choice = st.sidebar.selectbox()

    st.sidebar.subheader('Select the Task You want to Perform')
    if st.sidebar.button('EDA Only'):
        st.title('We are doing the EDA')
        prep_eda(df, target_variable, target_type)



    if st.sidebar.button('Models Only'):
        st.title('We are Building Models Only no EDA')
    if st.sidebar.button('EDA&Models'):
        st.write('We are Building Models and EDA')








if __name__ == "__main__":
    main()