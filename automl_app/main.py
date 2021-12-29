import streamlit as st
import streamlit.components.v1 as stc

from eda import prep_eda
from dprep import prep_data
from models_simple import model_lazy_pred
from models_classifiers import classifiers_models
from models_regression import regressor_models
from models_logistic import logistic_reg_models

# EDA Pkgs
import pandas as pd
import numpy as np
import neattext.functions as nfx
import os
import shortuuid

# # Plotting Pkgs
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image,ImageFilter,ImageEnhance

from utils import (
    HTML_BANNER,
    HTML_RANDOM_TEMPLATE
)

import plotly.express as px
import GPUtil


# @st.cache(persist=True, allow_output_mutation=False)
# @st.cache(allow_output_mutation=False)
def explore_data(file):
    df = pd.read_csv(file)
    return df

def convert_df(df):
    # return df.to_csv('final_processed_df.csv', index=False)
    return df.to_csv().encode('utf-8')


def create_folders(file_name):
    parent_directory = 'outputs/'

    unique_name = shortuuid.ShortUUID().random(length=5)

    head, tail = os.path.split(file_name.name)
    tail_folder = tail.split('.')[0]

    tail_folder = str(tail_folder) + '_' + str(unique_name)
    data_folder = 'data'


    path_folder = os.path.join(parent_directory, tail_folder)
    path_data = os.path.join(path_folder, data_folder)


    try:
        os.mkdir(path_folder)
        os.mkdir(path_data)
        st.write('Your Folder name is ', tail_folder)
        st.success('File Saved Successfully')
    except OSError as error:
        st.error(error)




def main():
    st.set_page_config(layout='wide')
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
    
    ## Please Select a Dataset From Your Directory and Follow through the Steps in the Sidebar
    
    """
    )



    # sidebar controls and stuff
    st.sidebar.title('App Controls')
    st.sidebar.subheader('Please Select and Upload the Dataset')
    filename = st.sidebar.file_uploader('Browse the CSV file', type=('csv'))
    print(filename)
    # df = pd.read_csv(filename)

    # To Improve speed and cache data
    # @st.cache(persist=True)
    # def explore_data(file):
    #     df = pd.read_csv(file)
    #     return df

    #check if only file path is given
    if filename:
        df = explore_data(filename)
        colls = df.columns
        st.sidebar.subheader('Select the Target Variable')
        target_variable = st.sidebar.selectbox('Find the Target Column by all the Columns of the Dataset given below', colls)

        # ml type selection
        ml_types = ['Regression', 'Multi-Label Classification', 'Binary Classification']
        st.sidebar.subheader('Select The Machine Learning Model Type')
        target_type = st.sidebar.selectbox('Find the Target Column type which will Represent the Machine Learning Type of your Models',ml_types)
        # print(target_type)
        model_approach = ['Beginner', 'Professional']
        # Select the models based on the target machine learning type
        ml_reg_models = ['Select', 'RandomForestRegressor', 'ExtraTreesRegressor', 'LightGBMRegressor']
        ml_mclass_models = ['Select', 'RandomForestClassifier', 'ExtraTreesClassifier', 'LightGBMClassifier']
        ml_binary_models = ['Select','LogisticRegression']

        st.sidebar.subheader('Select the Machine Learning Models Approach')
        model_approach_selcted = st.sidebar.selectbox('Please select the Model Approach', model_approach)
        # print(target_type[0])
        # print(target_type[1])
        lazy_pred_models = ['Select','Try All Models']
        #model type selection
        # print(target_model)


        # task selection
        st.sidebar.subheader('Select the Task You want to Perform')
        menu = ['Select', 'EDA Only', 'Models Only']
        choice = st.sidebar.selectbox("Please Select the Task You Wanna Perform on Dataset", menu,0) # default_value= index 0)

        # if st.sidebar.button('EDA Only'):
        if choice =='EDA Only':
            st.title('We are doing the EDA')
            prep_eda(df, target_variable, target_type)



        # if st.sidebar.button('Models Only'):
        if choice == 'Models Only':
            st.title('We are Building Models Only no EDA')
            df = prep_data(df, target_variable)
            # st.title('Building the Models')
            # st.write(ml_binary_models,' ', ml_reg_models,' ', ml_mclass_models)
            # st.write(target_type)
            # st.write(target_model)
            # st.dataframe(df.head())
            view_final_dataset = st.checkbox('View The Final Dataset', value=False)
            if view_final_dataset:
                # st.write('Customized DataFrame')
                # df = df[customized_cols]
                st.dataframe(df.head())

            st.title('Building the Models')
            st.subheader('Please select your Options below to build Models')

            if target_type == "Regression" and model_approach_selcted == 'Beginner':
                target_model = st.selectbox('Please select the Model', lazy_pred_models)
                if target_model == 'Try All Models':

                    st.subheader('All Regression Models with Scores')
                    models,predictions = model_lazy_pred(df, target_variable,target_type, target_model)
                    st.write(models)

                    st.subheader('Histogram for Scores Models')
                    models["R-Squared"] = [0 if i < 0 else i for i in models.iloc[:, 0]]
                    fig1 = px.bar(models, x='R-Squared', y=models.index, orientation='h')
                    st.plotly_chart(fig1)

                    st.subheader('Histogram for Time Taken Models')
                    fig1 = px.bar(models, x='Time Taken', y=models.index, orientation='h')
                    st.plotly_chart(fig1)



            elif target_type == 'Regression' and model_approach_selcted == 'Professional':
                target_model = st.selectbox("Please select the Model", ml_reg_models)
                # if target_model == 'Desicion Tree Regressor':
                st.subheader('Hyperparameter Tunning to Improve the Model Performance')

                if target_model == 'RandomForestRegressor':
                    st.subheader('Please select the Variables for Hyperparameter Tunning for RandomForest Regressor')
                    model = regressor_models(df, target_variable, target_model, target_type)


                elif target_model == 'ExtraTreesRegressor':
                    st.subheader('Please select the Variables for Hyperparameter Tunning for ExtraTrees Regressor')
                    model = regressor_models(df, target_variable, target_model, target_type)

                elif target_model == 'LightGBMRegressor':
                    st.subheader('Please select the Variables for Hyperparameter Tunning for LightGBM Regressor')
                    model = regressor_models(df, target_variable, target_model, target_type)



            if target_type == str('Multi-Label Classification') and model_approach_selcted == 'Beginner':
                target_model = st.selectbox('Please select the Model', lazy_pred_models)
                if target_model == 'Try All Models':
                    st.subheader('All Classification Models With Scores')
                    modelss, predictions = model_lazy_pred(df, target_variable, target_type)

                    st.write(modelss)
                    st.subheader('Histogram for Scores Models')
                    modelss["Accuracy"] = [0 if i < 0 else i for i in modelss.iloc[:, 0]]
                    fig1 = px.bar(modelss, x='Accuracy', y=modelss.index, orientation='h')
                    st.plotly_chart(fig1)

                    st.subheader('Histogram for Time Taken Models')
                    fig1 = px.bar(modelss, x='Time Taken', y=modelss.index, orientation='h')
                    st.plotly_chart(fig1)


            elif target_type == str('Multi-Label Classification') and model_approach_selcted == 'Professional':
                target_model = st.selectbox('Please Select the Model', ml_mclass_models)

                st.subheader('Hyperparameter Tunning to Improve the Model Performance')

                if target_model == 'RandomForestClassifier':
                    st.subheader('Please select the Variables for Hyperparameter Tunning for RandomForestClassifier')
                    model = classifiers_models(df, target_variable, target_model, target_type)


                elif target_model == 'ExtraTreesClassifier':
                    st.subheader('Please select the Variables for Hyperparameter Tunning for ExtraTreesClassifier')
                    model = classifiers_models(df, target_variable, target_model, target_type)

                elif target_model == 'LightGBMClassifier':
                    st.subheader('Please select the Variables for Hyperparameter Tunning for LightGBM Classifier')
                    model = classifiers_models(df, target_variable, target_model, target_type)





            elif target_type == str('Binary Classification') and model_approach_selcted == 'Beginner':
                target_model = st.selectbox('Please Select the Model', lazy_pred_models)
                if target_model == 'Try All Models':

                    st.subheader('All Classification/Binary Models with Scores')
                    models, predictions = model_lazy_pred(df, target_variable,target_type, target_model)
                    st.write(models)

                    st.subheader('Histogram for Scores Models')
                    models["Accuracy"] = [0 if i < 0 else i for i in models.iloc[:, 0]]
                    fig1 = px.bar(models, x='Accuracy', y=models.index, orientation='h')
                    st.plotly_chart(fig1)

                    st.subheader('Histogram for Time Taken Models')
                    fig1 = px.bar(models, x='Time Taken', y=models.index, orientation='h')
                    st.plotly_chart(fig1)


            elif target_type == str('Binary Classification') and model_approach_selcted == 'Professional':
                target_model = st.selectbox('Please select the Model', ml_binary_models)

                st.subheader('Hyperparameter Tunning to Improve the Model Performance')

                if target_model == 'LogisticRegression':
                    st.subheader('Please select the Variables for Hyperparameter Tunning for LogisticRegression')
                    model = logistic_reg_models(df, target_variable, target_model, target_type)





            st.download_button(label="Download dataset as CSV", data=convert_df(df), file_name='final_df.csv', mime='text/csv')
                # st.download_button(label= 'Download the Model as pickle', data = model_out, file_name='final_model.pkl')







if __name__ == "__main__":
    main()