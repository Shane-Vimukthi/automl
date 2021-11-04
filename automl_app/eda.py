import streamlit as st
import streamlit.components.v1 as stc

# EDA Pkgs
import pandas as pd
import numpy as np
import neattext.functions as nfx
import os
import io

# Plotting Pkgs
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image,ImageFilter,ImageEnhance
import plotly_express as px





def prep_eda(df, target_variable, target_type):

    # show the dataset
    st.subheader('View the Data Frame')
    # st.dataframe(df.head(250))
    if df.shape[0]>250:
        st.dataframe(df.sample(n=250))
    else:
        st.dataframe(df)

    # view the head and tail of the dataset
    #read the head and tail of the dataset
    st.subheader('\n\nSelect To View Head or Tail')
    if st.checkbox('View Head'):
        st.write(df.head())
    if st.checkbox('View Tail'):
        st.write(df.tail())

    # Get the shape of the dataset
    st.subheader('\n\n Rows and Columns Count of the Dataset')
    st.write('Number of Columns : ', df.shape[1])
    st.write('Number of Rows : ', df.shape[0])

    # Get the info of the dataset
    st.subheader('\n\n Dataset Information')
    # st.write(df.info(verbose=False))
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    #write the info to a txt file
    with open("df_info.txt", "w",encoding="utf-8") as f:
        f.write(s)
        # st.write(s)
        f.close()
    # read the file skip some lines
    txt_info = open("df_info.txt", "r")
    word1 = "#"
    word2 = "--"
    word3 = "<class"
    for line in txt_info:
        if word3 in line:
            continue
        elif word1 in line:
            continue
        elif word2 in line:
            continue
        else:
            st.write(line)
    txt_info.close()

    # Get the description of the dataset
    st.subheader('\n\n Dataset Description')
    st.write(df.describe())

    # Get the null list
    st.subheader('\n\n Dataset Null Availability ColumnWise')
    null_df = pd.DataFrame(df.isnull().sum(), columns=['null_count'])
    # null_df.head(15) add the null percentage too
    st.dataframe(null_df)

    # Get the null Graph
    st.subheader('\n\n Dataset Null Heatmap')
    fig = plt.figure()
    sns.heatmap(null_df, cmap='coolwarm')
    st.pyplot(fig)

    st.subheader('\n\n Target Column Properties')
    col1_target, col2_target, col3_target = st.columns([2, 2, 3])
    col1_target.subheader('\n\n Target Column')
    col1_target.dataframe(df[target_variable])
    col2_target.subheader('\n\n Unique Values')
    col2_target.write(df[target_variable].unique())
    col3_target.subheader('\n\n Target Variable Details')
    col3_target.text('\n\n Target Variable DataType:')
    col3_target.write(df[target_variable].dtypes)
    col3_target.text('\n\n Target Variable Unique count:')
    col3_target.write(df[target_variable].nunique())

    # plot some graphs of target col
    st.subheader('\n\n Target Column Plot')
    if df[target_variable].dtypes != 'object':
        fig, axes = plt.subplots(1, 3, figsize=(25, 8))

        sns.histplot(x=target_variable, data=df, ax=axes[0], bins=50)
        sns.kdeplot(x=target_variable, data=df, ax=axes[1], fill=True)
        sns.boxplot(data=df, ax=axes[2], x=target_variable)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        # Categorical target Variable
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        # fig = plt.figure()
        df[target_variable].value_counts()[::-1].plot(kind='pie', title=target_variable, autopct='%.0f', fontsize=10, ax=axes[0])
        # st.pyplot(fig)
        # fig = plt.figure()
        sns.histplot(x=target_variable, data=df, bins=50, ax=axes[1])
        st.pyplot(fig)

    # Statistical Plots
    # corr plot
    st.subheader('\n\n Dataset Corelation')
    fig = plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap='PiYG')
    st.pyplot(fig)

    # highly corelated columns
    # get the corelations morethan 0.5
    cormat = df.corr()
    # get the most corelated features
    st.subheader('\n\n Highly Corelated Features with Target Variable')
    def get_corr_features(corrdata, threshold):
        feature = []
        value = []
        for i, index in enumerate(corrdata.index):
            if abs(corrdata[index]) > threshold:
                feature.append(index)
                value.append(corrdata[index])
        df_cor = pd.DataFrame(data=value, index=feature, columns=['corr_value'])

        return df_cor
    threshold_cor = st.slider('Threshold For Corelation', 0.0, 1.0, 0.2)
    cor_cols = df.select_dtypes(exclude='object').columns.tolist()
    cor_col = st.selectbox('Select the Colum to test the corelation', cor_cols)

    corr_df = get_corr_features(cormat[cor_col], threshold_cor)
    st.write(corr_df)
    st.caption('*If the Target Variable is correlated with itself please ignore')

    # plot the datset
    # st.subheader('\n\n Pairplot For all Numerical Features')
    # fig = sns.pairplot(df, hue= target_variable)
    # st.pyplot(fig)


    # line and area plot of the all numericals
    st.subheader('\n\n Numerical Columns Line Chart')
    st.line_chart(df[cor_cols])

    # st.subheader('\n\n Numerical Columns Bar Chart')
    # st.bar_chart(df[cor_cols])

    st.subheader('\n\n Numerical Columns Area Chart')
    st.area_chart(df[cor_cols])


    # -------box hist kde plot for the numerical data use plotly express-------------------------------------------------
    # pairplot with plotly
    # st.subheader('Plotly Pair Plot')
    # pair_plt = px.scatter_matrix(df, color=target_variable)
    # st.plotly_chart(pair_plt)

    # make selection for all numeric columns with 3 types of plots
    st.subheader('Plot for each numerical column')
    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    # Plot the 3 plots for each num col
    for k in num_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        sns.histplot(x=k, data=df, ax=axes[0], bins=50)
        # px.histogram(df,x=k)
        sns.kdeplot(x=k, data=df, ax=axes[1], fill=True)
        sns.boxplot(data=df, ax=axes[2], x=k)
        plt.tight_layout()
        st.pyplot(fig)
        # st.plotly_chart(fig)

    # make selection for all categorical columns with 2 plots
    # Categorical Variable check
    st.subheader('Plot for each categorical column')
    for i in cat_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        df[i].value_counts()[::-1].plot(kind='pie', ax=axes[0], title=i, autopct='%.0f', fontsize=10)
        sns.countplot(x=i, data=df, ax=axes[1])
        st.pyplot(fig)


    st.subheader('\n\n Plotly Customized Plots')
