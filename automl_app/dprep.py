import random
import warnings

import streamlit as st
import streamlit.components.v1 as stc

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import time
from st_aggrid import AgGrid
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
import itertools as it
plt.style.use('seaborn-whitegrid')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import os
from collections import defaultdict


import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
import joblib

# from lightgbm import LGBMClassifier
# # from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_colwidth', -1)

import warnings
warnings.simplefilter('ignore')

from sklearn.model_selection import RandomizedSearchCV

# find the colum types if num/cat/target/id cols
def feat_types(df, target):
    # start_time = time.clock()

    id_cols = []
    # target_cols = []
    cat_cols_obj = []
    num_cols_all = []

    #     identify the target column and remove in this function
    target_cols = [target]
    df_original = df.copy()

    # df.drop([target], axis=1, inplace=True)  # remove the target column from the dataset

    #     do identify the target column and convert to num if categorical

    #     identify the ids and remove the ids
    cols_without_target = [i for i in df.columns if i not in target_cols]
    for col in cols_without_target:
        # print(col)
        perc_unique = df[col].nunique() / len(df[col]) * 100
        if perc_unique > 97.0:  # get the ids
            id_cols.append(col)
        elif df.dtypes[col] == 'object':  # get the string cat features
            cat_cols_obj.append(col)
        else:
            num_cols_all.append(col)  # get all the other int and float numericals

    ''' this cat variable identification is kept as a feature engineering method in the other modules
        for col in num_cols_all():
          if 10 > df_copy.col.nunique() > 2:
            num_cols_cat.append(col)
          else:
            num_cols_cont.append(col)'''

    # identify the cat cols in the numerical all columns in other words identify the discrete and continuous
    # num_cols_all = [i for i in df_copy.columns if i not in [target_col] and i not in id_cols and i not in cat_obj_cols]

    # print(time.clock() - start_time, "seconds took to finish dtype identifier")

    return df, id_cols, target_cols, cat_cols_obj, num_cols_all


# Highly corelated feature identification
# in here null values are not to be considered
def cor_identifier(df, num_cols_all):
    threshold = 0.9
    # get the corelations morethan 0.5
    cormat = df.corr()
    # get the most corelated features
    feature = []
    value = []

    for col in num_cols_all:
        corrdata = cormat[col]
        for i, index in enumerate(corrdata.index):
            if abs(corrdata[index]) > threshold:
                feature.append(index)
                value.append(corrdata[index])
        df_corr = pd.DataFrame(data=value, index=feature, columns=['corr_value'])

    return df_corr



# skewed numerical continuous columns identification
def skew_identifier(df, target_cols):
    # start_time = time.clock()
    df = df.drop(target_cols, 1)
    num_cols_skewed = []
    num_cols_norm = []
    df_skew = df.skew(axis =0, skipna=True)
    for col in df_skew.index:
        if abs(df_skew[col]) >1:
          num_cols_skewed.append(col)
        else:
          num_cols_norm.append(col)
      # print(time.clock() - start_time, "seconds took to finish skew identifier")
    return num_cols_skewed, num_cols_norm

# skewed columns fixing
# from scipy.stats import boxcox

def skew_fix(df, num_cols_skewed, num_cols_norm):
    # start_time = time.clock()
    num_cols_skewed_done = []

    # ln transformation
    for col in num_cols_skewed:
        new_col = str(col) + ':num_cols_skewed_log'
        df[new_col] = np.log(df[col] + 1)
        # add new colum to the new done list
        num_cols_skewed_done.append(new_col)
        # print(df.head())
    # squar root transformation
    for col in num_cols_skewed:
        new_col = str(col) + ':num_cols_skewed_sqr'
        df[new_col] = df[col] ** (1 / 2)
        num_cols_skewed_done.append(new_col)
        # print(df.head())

    num_cols_norm = num_cols_norm + num_cols_skewed_done

    # boxcox transformation
    # for col in num_cols_skewed:
    #   df[str(col)+'_num_cols_skewd'+'_boxcox'] = boxcox(df[col], lambda = None)
    # df = pd.DataFrame()
    # print(time.clock() - start_time, "seconds took to finish skew fix")
    return df, num_cols_skewed_done, num_cols_norm


# Outlier Identification and Treatment function
def outlier_identifier_fix(df, num_cols_norm, num_cols_skewed):
    # start_time = time.clock()
    num_cols_norm_outliers = []
    num_cols_skewed_outliers = []
    num_cols_norm_outliers_done = []
    num_cols_skewed_outliers_done = []
    # num_cols_skewed_done_outliers = [] skewed done means they are normal so add to the normal list
    # num_cols_skewed_done_outliers_done = []

    # outliers which follow the normal distribution
    for col in num_cols_norm:
        upper_limit_norm = df[col].mean() + 3 * df[col].std()
        lower_limit_norm = df[col].mean() - 3 * df[col].std()
        outliers_norm = df[df[col] > upper_limit_norm] + df[df[col] < lower_limit_norm]
        if len(outliers_norm) >= 1:
            num_cols_norm_outliers.append(col)
            # fixing the outliers norm   ----------------------------------------------------------- capping
            new_col = str(col) + ':num_cols_norm_outliers_cap'
            df[new_col] = np.where(df[col] > upper_limit_norm, upper_limit_norm,
                                   np.where(df[col] < lower_limit_norm, lower_limit_norm, df[col]))
            num_cols_norm_outliers_done.append(new_col)

        # outliers which are skewed
    for col in num_cols_skewed:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_limit_skew = q1 + 1.5 * iqr
        lower_limit_skew = q1 - 1.5 * iqr
        outliers_skew = len(df[df[col] > upper_limit_skew]) + len(df[df[col] < lower_limit_skew])
        if outliers_skew >= 1:
            num_cols_skewed_outliers.append(col)
            # fixing the outliers skewed   -------------------------------------------------------------- capping
            new_col = str(col) + ':num_cols_skewed_outliers_cap'
            df[new_col] = np.where(df[col] > upper_limit_skew, upper_limit_skew,
                                   np.where(df[col] < lower_limit_skew, lower_limit_skew, df[col]))
            num_cols_skewed_outliers_done.append(new_col)

    num_cols_norm = num_cols_norm + num_cols_norm_outliers_done + num_cols_skewed_outliers_done

    # print(time.clock() - start_time, "seconds took to finish outlier identifier and fix")

    return df, num_cols_skewed_outliers, num_cols_norm_outliers, num_cols_norm_outliers_done, num_cols_skewed_outliers_done, num_cols_norm


# at the end all the features which are null have different list regarding that speciality of the feature
# for egsample null columns which are normaly distributed are in 'num_cols_norm_null' where which are not null is in 'num_cols_norm'
# each and every feat considered null are removed from the previoustly available list

def null_identifier(df, cat_cols_obj, target_cols):
    # start_time = time.clock()
    cols_null_to_del = []
    cols_all_null = []
    cols_all_not_null = []

    for col in df.columns:
        if df[col].isnull().sum() / df.shape[0] * 100 > 60:
            cols_null_to_del.append(col)
        elif df[col].isnull().sum() / df.shape[0] * 100 != 0:
            cols_all_null.append(col)
        else:
            cols_all_not_null.append(col)

    cat_cols_obj_null = [i for i in cols_all_null if i in cat_cols_obj and i not in target_cols]
    cat_cols_obj_not_null = [i for i in cols_all_not_null if i in cat_cols_obj and i not in target_cols]
    num_cols_null = [i for i in cols_all_null if i not in cat_cols_obj and i not in target_cols]
    num_cols_not_null = [i for i in cols_all_not_null if i not in cat_cols_obj and i not in target_cols]

    # print(time.clock() - start_time, "seconds took to finish null identifier")

    return cols_null_to_del, cat_cols_obj_null, cat_cols_obj_not_null, num_cols_null, num_cols_not_null




def null_fix(df,cols_null_to_del, cat_cols_obj_null, num_cols_null):
  # fill null with extreme values numerical
  # fill null with 'null' category
  # make another column where null values are 0 and not null are 1
  # mean null filling numerical
  # median null filling numerical
  # mode null filling numerical
  # mode null filling categorical
  # filling null with feature combining krish method
  # start_time = time.clock()

  cat_cols_obj_null_done = []
  num_cols_null_done = []
  col_null_to_del_done = []


  # delete the features in the del list
  df_null_not_del = df.copy()
  for col in cols_null_to_del:
    # create another feature wher null cols available
    new_col = str(col)+':to_del_null_binary'
    df[new_col] = np.where(df[col].isnull(), 1,0)
    col_null_to_del_done.append(new_col)

    df.drop([col], axis=1, inplace=True) # delete all the null delete columns


  # null filling in categorical features
  for col in cat_cols_obj_null:

    # create another feature wher null cols available
    new_col = str(col)+':cat_cols_obj_null_binary'
    df[new_col] = np.where(df[col].isnull(), 1,0)
    cat_cols_obj_null_done.append(new_col)

    # filling the null values with mode of the category
    new_col = str(col)+':cat_cols_obj_null_mode'
    df[new_col] = df[col].fillna(df[col].mode())
    cat_cols_obj_null_done.append(new_col)

    # filling the null values with another category of the category
    new_col = str(col)+':cat_cols_obj_null_extreme'
    df[new_col] = df[col].fillna('null')
    cat_cols_obj_null_done.append(new_col)


  # null filling in numerical features
  for col in num_cols_null:
    # create another feature wher null cols available
    new_col = str(col)+':null_binary'
    df[new_col] = np.where(df[col].isnull(), 1,0)
    num_cols_null_done.append(new_col)

    # filling the null values with mode of the category
    new_col = str(col)+':null_mode'
    df[new_col] = df[col].fillna(df[col].mode())
    num_cols_null_done.append(new_col)

    # filling the null values with mean of the category
    new_col = str(col)+':null_mode'
    df[new_col] = df[col].fillna(df[col].mean())
    num_cols_null_done.append(new_col)

    # filling the null values with mean of the category
    new_col = str(col)+':median'
    df[new_col] = df[col].fillna(df[col].median())
    num_cols_null_done.append(new_col)

    # filling the null values with mean of the category
    new_col = str(col)+':extreme'
    max_val = df[col].max()
    df[new_col] = df[col].fillna(max_val+1000)
    num_cols_null_done.append(new_col)

    # special null fix more customized methods to be continued in further final sets
    # new_col = str(col)+'num_cols_cor_null_special'
    # max_val = df[col].max()
    # df[new_col] = df[col].fillna(max_val+1000)

    # num_cols_norm_outliers_null_done.append(new_col)

  # delete the null cols where they are not longer neccessary
    # delete the features in the del list
  df_null_done_del = df.copy()
  df.drop(cat_cols_obj_null + num_cols_null, axis=1, inplace=True) # delete all the null delete columns
# at the end of the null filling all the null columns needs to be removed which means only null_done lists are proceed as well as not null ones
#   print(time.clock() - start_time, "seconds took to finish null fix")

  return df, num_cols_null_done, cat_cols_obj_null_done, col_null_to_del_done


def feat_encode(df, cat_cols_obj_null_done, cat_cols_obj_not_null):
    #   start_time = time.clock()
    #   df_cat_save_loc = '/content/assets/data/av/cat_dfs/'

    labelencoder = LabelEncoder()
    onehotencoder = OneHotEncoder()

    #     col_for_one = []
    col_names = []
    cat_cols_encoded_all_dict = {}

    # cat_cols_encoded_all = cat_cols_obj_null_done + cat_cols_obj_not_null
    # at the end of all the encoding we have to delete all the columns which are categorical obj
    cat_cols_obj_all = cat_cols_obj_null_done + cat_cols_obj_not_null

    for col in cat_cols_obj_all:
        # split to make each columns
        col_name = str(col.split('_')[0])
        col_names.append(col_name)
    col_names = set(col_names)
    col_names = list(col_names)

    cat_cols_encoded_all_dictt = dict.fromkeys(col_names)
    # print('first dictt: ', cat_cols_encoded_all_dictt)
    cat_cols_encoded_all_dictt = defaultdict(list)
    # print('second dictt: ', cat_cols_encoded_all_dictt)


    for col in cat_cols_obj_all:

        cols_new_df = None
        col_for_one = []

        col_ori = str(col.split('_')[0])
        #         cat_cols_encoded_all_dictt = {}

        # create dictionary for each categorical columns
        # label encoding for all as well as if > 16 in the above case
        new_col = str(col) + ':label'
        df[new_col] = labelencoder.fit_transform(df[col].astype(str))
        col_for_one.append(new_col)

        #         print('col', col)
        # categorical encoding one hot encoding only if the nuniques < 16
        # one method is save the onhot df with the name and append to a df one hot list
        # two method is
        if df[col].nunique()>2:

            if df[col].nunique() <= 10:
                new_df = str(col) + ':onehot'
                # df_sample = pd.DataFrame(df[col])
                new_df = pd.get_dummies(df[col], drop_first=True, prefix=str(col + ':onehot'))
                #             print('new_df', new_df)
                df = pd.concat([df, new_df], axis=1)
                #             print('df', df)
                cols_new_df = new_df.columns.tolist()
                #             print(cols_new_df)
                col_for_one.append(cols_new_df)
                #             print('cols_for_one', col_for_one)

            else:
                # df[col].nunique() != 2:
                # # effect encoding is useless at this stage
                # # effect encoding - deviation or sum encoding
                # # new_df = str(col) + '_effect'
                # # effect_encoder = ce.sum_coding.SumEncoder(cols=col, verbose=False, )
                # # new_df = effect_encoder.fit_transform(df[col])
                # # df = pd.concat([df, new_df], axis=1)
                # # cols_new_df = new_df.columns.tolist()
                # # col_for_one.append(cols_new_df)
                #
                # # df[col].nunique() != 2:
                # # binary encoding where hashing is also taken
                # new_df = str(col) + ':binary'
                # bin_encoder = ce.BinaryEncoder(cols=col, return_df=True)
                # # Fit and Transform Data
                # new_df = bin_encoder.fit_transform(df[col])
                # df = pd.concat([df, new_df], axis=1)
                # cols_new_df = new_df.columns.tolist()
                # col_for_one.append(cols_new_df)

                # base N encoding where in this case base is taken as 5
                new_df = str(col) + ':base5'
                base_encoder = ce.BaseNEncoder(cols=[col], return_df=True, base=5)
                # Fit and Transform Data
                new_df = base_encoder.fit_transform(df[col])
                df = pd.concat([df, new_df], axis=1)
                cols_new_df = new_df.columns.tolist()
                col_for_one.append(cols_new_df)


    #         added each newly changes to the dictionary
        cat_cols_encoded_all_dict[str(col)] = col_for_one
        cat_cols_encoded_all_dictt[col_ori].append(col_for_one)

    # drop all the categorical cols from the df
    df.drop(cat_cols_obj_all, axis=1, inplace=True)  # delete all the null delete columns

    #     print(time.clock() - start_time, "seconds took to encode cat fix and save df")
    for key, value in cat_cols_encoded_all_dictt.items():
        flat_list = [item for sublist in value for item in sublist]
        cat_cols_encoded_all_dictt[key] = flat_list

    return df, cat_cols_encoded_all_dict, cat_cols_encoded_all_dictt




def shuffle_df(cat_cols_encoded_all_dictt, num_cols_null_done, num_cols_not_null, col_null_to_del_done):

    col_names = []
    num_cols_fixed_all_dict = {}
    num_cols_fixed_all =  num_cols_null_done + num_cols_not_null + col_null_to_del_done

    for col in num_cols_fixed_all:
        # split to make each columns
        col_name = str(col.split(':')[0])
        col_names.append(col_name)
    col_names = set(col_names)
    col_names = list(col_names)

    num_cols_fixed_all_dict = dict.fromkeys(col_names)

    num_cols_fixed_all_dictt = defaultdict(list)

    for col in num_cols_fixed_all_dict.keys():

        for num_col in num_cols_fixed_all:
            col_ori = str(num_col.split(':')[0])
            if str(col) == col_ori:
                num_cols_fixed_all_dictt[col_ori].append(num_col)

    cat_cols_encoded_all_dictt.update(num_cols_fixed_all_dictt)
    cat_num_cols_final_dict = cat_cols_encoded_all_dictt

    # make combinations
    allNames = sorted(cat_cols_encoded_all_dictt)
    combinations = it.product(*(cat_cols_encoded_all_dictt[name] for name in allNames))
    combinations = list(combinations)



    return combinations, cat_num_cols_final_dict



def save_files(df, ):
    pass





def prep_data(df, target_variable):

    # print the columns details of the dataset where numerical categorical and targets
    st.subheader('\n\n Feature Types based on the Values')
    df_ori = df.copy()
    df, id_cols, target_cols, cat_cols_obj, num_cols_all = feat_types(df, target_variable)
    st.write('Target Columns :   ', *target_cols)
    st.write('ID Columns :   ', (', '.join(id_cols)))
    # st.write('ID Columns: ', str(id_cols)[1:-1])
    st.write('Numerical Columns :   ', (', '.join(num_cols_all)))
    st.write('Categorical Columns :   ', (', '.join(cat_cols_obj)))

    # Highly corelated feature identiftion
    st.subheader('\n\n Highly Corelated Features')
    corr_df = cor_identifier(df, num_cols_all)
    st.dataframe(corr_df)
    st.text('*if the Individual Columns are selected please ignore')
    df.drop(id_cols, axis=1, inplace=True)  # delete all the null delete columns

    # Skewed Continuous Features identification and treatment
    st.subheader('\n\n Skewed Features and Normal Distributed Features')
    num_cols_skewed, num_cols_norm = skew_identifier(df,target_cols)
    st.write('Numerical Skewed Columns :   ', (', '.join(num_cols_skewed)))
    st.write('Numerical Normaly Distributed Columns :   ', (', '.join(num_cols_norm)))

    st.subheader('\n\n Skewed Columns Transformation Output')
    st.write('Skewed Columns are transformed to log and square root')
    df, num_cols_skewed_done, num_cols_norm = skew_fix(df, num_cols_skewed, num_cols_norm)
    st.dataframe(df.head(10))


    # Outlier Identification and Treatments
    st.subheader('\n\n Outliers Identification and Fixing')
    st.write('Outliers are Fixed Using Capping')
    df, num_cols_skewed_outliers, num_cols_norm_outliers, num_cols_norm_outliers_done,\
    num_cols_skewed_outliers_done, num_cols_norm = outlier_identifier_fix(df, num_cols_norm, num_cols_skewed)
    outliers_done = num_cols_norm_outliers_done + num_cols_skewed_outliers_done
    st.write('Numerical Outliers Capped Columns :   ', (', '.join(outliers_done)))
    st.dataframe(df.head(10))


    # Null Columns Find and Treatments
    # all the changes new cols fixes are added to the dataframe so again the columns have to be checked with the numerical and categorical types
    st.subheader('\n\n Null Columns Identification')
    cols_null_to_del, cat_cols_obj_null, cat_cols_obj_not_null, num_cols_null, num_cols_not_null = null_identifier(df, cat_cols_obj, target_cols)
    st.write('Numerical Null Columns to Delete :   ', (', '.join(cols_null_to_del)))
    st.text('*Morethan 60% of the Null value columns are considered to be deleted')
    st.write('Categorical Null Columns :   ', (', '.join(cat_cols_obj_null)))
    st.write('Categorical Not Null Columns :   ', (', '.join(cat_cols_obj_not_null)))
    st.write('Numerical Null Columns :   ', (', '.join(num_cols_null)))
    st.write('Numerical Not Null Columns :   ', (', '.join(num_cols_not_null)))


    # Fixing the Null Values
    st.subheader('\n\n Null Columns Fixing/Remove')
    df, num_cols_null_done, cat_cols_obj_null_done, col_null_to_del_done = null_fix(df, cols_null_to_del, cat_cols_obj_null, num_cols_null)
    # st.write('Null Columns are fixed with Mode, Median, Mean, added null column, extreme values')
    # st.dataframe(df)
    st.write('Categorical Columns Null fixed ', cat_cols_obj_null_done)
    st.write('Categorical Columns Null fixed ', cat_cols_obj_not_null)
    # st.write(df.info())


    # Categorical Feature Encoding
    st.subheader('\n\n Categorical Columns Convertion to Numerical')
    # df, cat_cols_obj_all_done, cat_cols_obj_onehot_df_done, cat_cols_obj_all_effect_df_done,\
    df, cat_cols_encoded_all_dict, cat_cols_encoded_all_dictt = feat_encode(df, cat_cols_obj_null_done, cat_cols_obj_not_null)
    combinations, cat_num_cols_final_dict = shuffle_df(cat_cols_encoded_all_dictt, num_cols_null_done, num_cols_not_null, col_null_to_del_done)
    st.write('Categorical Columns Converted to Numerical')
    # st.write('Categorical Columns Null fixed ', cat_cols_obj_null_done)
    # st.write('Categorical Columns Null fixed ', cat_cols_obj_not_null)
    # st.write('categorical columns all ', cat_cols_obj_all)
    st.write(cat_cols_encoded_all_dictt)
    # st.write(combinations)
    # st.dataframe(df.head())
    # st.write(df.head())
    # print(df.head())
    # print(combinations)
    # print(len(combinations))
    st.write('Numerical Null Columns to Delete :   ', (', '.join(cols_null_to_del)))
    st.text('*Morethan 60% of the Null value columns are considered to be deleted')
    st.write('Categorical Null Columns :   ', (', '.join(cat_cols_obj_null)))
    st.write('Categorical Not Null Columns :   ', (', '.join(cat_cols_obj_not_null)))
    st.write('Numerical Null Columns :   ', (', '.join(num_cols_null)))
    st.write('Numerical Not Null Columns :   ', (', '.join(num_cols_not_null)))
    st.write('Numerical Null Columns done :   ', (', '.join(num_cols_null_done)))
    st.write('Categorical Null Columns done :   ', (', '.join(cat_cols_obj_null_done)))
    st.write('To delete  Null Columns done :   ', (', '.join(col_null_to_del_done)))
    # df.to_csv('final_all_df.csv', index=False)

    all_columns_before= cat_cols_obj_null_done + cat_cols_obj_not_null + num_cols_null_done + num_cols_not_null + col_null_to_del_done
    all_columns_before= df.columns.values.tolist()
    all_columns_before = [i for i in all_columns_before if i not in target_cols]
    st.write('All the Colums/Features before cat encoding: ', (', '.join(all_columns_before)) )
    st.dataframe(df.head())
    # all_columns_= cat_cols_obj_null_done + cat_cols_obj_not_null + num_cols_null_done
    # st.write('all the combinations: ',combinations)
    st.write('Number of Combinations make from the features : ', len(combinations))

    # selecting the datasets
    st.subheader('\n\n Select the Dataset')
    st.write('\n For Random Datasets Select Random Button and For Custom Dataset Select Custom Button with columns')

    dataset_select = st.radio('Select Your Choice for Dataset',('Create Random Datasets', 'Create a Customized Dataset'))

    if dataset_select == 'Create Random Datasets':
        st.write('Select the Number of Random Datasets to Create')
        # if len(combinations) > 10:
        #     random_dataset_limit=1
        # else:
        #     random_dataset_limit = len(combinations)
        random_datasets = 1
        # random_datasets = st.slider('Adjust the slider to get number of datasets',1, random_dataset_limit, 1)
        st.write(random_datasets, ' Random Datasets Will be Created')
        random_dataset_cols = random.sample(combinations, random_datasets)
        # view_random_dataset = st.checkbox('View Random Datasets', value=False)
        # if view_random_dataset:
        #     st.write('Random Dataframes')
        for a,i in enumerate(random_dataset_cols):
            i = list(i)
            flatten_i = lambda *n: (e for a in n for e in (flatten_i(*a) if isinstance(a, (tuple, list)) else (a,)))
            i = list(flatten_i(i))
            df = df[i]
            df = pd.concat([df, df_ori[target_cols]], axis=1)
            # df = df[i]
            # df_random_dataset.to_csv('df_random_dataset' + str(a) +'.csv')
            # st.dataframe(df.head())




    elif dataset_select == 'Create a Customized Dataset':
        st.write("Please Select the Columns for the Dataset")
        customized_cols = st.multiselect('Select the Columns ', all_columns_before)
        st.write('Selected Columns: ', (',  '.join(customized_cols)))
        # view_custom_dataset = st.checkbox('View The Customized Dataset', value=False)
        # if view_custom_dataset:
        #     st.write('Customized DataFrame')
        df = df[customized_cols]
        df = pd.concat([df, df_ori[target_cols]], axis=1)
            # st.dataframe(df.head())


    return df

    # if st.button('Save&Build Models'):
    #     st.write('hi we click here')

    # if st.button('Save Datasets Only'):
    #     st.write('Save')
    #     df_customized.to_csv().encode('utf-8')
    #     st.download_button(label='Download the Dataset as CSV', data=df_customized, file_name='Custom.csv', mime='text/csv')
