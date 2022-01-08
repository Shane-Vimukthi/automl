import warnings
### importing lazypredict library
import lazypredict
### importing LazyClassifier for classification problem
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
# from lazypredict.Supervised import LazyClassifier
#
import streamlit as st
import streamlit.components.v1 as stc

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import time

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
plt.style.use('seaborn-whitegrid')
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import category_encoders as ce
# import os
# from collections import defaultdict
# from sklearn.model_selection import train_test_split


# import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import accuracy_score, f1_score
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
#
# from sklearn.linear_model import LogisticRegression
# import joblib

# # from lightgbm import LGBMClassifier
# # # from catboost import CatBoostClassifier
# # from xgboost import XGBClassifier
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import category_encoders as ce
# import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_colwidth', -1)

import warnings
warnings.simplefilter('ignore')
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error, accuracy_score



def cross_val(target_type, y_true, y_pred):

    if target_type == 'Classification':
        return accuracy_score(y_true, y_pred)
    else:
        return np.sqrt(mean_squared_error(y_true, y_pred))




def model_lazy_pred(df,target_variable, target_type):

    # do for the classification using for custom or one dataset so the problem is over
    y_data = df[target_variable]
    X_data = df.drop([target_variable], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.3, random_state=123)

    if target_type == str('Multi-Label Classification') :
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    elif target_type == str('Regression'):
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    elif target_type == str('Binary Classification') :
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)


    # convert the datatype of the target variable


    # do for the regression

    return models, predictions