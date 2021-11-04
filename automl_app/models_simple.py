import warnings

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
from sklearn.metrics import mean_squared_error, mean_squared_log_error, accuracy_score



def cross_val(target_type, y_true, y_pred):

    if target_type == 'Classification':
        return accuracy_score(y_true, y_pred)
    else:
        return 1000 * np.sqrt(mean_squared_error(y_true, y_pred))





def model_simple(df, target_type):
    pass