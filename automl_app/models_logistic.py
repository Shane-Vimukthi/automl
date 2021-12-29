import warnings
import streamlit as st


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-whitegrid')
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_colwidth', -1)
import warnings
warnings.simplefilter('ignore')
from sklearn.metrics import mean_squared_error, mean_squared_log_error, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix




def convert_df(df):
    # return df.to_csv('final_processed_df.csv', index=False)
    return df.to_csv().encode('utf-8')

def download_model(model):
    # filename = 'finalized_model.sav'
    # joblib.dump(model, filename)
    # pickle_out = open('model.pkl', model='wb')
    model_name = 'outputs/final_model.pkl'
    pickle.dump(model, open(model_name, 'wb'))
    # pickle_out.close()
    # return model_out


def cross_val(target_type, y_true, y_pred):

    if target_type == 'Regression':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        return accuracy_score(y_true, y_pred)


def result_grid_search(model, param_grid, df, target_variable, target_type, rand_sets, search_type):
    # if
    # score = 'neg_mean_squared_log_error'
    # neg_mean_squared_log_error, neg_root_mean_squared_error
    # iterations = 10
    # define the scoreing metric
    # ‘roc_auc’ = metrics.roc_auc_score
    # ‘f1’ = metrics.f1_score
    # if target_type == 'Regression':
    #     score = mean_squared_error
    # else:
    #     score = accuracy_score
    # accuracy= metrics.accuracy_score
    # ‘neg_root_mean_squared_error’=metrics.mean_squared_error
    # ‘neg_mean_squared_log_error’=metrics.mean_squared_log_error
    # ‘neg_root_mean_squared_error’=metrics.mean_squared_error
    lb_make = LabelEncoder()
    df[target_variable] = lb_make.fit_transform(df[target_variable])


    y_data = df[target_variable]
    X_data = df.drop([target_variable], axis=1)
    scaler = StandardScaler()
    _ = scaler.fit(X_data)
    # X_data = df.drop([target_variable], axis=1)
    X_trn = scaler.transform(X_data)
    # , scoring = score
    if search_type == 'Random':
        grid = RandomizedSearchCV(model, param_grid, cv=5, verbose=3, n_jobs=-1, n_iter=rand_sets, random_state=101)
        grid.fit(X_trn, df[target_variable])
        best_estimator = grid.best_estimator_
        best_score = grid.best_score_
        return best_estimator, best_score

    else:
        grid = GridSearchCV(model, param_grid, cv=5, verbose=3, n_jobs=-1)
        grid.fit(X_trn, df[target_variable])
        best_estimator = grid.best_estimator_
        best_score = grid.best_score_
        return best_estimator, best_score

# random forest parma grids

c_vals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
penalties = ['none', 'l2', 'elasticnet']
solvers = ['newton-cg', 'lbfgs', 'liblinear']

def lr_params():

    c_val_lr = st.multiselect('Please Select the C Values: ', c_vals)
    penalty_lr = st.multiselect('Please Select the Penalties', penalties)
    solvers_lr = st.multiselect('Please Select the Solvers', solvers)
    # max_depth_rf = st.multiselect('Please Select the Max Depth for tree', max_depths)
    # random_state_rf = st.multiselect('Please Select the Random State',rands)

    param_grid_lr = {
                  'C': c_val_lr,
                  'penalty': penalty_lr,
                  'solver': solvers_lr,
                  'n_jobs': [-1]}

    return param_grid_lr
    # best_rf = result_grid_search(model, param_grid, train_proc1, features_dummy, 'random')

def data_pred_kfold(df, target_variable):

    # y_data = df[target_variable]
    # X_data = df.drop([target_variable], axis=1)
    lb_make = LabelEncoder()
    df[target_variable] = lb_make.fit_transform(df[target_variable])
    features = df.columns.values.tolist()
    features = [i for i in features if i not in target_variable]
    train_df = df.sample(frac=0.8)
    test_df = df.drop(train_df.index)



    return train_df, test_df, features

#kfold cross validation
def run_clf_kfold(clf, train, test, features, target_variable, target_type):
    N_SPLITS = 5
    oofs = np.zeros(len(train))
    preds = np.zeros(len(test))

    target = train[target_variable]

    folds = StratifiedKFold(n_splits=N_SPLITS)

    skfold_target = pd.qcut(train[target_variable], N_SPLITS, labels=False,
                            duplicates='drop')
    st.subheader('K fold Scores for 5 Splits')
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, skfold_target)):
        # st.write('K Fold Scores')
        # print(f'\n------------- Fold {fold_ + 1} -------------')
        st.write('Fold Number : ', (fold_ + 1))
        ############# Get train, validation and test sets along with targets ################

        ### Training Set
        X_trn, y_trn = train[features].iloc[trn_idx], target.iloc[trn_idx]

        ### Validation Set
        X_val, y_val = train[features].iloc[val_idx], target.iloc[val_idx]

        ### Test Set
        X_test = test[features]

        ############# Scaling Data ################
        scaler = StandardScaler()
        _ = scaler.fit(X_trn)

        X_trn = scaler.transform(X_trn)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        ############# Fitting and Predicting ################

        #         _ = clf.fit(X_trn, y_trn,**params)
        _ = clf.fit(X_trn, y_trn)

        ### Instead of directly predicting the classes we will obtain the probability of positive class.
        preds_val = clf.predict(X_val)
        preds_test = clf.predict(X_test)

        fold_score = cross_val(target_type ,y_val, preds_val)
        # print(len(preds_val))
        st.write('length of validation set is ',len(preds_val))
        st.write('Metric score for validation set is ', fold_score)
        # print(f'\nMetric score for validation set is {fold_score}')

        oofs[val_idx] = preds_val
        preds += preds_test / N_SPLITS

    # conf_matrix = confusion_matrix(target, oofs)
    oofs_score = cross_val(target_type,target, oofs)
    download_model(clf)


    return oofs, preds, clf, oofs_score





def logistic_reg_models(df, target_variable, class_model, target_type):

    search_cv = ['RandomSearch', 'GridSearch']

# Random Forest Classifier
    if class_model == 'LogisticRegression':
        param_grid_rf = lr_params()
        # print(param_grid_rf)
        # st.checkbox('Show the parameters', param_grid_rf)
        model_to_search =LogisticRegression()
        # check the gpu availability
        st.subheader('Perform the Hyperparameter Tunning')
        hyper_tune = st.selectbox('Select the Hyperparameter Tunning Search Method', search_cv)
        if hyper_tune == 'RandomSearch':
            st.write('You are Doing the Random Hyper parameter tuning')
            rand_sets = st.slider('Select the Number of Random Sessions to Search', 1, 20, 8, 1)
            st.write('\n\n')
            if st.button('Build Models'):
                best_estimator_rf, best_score_rf = result_grid_search(model_to_search, param_grid_rf, df, target_variable, target_type,rand_sets, search_type='Random')
                st.write('Best Estimator', best_estimator_rf)
                st.write('Best Score', best_score_rf)
                train_df, test_df, features = data_pred_kfold(df, target_variable)
                # st.write('train set', train_df)
                # st.write('test set', test_df)
                # st.write('features', features)
                oofs_rf, preds_rf, clf_rf, oofs_rf_score = run_clf_kfold(best_estimator_rf,train_df, test_df, features, target_variable, target_type)
                st.subheader('Final Average Score for the Kfold Cross Validation')
                st.write(oofs_rf_score)

                # # feature Importances
                # importances = clf_rf.feature_importances_
                # std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
                # forest_importances = pd.Series(importances, index=features)
                #
                st.subheader('Histogram for Feature Importance in LogisticRegression Model')
                # fig2 = px.bar(forest_importances, y=features, x=std, orientation='h', labels=dict(x="Feature Importance", y="Features"))
                # st.plotly_chart(fig2)

                # Those values, however, will show that the second parameter
                # is more influential
                # st.write('feature importance', (np.std(features, 0) * clf_rf.coef_))

                feature_importance = abs(clf_rf.coef_[0])
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                sorted_idx = np.argsort(feature_importance)
                pos = np.arange(sorted_idx.shape[0]) + .5

                featfig = plt.figure()
                featax = featfig.add_subplot(1, 1, 1)
                featax.barh(pos, feature_importance[sorted_idx], align='center')
                featax.set_yticks(pos)
                featax.set_yticklabels(np.array(features)[sorted_idx], fontsize=8)
                featax.set_xlabel('Relative Feature Importance')

                st.plotly_chart(featfig)


                with open('outputs/final_model.pkl', 'rb') as file:
                    st.download_button('Download Model as pkl', file, file_name='Final_model.pkl')



        elif hyper_tune == 'GridSearch':
            st.write('You are Doing the Grid Search Hyper parameter tuning')
            # rand_sets = st.slider('Select the Number of Random Sessions to Search', 1, 20, 8, 1)
            st.write('\n\n')
            if st.button('Build Models'):
                best_estimator_rf, best_score_rf = result_grid_search(model_to_search, param_grid_rf, df, target_variable, target_type, rand_sets=0, search_type='GridSearch')
                st.write('Best Estimator', best_estimator_rf)
                st.write('Best Score', best_score_rf)
                train_df, test_df, features = data_pred_kfold(df, target_variable)
                # st.write('train set', train_df)
                # st.write('test set', test_df)
                # st.write('features', features)
                oofs_lr, preds_lr, clf_lr, oofs_rf_score = run_clf_kfold(best_estimator_rf,train_df, test_df, features, target_variable, target_type)
                st.subheader('Final Average Score for the Kfold Cross Validation')
                st.write(oofs_rf_score)

                st.subheader('Histogram for Feature Importance in LogisticRegression Model')
                feature_importance = abs(clf_lr.coef_[0])
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                sorted_idx = np.argsort(feature_importance)
                pos = np.arange(sorted_idx.shape[0]) + .5

                featfig = plt.figure()
                featax = featfig.add_subplot(1, 1, 1)
                featax.barh(pos, feature_importance[sorted_idx], align='center')
                featax.set_yticks(pos)
                featax.set_yticklabels(np.array(features)[sorted_idx], fontsize=8)
                featax.set_xlabel('Relative Feature Importance')

                st.plotly_chart(featfig)


                with open('outputs/final_model.pkl', 'rb') as file:
                    st.download_button('Download Model as pkl', file, file_name='Final_model.pkl')


