import warnings
import streamlit as st


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-whitegrid')
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_colwidth', -1)
import warnings
warnings.simplefilter('ignore')
from sklearn.metrics import mean_squared_error, mean_squared_log_error, accuracy_score

# # simple classification output check without log transformation
# from sklearn import svm
# from sklearn import metrics
# from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, log_loss
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, LinearSVC, NuSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle



def convert_df(df):
    # return df.to_csv('final_processed_df.csv', index=False)
    return df.to_csv().encode('utf-8')

def download_model(model):
    # filename = 'finalized_model.sav'
    # joblib.dump(model, filename)
    # pickle_out = open('model.pkl', model='wb')
    model_name = 'final_model.pkl'
    pickle.dump(model, open(model_name, 'wb'))
    # pickle_out.close()
    # return model_out


def cross_val(target_type, y_true, y_pred):

    if target_type == 'Regression':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        return accuracy_score(y_true, y_pred)


def result_grid_search(model, param_grid, df, target_variable, target_type, rand_sets, search_type):

    if target_type == 'Regression':
        score = mean_squared_error
    else:
        score = accuracy_score
    # score = 'neg_mean_squared_log_error'
    # neg_mean_squared_log_error, neg_root_mean_squared_error
    # iterations = 10
    # define the scoreing metric
    # ‘roc_auc’ = metrics.roc_auc_score
    # ‘f1’ = metrics.f1_score
    # ‘accuracy’= metrics.accuracy_score
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

    if search_type == 'Random':
        grid = RandomizedSearchCV(model, param_grid, cv=5,  verbose=3, n_jobs=-1, n_iter=rand_sets, random_state=101)
        grid.fit(X_trn, df[target_variable])
        best_estimator = grid.best_estimator_
        best_score = grid.best_score_
        return best_estimator, best_score

    else:
        grid = GridSearchCV(model, param_grid, cv=5,  verbose=3, n_jobs=-1)
        grid.fit(X_trn, df[target_variable])
        best_estimator = grid.best_estimator_
        best_score = grid.best_score_
        return best_estimator, best_score

# random forest parma grids
# random forest parma grids
criterions = ['gini', 'entropy']
num_trees = [10, 20,30,50,80,100, 200, 300, 500, 700, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 7500, 10000]
min_leafs = [1, 2, 3, 4, 5, 6, 7]
num_leaves = [2, 4, 8, 16, 20, 32, 50, 64, 80, 100, 128, 150, 200, 256]
min_data_in_leafs = [1,2,4,6,8,10,12,14,16,18,20]
learning_rates = [0.1, 0.2, 0.3, 0.01, 0.02, 0.05, 0.08]
sub_samples = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
bagging_sets = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
min_child_samples = [2, 3,5,7,10,12,15]
reg_alphas = [0,0.01,0.03]
# colsample_bytrees = [0.65, 0.75, 0.8]
min_splits = [2,3,4,5,6,7]
max_depths = [2,3,4,5,6,7,8,9,10,11,12,15]
rands = [100, 99, 41, 101, 111, 42]
max_features = ['auto', 'sqrt', 'log2']


def rf_params():

    n_estimators_rf = st.multiselect('Please Select the Number of Estimators: ', num_trees)
    min_samples_leaf_rf = st.multiselect('Please Select the Min Samples Size for Leaf', min_leafs)
    min_samples_split_rf = st.multiselect('Please Select the Min Samples Size for Splitting', min_splits)
    max_depth_rf = st.multiselect('Please Select the Max Depth for tree', max_depths)
    random_state_rf = st.multiselect('Please Select the Random State',rands)

    param_grid_rf = {
                  'n_estimators': n_estimators_rf,
                  'min_samples_leaf': min_samples_leaf_rf,
                  'min_samples_split': min_samples_split_rf,
                  'max_depth': max_depth_rf,
                  'random_state': random_state_rf,
                  'n_jobs': [-1]}

    return param_grid_rf
    # best_rf = result_grid_search(model, param_grid, train_proc1, features_dummy, 'random')

# extra trees classifier param grids
def xt_params():

    n_trees_xt = st.multiselect('Please Select the Number of Estimators: ', num_trees)
    min_samples_leaf_xt = st.multiselect('Please Select the Min Samples Size for Leaf', min_leafs)
    min_samples_split_xt = st.multiselect('Please Select the Min Samples Size for Splitting', min_splits)
    max_depth_xt = st.multiselect('Please Select the Max Depth for tree', max_depths)
    max_featrues_xt = st.multiselect('Please Select the Max Features ',max_features)

    param_grid_xt = {
                  'n_estimators': n_trees_xt,
                  'min_samples_leaf': min_samples_leaf_xt,
                  'min_samples_split': min_samples_split_xt,
                  'max_depth': max_depth_xt,
                  'max_features': max_featrues_xt,
                  'n_jobs': [-1]}

    return param_grid_xt

# lightgbm classifier parameters
def lgbmcl_params():

    n_estimators_lg = st.multiselect('Please Select the Number of Estimators: ', num_trees)
    num_leaves_lg = st.multiselect('Please Select the Number of leaves: ', num_leaves)
    min_child_samples_lg = st.multiselect('Please Select the Min child samples', min_leafs)
    learning_rates_lg = st.multiselect('Please Select the learning_rates', learning_rates)
    min_data_in_leafs_lg = st.multiselect('Please Select the min_data_in_leafs', min_data_in_leafs)
    max_depth_lg = st.multiselect('Please Select the Max Depth for tree', max_depths)
    reg_alpha_lg = st.multiselect('Please Select the reg_alphas', reg_alphas)
    subsample_lg = st.multiselect('Please Select the sub_samples', sub_samples)
    bagging_fraction_lg = st.multiselect('Please Select the bagging_fraction', bagging_sets)

    param_grid_lg = {
                  'n_estimators': n_estimators_lg,
                    'num_leaves': num_leaves_lg,
                  'min_child_samples': min_child_samples_lg,
                  'bagging_fraction': bagging_fraction_lg,
                  'max_depth': max_depth_lg,
                  'learning_rate': learning_rates_lg,
                    'min_data_in_leaf': min_data_in_leafs_lg,
        'reg_alpha': reg_alpha_lg, 'subsample':subsample_lg

                  # 'n_jobs': [-1]
    }

    return param_grid_lg

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

    oofs_score = cross_val(target_type,target, oofs)
    download_model(clf)

    return oofs, preds, clf, oofs_score





def regressor_models(df, target_variable, class_model, target_type):

    search_cv = ['RandomSearch', 'GridSearch']

# Random Forest Classifier
    if class_model == 'RandomForestRegressor':
        param_grid_rf = rf_params()
        # print(param_grid_rf)
        # st.checkbox('Show the parameters', param_grid_rf)
        model_to_search =RandomForestRegressor()
        # check the gpu availability
        st.subheader('Perform the Hyperparameter Tunning')
        hyper_tune = st.selectbox('Select the Hyperparameter Tunning Search Method', search_cv)
        if hyper_tune == 'RandomSearch':
            st.write('You are Doing the Random Hyper parameter tuning')
            rand_sets = st.slider('Select the Number of Random Sessions to Search', 1, 20, 8, 1)
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

                # feature Importances
                importances = clf_rf.feature_importances_
                std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
                forest_importances = pd.Series(importances, index=features)

                st.subheader('Histogram for Feature Importance in RandomForest Model')
                fig2 = px.bar(forest_importances, y=features, x=std, orientation='h', labels=dict(x="Feature Importance", y="Features"))
                st.plotly_chart(fig2)


                with open('outputs/final_model.pkl', 'rb') as file:
                    st.download_button('Download Model as pkl', file, file_name='Final_model.pkl')



        elif hyper_tune == 'GridSearch':
            st.write('You are Doing the Grid Search Hyper parameter tuning')
            # rand_sets = st.slider('Select the Number of Random Sessions to Search', 1, 20, 8, 1)
            if st.button('Build Models'):
                best_estimator_rf, best_score_rf = result_grid_search(model_to_search, param_grid_rf, df, target_variable, target_type, rand_sets=0, search_type='GridSearch')
                st.write('Best Estimator', best_estimator_rf)
                st.write('Best Score', best_score_rf)
                train_df, test_df, features = data_pred_kfold(df, target_variable)
                # st.write('train set', train_df)
                # st.write('test set', test_df)
                # st.write('features', features)
                oofs_rf, preds_rf, clf_rf, oofs_rf_score = run_clf_kfold(best_estimator_rf, train_df, test_df, features,target_variable, target_type)
                st.subheader('Final Average Score for the Kfold Cross Validation')
                st.write(oofs_rf_score)

                # feature Importances
                importances = clf_rf.feature_importances_
                std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
                forest_importances = pd.Series(importances, index=features)

                st.subheader('Histogram for Feature Importance in RandomForest Model')
                fig2 = px.bar(forest_importances, y=features, x=std, orientation='h', labels=dict(x="Feature Importance", y="Features"))
                st.plotly_chart(fig2)

                with open('outputs/final_model.pkl', 'rb') as file:
                    st.download_button('Download Model as pkl', file, file_name='Final_model.pkl')


    elif class_model == 'ExtraTreesRegressor':
        param_grid_xt = xt_params()
        # st.write('param_grid_xt',param_grid_xt)
        # st.checkbox('Show the parameters', param_grid_rf)
        model_to_search = ExtraTreesRegressor()
        # check the gpu availability
        st.subheader('Perform the Hyperparameter Tunning')
        hyper_tune = st.selectbox('Select the Hyperparameter Tunning Search Method', search_cv)
        if hyper_tune == 'RandomSearch':
            st.write('You are Doing the Random Hyper parameter tuning')
            rand_sets = st.slider('Select the Number of Random Sessions to Search', 1, 20, 8, 1)
            if st.button('Build Models'):
                best_estimator_rf, best_score_rf = result_grid_search(model_to_search, param_grid_xt, df, target_variable, target_type, rand_sets,  search_type='Random')
                st.write('Best Estimator', best_estimator_rf)
                st.write('Best Score', best_score_rf)
                train_df, test_df, features = data_pred_kfold(df, target_variable)
                # st.write('train set', train_df)
                # st.write('test set', test_df)
                # st.write('features', features)
                oofs_xt, preds_xt, clf_xt, oofs_xt_score = run_clf_kfold(best_estimator_rf, train_df, test_df, features, target_variable, target_type)
                st.subheader('Final Average Score for the Kfold Cross Validation')
                st.write(oofs_xt_score)

                # feature Importances
                importances = clf_xt.feature_importances_
                std = np.std([tree.feature_importances_ for tree in clf_xt.estimators_], axis=0)
                xt_importances = pd.Series(importances, index=features)

                st.subheader('Histogram for Feature Importance in ExtratreeClassifier Model')
                fig2 = px.bar(xt_importances, y=features, x=std, orientation='h', labels=dict(x="Feature Importance", y="Features"))
                st.plotly_chart(fig2)

                with open('outputs/final_model.pkl', 'rb') as file:
                    st.download_button('Download Model as pkl', file, file_name='Final_model.pkl')


        elif hyper_tune == 'GridSearch':
            st.write('You are Doing the Grid Search Hyper parameter tuning')
            # rand_sets = st.slider('Select the Number of Random Sessions to Search', 1, 20, 8, 1)
            if st.button('Build Models'):
                best_estimator_rf, best_score_rf = result_grid_search(model_to_search, param_grid_xt, df, target_variable, target_type, rand_sets=0, search_type='GridSearch')
                st.write('Best Estimator', best_estimator_rf)
                st.write('Best Score', best_score_rf)
                train_df, test_df, features = data_pred_kfold(df, target_variable)
                # st.write('train set', train_df)
                # st.write('test set', test_df)
                # st.write('features', features)
                oofs_xt, preds_xt, clf_xt, oofs_xt_score = run_clf_kfold(best_estimator_rf, train_df, test_df,
                                                                         features, target_variable, target_type)
                st.subheader('Final Average Score for the Kfold Cross Validation')
                st.write(oofs_xt_score)

                # feature Importances
                importances = clf_xt.feature_importances_
                std = np.std([tree.feature_importances_ for tree in clf_xt.estimators_], axis=0)
                xt_importances = pd.Series(importances, index=features)

                st.subheader('Histogram for Feature Importance in ExtratreeClassifier Model')
                fig2 = px.bar(xt_importances, y=features, x=std, orientation='h', labels=dict(x="Feature Importance", y="Features"))
                st.plotly_chart(fig2)

                with open('outputs/final_model.pkl', 'rb') as file:
                    st.download_button('Download Model as pkl', file, file_name='Final_model.pkl')

# lightgbm classifier
    elif class_model == 'LightGBMRegressor':
        param_grid_lg = lgbmcl_params()
        # st.write('param_grid_xt',param_grid_xt)
        # st.checkbox('Show the parameters', param_grid_rf)
        model_to_search = LGBMRegressor()
        # check the gpu availability
        st.subheader('Perform the Hyperparameter Tunning')
        hyper_tune = st.selectbox('Select the Hyperparameter Tunning Search Method', search_cv)
        if hyper_tune == 'RandomSearch':
            st.write('You are Doing the Random Hyper parameter tuning')
            rand_sets = st.slider('Select the Number of Random Sessions to Search', 1, 20, 8, 1)
            if st.button('Build Models'):
                best_estimator_rf, best_score_rf = result_grid_search(model_to_search, param_grid_lg, df, target_variable, rand_sets, target_type,  search_type='Random')
                st.write('Best Estimator', best_estimator_rf)
                st.write('Best Score', best_score_rf)
                train_df, test_df, features = data_pred_kfold(df, target_variable)
                # st.write('train set', train_df)
                # st.write('test set', test_df)
                # st.write('features', features)
                oofs_lg, preds_lg, clf_lg, oofs_lg_score = run_clf_kfold(best_estimator_rf, train_df, test_df, features, target_variable, target_type)
                st.subheader('Final Average Score for the Kfold Cross Validation')
                st.write(oofs_lg_score)

                # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
                feature_imp = pd.DataFrame(sorted(zip(clf_lg.feature_importances_, features)),
                                           columns=['Feature Importance', 'Features'])
                st.subheader('Histogram for Feature Importance in LightGBM Classifier Model')
                fig2 = px.bar(feature_imp.sort_values(by="Feature Importance", ascending=True), y='Features', x='Feature Importance', orientation='h',
                              labels=dict(x="Feature Importance", y="Features"))
                st.plotly_chart(fig2)

                with open('outputs/final_model.pkl', 'rb') as file:
                    st.download_button('Download Model as pkl', file, file_name='Final_model.pkl')


        elif hyper_tune == 'GridSearch':
            st.write('You are Doing the Grid Search Hyper parameter tuning')
            # rand_sets = st.slider('Select the Number of Random Sessions to Search', 1, 20, 8, 1)
            if st.button('Build Models'):
                best_estimator_rf, best_score_rf = result_grid_search(model_to_search, param_grid_lg, df, target_variable, target_type, rand_sets=0, search_type='GridSearch')
                st.write('Best Estimator', best_estimator_rf)
                st.write('Best Score', best_score_rf)
                train_df, test_df, features = data_pred_kfold(df, target_variable)
                # st.write('train set', train_df)
                # st.write('test set', test_df)
                # st.write('features', features)
                oofs_lg, preds_lg, clf_lg, oofs_lg_score = run_clf_kfold(best_estimator_rf, train_df, test_df,
                                                                         features, target_variable, target_type)
                st.subheader('Final Average Score for the Kfold Cross Validation')
                st.write(oofs_lg_score)

                # feature Importances
                # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
                feature_imp = pd.DataFrame(sorted(zip(clf_lg.feature_importances_, features)),
                                           columns=['Feature Importance', 'Features'])
                st.subheader('Histogram for Feature Importance in LightGBM Classifier Model')
                fig2 = px.bar(feature_imp.sort_values(by="Feature Importance", ascending=True), y='Features', x='Feature Importance',
                              orientation='h', labels=dict(x="Feature Importance", y="Features"))
                st.plotly_chart(fig2)

                with open('outputs/final_model.pkl', 'rb') as file:
                    st.download_button('Download Model as pkl', file, file_name='Final_model.pkl')
