import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.tree import DecisionTreeClassifier

#from custom_scorer_module import custom_scoring_function
from sklearn.metrics import (accuracy_score,
                            confusion_matrix,
                            log_loss,
                            brier_score_loss,
                            roc_auc_score,
                            confusion_matrix)

def run_cv(X,y,model_type):
    # Construct a kfolds object
    kf = KFold(n_splits=5,shuffle=True)
    y_pred = y.copy()
    log_losses = []
    briers = []
    aucs = []

    # Iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        model_type.fit(X_train,y_train)
        y_pred[test_index] = model_type.predict_proba(X_test)[:,1]
        log_losses.append(log_loss(y,y_pred))
        briers.append(brier_score_loss(y,y_pred))
        aucs.append(roc_auc_score(y,y_pred))
    return y_pred, np.mean(log_losses), np.mean(briers), np.mean(aucs)


def auto_dummy(df, cols, drop, pprint=False):
    '''
    cols as list of columns
    drop, which value to drop, 'drop_first' -> automatically drop first
    '''
    df_temp = df.copy()
    if type(cols) == str:
        cols = [cols]
    if type(drop) == str:
        drop = [drop]
    for i,col in enumerate(cols):
        if pprint: print(col)
        pref = ''#col.strip()[:2]
        if drop[i] == 'drop_first':
            if pprint: print(col, ' dropping first')
            df_add = pd.get_dummies(df_temp[col], prefix=pref, drop_first=True)
            df_temp = df_temp.join(df_add)
            df_temp.drop(col, axis=1, inplace=True)
        else:
            df_add = pd.get_dummies(df_temp[col], prefix=pref)
            df_temp = df_temp.join(df_add)
            if pprint: print('droping ', col)
            try:
                df_temp.drop([col, pref + '_' + str(drop[i])], axis=1, inplace=True)
            except Exception as e: print('oops', col, e)
    return df_temp


def googone(df, train = True, drops = None, pprint=False):
    '''
    df = pandas df
    train = bool = is it training set (vs test set)
    drops = list or str = what columns to drop immediately
    '''
    df_temp = df.copy()

    #CREATE churn column:
    df_temp["signup_date"] = pd.to_datetime(df_temp["signup_date"])
    df_temp["last_trip_date"] = pd.to_datetime(df_temp["last_trip_date"])
    df_temp["churn?"] = df_temp['last_trip_date'] \
            .apply(lambda x : 1 if x < dt.datetime.strptime('2014-06-01',"%Y-%m-%d") else 0)

    #DROP USELESS
    if drops != None:
        if type(drops) == str: drops = [drops]
        keeps = [col for col in df_temp.columns if col not in drops]
        df_temp = df_temp[keeps]

    #DROP LAST_TRIP_DATE
    df_temp.drop('last_trip_date', axis=1, inplace=True)

    #DROP ROWS // DON'T DO THESE FOR THE TES TEST SET
    if train:
        df_temp.dropna(inplace=True)  #DROP ALL NULLS FOR NOW
        df_temp = df_temp[df_temp['avg_dist']>0] #DROP ALL AVG_DIST = 0
        df_temp = df_temp[df_temp['signup_date'] < '2014-06-01'] #DROP SIGNUPS IN JUNE - INVALID FOR PREDICTION


    #CLEAN NULLS - ave rating driver by/of driver, phone
    df_temp['phone'] = df_temp['phone'].fillna('Null')
    df_temp['avg_rating_by_driver'] = df_temp['avg_rating_by_driver'].fillna( np.mean(df_temp['avg_rating_by_driver']) )
    df_temp['avg_rating_of_driver'] = df_temp['avg_rating_of_driver'].fillna( np.mean(df_temp['avg_rating_of_driver']) )



    #convert DATES
    df_temp['signup_date'] = pd.to_datetime(df_temp['signup_date'])


    #converty BOOLEANS
    df_temp['luxury_car_user']  = df_temp['luxury_car_user']*1

    #ONE HOT FEATURES - Drop original features
    features = ['city','phone']

    tops = [] #determine the most common value for the feature
    if pprint: print('doing dummies')
    for f in features:
        top = df_temp.groupby(f).count().sort_values(by=df_temp.columns[0], ascending=False).reset_index()[f][0]
        tops.append(top)
    df_temp = auto_dummy(df_temp, features, tops, pprint=pprint)

    df_temp.drop('signup_date', axis=1, inplace=True)
    return df_temp


#Explore the data and visualize:

def plot_classification_scatter(df, y):
    '''
    Takes in a dataframe, this can include churn or not
    y must be the churn column as 0 and 1
    '''
    # maps y to a color for the scatter matrix
    colors = np.array(['#0392cf', '#e41a1c'])[y]
    # scatter_matrix of features with colors corresponding to
    ax = scatter_matrix(df, color=colors, alpha=0.6, figsize=(25,25), diagonal='kde')
    plt.tight_layout()
    plt.show()

def feature_graph(estimator, X):
    '''
    To be called after modeling is done, takes in the ex: clf or ada into the
    estimator
    X must be a dataframe without churn to be able to get colnames

    '''
    # X should not just be .values it must be a dataframe to get colnames
    feat_scores = pd.Series(estimator.feature_importances_,
                           index= X.columns)
    # sort feature scores for graph
    feat_scores = feat_scores.sort_values()
    # plot feature scores
    ax = feat_scores.plot(kind='barh',
                         figsize=(10,8),
                         color='b')
    # model name for every estimator
    model_name = type(estimator).__name__
    ax.set_title('Average Feature Importance for {} Model'.format(
        model_name))
    ax.set_xlabel('Average contribution to information gain')
    plt.show()
