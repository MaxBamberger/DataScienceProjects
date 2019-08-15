
#Import packages:
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pandas.plotting import scatter_matrix

from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier,
                              AdaBoostClassifier)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (accuracy_score,
                            confusion_matrix,
                            log_loss,
                            brier_score_loss,
                            roc_auc_score,
                            roc_curve,
                            confusion_matrix)

from sklearn.model_selection import (train_test_split,
                                    KFold,
                                    cross_val_score)

from helper_functions import (run_cv,
                             auto_dummy,
                             googone,
                             plot_classification_scatter,
                             feature_graph)

#Import Data:
df_raw_train = pd.read_csv('data/churn_train.csv')
df_raw_test = pd.read_csv('data/churn_test.csv')
df_train = df_raw_train.copy()
df_test = df_raw_test.copy()

#Clean, Transform, and Drop Data (both training and test data):
df_clean = googone(df_train, train = True)
df_clean_test = googone(df_test, train = False)

#If test data n_features does not match:
if df_clean_test.shape[1] != df_clean.shape[1]:
    col_test = df_clean_test.columns
    col_X = df_clean.columns
    dropThese = [col for col in col_test if col not in col_X]
    df_clean_test.drop(dropThese,axis=1,inplace=True)

#Create arrays for fitting:
y = df_clean["churn?"].values
X = df_clean.drop("churn?",axis=1).values

y_test = df_clean_test["churn?"].values
X_test = df_clean_test.drop("churn?",axis=1).values

#Explore the data and visualize.. will need to put this into a notebook maybe.:
# plot_classification_scatter(df_clean, y)

#Construct a series of models
gdbr = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, random_state=1)
clf = RandomForestClassifier(max_features='auto', oob_score=True, n_estimators=100, max_depth=6)
abr = AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.01, n_estimators=100, random_state=1)

#Cross validate on our Training data:
y_pred_gdbr, log_losses_gdbr, briers_gdbr, log_auc_gdbr = run_cv(X, y, gdbr)
y_pred_clf, log_losses_clf, briers_clf, log_auc_clf = run_cv(X, y, clf)
y_pred_abr, log_losses_abr, briers_abr, log_auc_abr = run_cv(X, y, abr)

#Evaluate loss function:
print("For Grad Boosting -- Log_Loss: {:2.3f}, Brier Score: {:2.3f}, AUC Score: {:2.3f},".format(log_losses_gdbr, briers_gdbr, log_auc_gdbr))
print("For RandForest Clf -- Log_Loss: {:2.3f}, Brier Score: {:2.3f}, AUC Score: {:2.3f},".format(log_losses_clf, briers_clf, log_auc_clf))
print("For Adapt Boosting -- Log_Loss: {:2.3f}, Brier Score: {:2.3f}, AUC Score: {:2.3f},".format(log_losses_abr, briers_abr, log_auc_abr))

#Evaluate the feature importance / coefficients:
df_X = df_clean.drop("churn?",axis=1)
feature_graph(gdbr,df_X)
feature_graph(clf,df_X)
feature_graph(abr,df_X)

#Choose best model (Random Forest Classifer):
choose = str(input('\nSelect (1) to use a GradientBoostingClassifier, \n(2) for RandForest or \n(3) for Adapt Boosting Decision Tree \n'))
if choose == '1':
    final_model = gdbr.fit(X, y)
if choose == '2':
    final_model = clf.fit(X, y)
if choose == '3':
    final_model = abr.fit(X, y)

#Test Model on CSV:
y_prob = final_model.predict_proba(X_test)
y_hat = final_model.predict(X_test)

#Output results:
results = pd.DataFrame({'probability':y_prob[:,1],'prediction':y_hat,'y_true':y_test})
results.to_csv('results.csv')
log_loss = log_loss(y_test,y_prob[:,1])
briers = brier_score_loss(y_test,y_prob[:,1])
auc = roc_auc_score(y_test,y_prob[:,1])
print("\nTest Data Results -- Log_Loss: {:2.3f}, Brier Score: {:2.3f}, AUC Score: {:2.3f}".format(log_loss, briers, auc))

#Create ROC curve:
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_prob[:,1])
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of test data")
plt.show()
