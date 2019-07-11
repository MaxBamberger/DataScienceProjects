# Here we will use boosting to solve a regression problem.
# Specifically we would like to predict Boston house prices based on 13 features.

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()
# House Prices:
y = boston.target
# The other 13 features:
x = boston.data

print(boston.DESCR)

    # :Attribute Information (in order):
    #     - CRIM     per capita crime rate by town
    #     - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    #     - INDUS    proportion of non-retail business acres per town
    #     - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    #     - NOX      nitric oxides concentration (parts per 10 million)
    #     - RM       average number of rooms per dwelling
    #     - AGE      proportion of owner-occupied units built prior to 1940
    #
    #     - DIS      weighted distances to five Boston employment centres
    #     - RAD      index of accessibility to radial highways
    #     - TAX      full-value property-tax rate per $10,000
    #     - PTRATIO  pupil-teacher ratio by town
    #     - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    #     - LSTAT    % lower status of the population
    #     - MEDV     Median value of owner-occupied homes in $1000's

X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=1)


#Create 3 models:

rf = RandomForestRegressor(n_estimators=100,
                            n_jobs=-1,
                            random_state=1)

gdbr = GradientBoostingRegressor(learning_rate=0.1,
                                  loss='ls',
                                  n_estimators=100,
                                  random_state=1)

abr = AdaBoostRegressor(DecisionTreeRegressor(),
                         learning_rate=0.1,
                         loss='linear',
                         n_estimators=100,
                         random_state=1)

#Create a cross validation score function for each model:
def cvs(model,x_data,y_data):
    mse = np.mean(-1*cross_val_score(model,x_data,y_data,cv=5,scoring='neg_mean_squared_error'))
    r2 = np.mean(cross_val_score(model,x_data,y_data,cv=5,scoring='r2'))
    return mse, r2

rf_mse, rf_r2 = cvs(rf, X_train,y_train)
gdbr_mse, gdbr_r2 = cvs(gdbr, X_train,y_train)
abr_mse, abr_r2 = cvs(abr, X_train,y_train)

#Compare the MSE and R2 for each model:
print("{:26s} {} | MSE: {:2.3f} | R2: {:2.3f}".format('RandomForestRegressor','TRAIN CV',rf_mse, rf_r2))
print("{:26s} {} | MSE: {:2.3f} | R2: {:2.3f}".format('GradientBoostingRegressor','TRAIN CV',gdbr_mse, gdbr_r2))
print("{:26s} {} | MSE: {:2.3f} | R2: {:2.3f}".format('AdaBoostRegressor','TRAIN CV',abr_mse, abr_r2))

#Make a Gradient Boosting model with a learning_rate of 1 and compare the result to the previous gdbr:
gdbr_1 = GradientBoostingRegressor(learning_rate=1,
                                  loss='ls',
                                  n_estimators=100,
                                  random_state=1)

gdbr_1_mse, gdbr_1_r2 = cvs(gdbr_1, X_train,y_train)

print("\n{:28s} {} | MSE: {:2.3f} | R2: {:2.3f}".format('GradientBoostingRegressor_1','TRAIN CV',gdbr_1_mse, gdbr_1_r2))


#Make function that plots the predictions and at each stage of the boosting algorithm.
#Plot both of the MSE for the predictions on the training data as well as the test data:
def stage_score_plot(estimator, ax, X_train, y_train, X_test, y_test):
    '''
    Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                X_train: 2d numpy array
                y_train: 1d numpy array
                X_test: 2d numpy array
                y_test: 1d numpy array

    Returns: A plot of the number of iterations vs the MSE for the model for
    both the training set and test set.
    '''
    estimator.fit(X_train,y_train)
    y_staged_predict_train = estimator.staged_predict(X_train)
    y_staged_predict_test = estimator.staged_predict(X_test)
    mse_train=[]
    mse_test=[]
    for y_pred in y_staged_predict_train:
        mse_train.append(mean_squared_error(y_train, y_pred))

    for y_pred in y_staged_predict_test:
        mse_test.append(mean_squared_error(y_test, y_pred))

    x = np.arange(0,100,1)
    ax.plot(x,mse_train,label="{} Training Data, learning rate: {}".format(estimator.__class__.__name__,estimator.learning_rate))
    ax.plot(x,mse_test,label="{} Testing Data, learning rate: {}".format(estimator.__class__.__name__,estimator.learning_rate))

def rf_score_plot(randforest, ax, X_train, y_train, X_test, y_test):
    '''
        Parameters: randforest: RandomForestRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array
        Returns: The prediction of a random forest regressor on the test set
    '''
    randforest.fit(X_train, y_train)
    y_test_pred = randforest.predict(X_test)
    test_score = mean_squared_error(y_test, y_test_pred)
    ax.axhline(test_score, alpha = 0.7, c = 'y', lw=3, ls='-.', label =
                                                        'Random Forest Test')

fig, ax = plt.subplots(1, figsize=(8,8))

#Feed the function the boosting algo that has learning_rate = 1:
stage_score_plot(gdbr, ax, X_train, y_train, X_test, y_test)

#Feed the function the boosting algo that has learning_rate = 0.1:
stage_score_plot(gdbr_1, ax, X_train, y_train, X_test, y_test)

#Add a horizontal line for our Random Forrest model on the test data:
rf_score_plot(rf, ax, X_train, y_train, X_test, y_test)

ax.set_title('Gradient Boosting Evaluation at stages, with Random Forrest Test')
ax.set_ylabel('Mean Squared Error')
ax.set_xlabel('n_iterations (boosting stages)')
plt.legend()
plt.show()
