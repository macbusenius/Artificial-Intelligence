'''
Program Name: Lab04busenium.py
Class: ENSE496AD Machine Learning

Name: Mckenzie Busenius
SID: 200378076

Professor: Dr. Kin-Choong Yow
Lab Professor: Usman Munawar

Description: Implementation of the algorithm, Support Vector Machine. Then we are apply it
             to two different datasets, Breast Cancer Wisconsin and Titanic Dataset.

Acknowledgments:
    [1] https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2#66a2

'''
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random as rd
from copy import deepcopy

from sklearn.metrics import confusion_matrix,accuracy_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import shuffle
import sys,os
# import seaborn as sns
# %matplotlib inline


################################################################
#   Problem 1: Breast Cancer Wisconsin Dataset                 #
#                                                              #
#   Description: Predict whether the cancer isbenign or        #
#                malignant                                     #
################################################################
'''
Part A: Dataset ‘breast-cancer-wisconsin-data’ is given in the assignment
        pack. Load the dataset in dataframe.
'''
breast_Cancer_df = pd.read_csv('breast-cancer-wisconsin-data.csv', low_memory=False)


'''
Part B: Inspect loaded dataframe using describe (), Head() and Info().
'''
#breast_Cancer_Dataframe.info()
'''
    Dataset Info:
        69 entries
        33 Colunms
        Range = 0 to 568
'''
breast_Cancer_df.describe()


'''
Part C: Split the train and test sets with ratio 60:40
'''
diagnosis_map = {'M':1, 'B':-1}
breast_Cancer_df['diagnosis'] = breast_Cancer_df['diagnosis'].map(diagnosis_map)
breast_Cancer_df.drop(breast_Cancer_df.columns[[-1, 0]], axis=1, inplace=True)

Y = breast_Cancer_df.loc[:, 'diagnosis']
X = breast_Cancer_df.iloc[:, 1:]

X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

X.insert(loc=len(X.columns), column='intercept', value=1)

print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.4, random_state=42)


'''
Part D: Define the Cost function which take Weight, Input(X) and Output(Y).

Function Name: cost_function
Parameters: Weight(w), Input(x), Output(Y)
Description: Also known as as the Objective Function. We use this function to try and
             maximize to acheive our objective.

Returns: weight
'''
def cost_function(W, X, Y):
    c = 10000
    # # calculate hinge loss
    # N = X.shape[0]
    # distances = 1 - Y * (np.dot(X, W))
    # distances[distances < 0] = 0  # equivalent to max(0, distance)
    # hinge_loss = c * (np.sum(distances) / N)

    # # calculate cost
    # cost = 1 / 2 * np.dot(W, W) + hinge_loss

    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = c * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

    return cost

'''
Part E: Define the gradient of cost function.

Function Name: gradient_cost
Parameters: Weight(w), Input(x), Output(Y)
Description: Calculates the gradient cost function

Returns: gradient cost
'''
def gradient_cost(W, X_batch, Y_batch):
    c = 10000
    # N = x.shape[0]

    # distances = 1 - y * (np.dot(x, w))
    # dw = np.zeros(len(w))

    # for ind, d in enumerate(distances):
    #     if max(0, d) == 0:
    #         di = w
    #     else:
    #         di = w - (c * y[ind] * x[ind])
    #     dw += di

    # return dw

    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    if max(0, distance) == 0:
        dw += W
    else:
        dw += W - (c * Y_batch * X_batch)

    return dw

'''
Part F: Define function for Stochastic Gradient Decent and set the stopping conditions.

Function Name: stochastic_gradient_decent
Parameters: features, outputs
Description: Calculates the stochastic gradient bases the stoppage criteria of when the current
             cost hasent decreased much as compares to the previous cost.

Returns: weights
'''
def stochastic_gradient_decent(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    n = 0
    learning_rate = 0.000001

    # previous_cost = float("inf")
    # cost_threshold = 0.01

    # for epoch in range(max_epochs):
    #     X, Y = shuffle(features, outputs)

    #     for x in enumerate(X):
    #         ascent = gradient_cost(weights, X, Y)
    #         wights = weights - (learning_rate * ascent)

    #     if epoch == 2 ** n or epoch == max_epochs - 1:
    #         cost = cost_function(weights, features, outputs)
    #         print("Epoch is: {} and cost is: {}".format(epoch, cost))

    #         if abs (previous_cost - cost) < cost_threshold * previous_cost:
    #             return weights

    #         previous_cost = cost
    #         n = 1 + n

    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01

    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = gradient_cost(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = cost_function(weights, features, outputs)
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1

    return weights



#######################################################################
#  Problem 2: Titanic Dataset                                         #
#                                                                     #
#  Description: Predict the passenger is survived or not              #
#######################################################################

## Pre-processing Taken from given file.
training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
training_data.sample(5)

training_data.describe()
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

transformed_train = transform_features(training_data)
transformed_train.head()

test_data.sample(5)

transformed_test = transform_features(test_data)
transformed_test.head()

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

data_train, data_test = encode_features(transformed_train, transformed_test)
data_train.head()
data_test.head()

X_Train_Titanic = data_train.drop(['Survived'], axis=1)
Y_Train_Titanic = data_train.Survived
X_Test_Titanic = data_test
Y_Test_Titanic = data_train.Survived



#######################################################################
#                           Main Program                              #
#                  ### PART 2: SVM Implementation ###                 #
#######################################################################
def main():

    ### Question 1: Predict whether the cancer is benign or malignant
        #Titianic Dataset Information
    # '''
    #     X_train
    #     X_test
    #     y_train
    #     y_test
    # '''
    print("Training Started...")
    W = stochastic_gradient_decent(X_train.to_numpy(), y_train.to_numpy())
    print("Training Finished.")
    print("Weights are: {}".format(W))

    y_test_predicted = np.array([])

    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i]))
        y_test_predicted = np.append(y_test_predicted, yp)

    print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
    print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    print("precision on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    print("")
    print("")
    print("")
    print("")


    ## Question 2: Predict the passenger is survived or not.
        #Titianic Dataset Information
    '''
        data_train
        data_test
        X_Train_Titanic = data_train.drop(['Survived'], axis=1)
        Y_Train_Titanic = data_train.Survived
        X_Test_Titanic = data_test
        Y_Test_Titanic = data_train.Survived
    '''
    print("Titanic Training Started...")
    tW = stochastic_gradient_decent(X_Train_Titanic.to_numpy(), Y_Train_Titanic.to_numpy())
    print("Training Finished.")
    print("Weights are: {}".format(tW))

    y_test_predicted_titanic = np.array([])

    for i in range(X_Test_Titanic.shape[0]):
        yp = np.sign(np.dot(tW, X_Test_Titanic.to_numpy()[i]))
        y_test_predicted_titanic = np.append(y_test_predicted_titanic, yp)

    print("accuracy on test dataset: {}".format(accuracy_score(Y_Test_Titanic.to_numpy(), y_test_predicted_titanic)))
    print("recall on test dataset: {}".format(recall_score(Y_Test_Titanic.to_numpy(), y_test_predicted_titanic)))
    print("precision on test dataset: {}".format(recall_score(Y_Test_Titanic.to_numpy(), y_test_predicted_titanic)))

main()
