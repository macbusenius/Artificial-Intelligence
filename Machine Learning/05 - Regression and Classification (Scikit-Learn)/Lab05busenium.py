'''
Program Name: Lab04busenium.py
Class: ENSE496AD Machine Learning

Name: Mckenzie Busenius
SID: 200378076

Professor: Dr. Kin-Choong Yow
Lab Professor: Usman Munawar

Acknowledgments:
    [1] https://medium.com/@gurupratap.matharu/end-to-end-machine-learning-project-on-predicting-housing-prices-using-regression-7ab7832840ab

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


################################################################
#   Part 1: California Housing Regression Problem              #
#           using Ski Learn                                    #
################################################################
print("")
print(" *************************************************** ")
print(" ************** PART 1 - Regression **************** ")
print(" *************************************************** ")

# Part a) Load the dataset in dataframe
housing_dataframe = pd.read_csv('./housing.csv')

# Part b) Inspect loaded dataframe using describe (), Head () and Info ().
    # housing_dataframe.head(5)
    # housing_dataframe.info()
    # housing_dataframe.hist(bins=50, figsize=(15,15))
    #plt.show()

# Part c) Apply feature selection technique[optional]
housing_dataframe["income_cat"] = np.ceil(housing_dataframe["median_income"]/ 1.5)
housing_dataframe["income_cat"].where(housing_dataframe["income_cat"] < 5, 5.0, inplace=True)

# Part d) Split the train and test sets with ratio 80:20
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_dataframe, housing_dataframe['income_cat']):
    train_set = housing_dataframe.loc[train_index]
    test_set = housing_dataframe.loc[test_index]

train_set.drop(["income_cat"], axis=1, inplace=True)
test_set.drop(["income_cat"], axis=1, inplace=True)

# Part e) Apply atleast 4 different algorithms such as Random forest, SVM, ANN and etc.
#         and train your models.
housing = train_set.drop("median_house_value", axis=1)
housing = housing.drop("ocean_proximity", axis=1)
housing_test = test_set.drop("median_house_value", axis=1)
housing_test = housing_test.drop("ocean_proximity", axis=1)

#X_train
housing_Model_train = housing.drop("total_bedrooms", axis=1)
#Y_train
housing_labels_train = train_set['median_house_value']
#X_test
housing_Model_test = housing_test.drop("total_bedrooms", axis=1)
#Y_test
housing_labels_test = test_set['median_house_value']

'''
Model 1: K-fold Cross Validation
'''
print("")
print("Model 1: K-fold Cross Validation ")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
lin_reg = LinearRegression()

scores = cross_val_score(lin_reg, housing_Model_train, housing_labels_train,
                        scoring="neg_mean_squared_error", cv=10)

root_mean_squares_error_scores = np.sqrt(-scores)

print("Mean:\t\t ", root_mean_squares_error_scores.mean(), "\nStandard Deviation:", root_mean_squares_error_scores.std())

'''
Model 2: Decision Tree Regressor with K-fold cross validation
'''
print("")
print("Model 2: Decision Tree Regressor with K-fold cross validation ")
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
tree_reg = DecisionTreeRegressor()
scores = cross_val_score(tree_reg, housing_Model_train, housing_labels_train,
                        scoring="neg_mean_squared_error", cv=10)

# find root mean squared error
root_mean_squares_error_scores = np.sqrt(-scores)
print("Mean:\t\t ", root_mean_squares_error_scores.mean(), "\nStandard Deviation:", root_mean_squares_error_scores.std())

'''
Model 3: Random Forest Regressor
'''
print("")
print("Model 3: Random Forest Regressor")
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_Model_train, housing_labels_train)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
forest_scores = cross_val_score(forest_reg, housing_Model_train, housing_labels_train,
                               scoring="neg_mean_squared_error", cv=10)
root_mean_squares_error_scores = np.sqrt(-forest_scores)
print("Mean:\t\t ", root_mean_squares_error_scores.mean(), "\nStandard Deviation:", root_mean_squares_error_scores.std())


# Part f) Evaluate your 4 models on test set.
# Part g) Compare accuracy and find the best model
# Part h) Print results of prediction [on testset] after finding the best model.

    #X_train
housing_Model_train = housing.drop("total_bedrooms", axis=1)
    #Y_train
housing_labels_train = train_set['median_house_value']
    #X_test
housing_Model_test = housing_test.drop("total_bedrooms", axis=1)
    #Y_test
housing_labels_test = test_set['median_house_value']

print("")
print("Linear Regression on Test Set")
lin_reg.fit(housing_Model_train, housing_labels_train)
prediction_Linear_Regression = lin_reg.predict(housing_Model_test)
final_mse_Linear_Regression = mean_squared_error(housing_labels_test, prediction_Linear_Regression)
final_rmse_Linear_Regression = np.sqrt(final_mse_Linear_Regression)
print("Prediction Root Mean Square Error: ", final_rmse_Linear_Regression)
print("Prediction Avarage: ", np.average(prediction_Linear_Regression))
print("")

print("Tree Regression on Test Set")
tree_reg.fit(housing_Model_train, housing_labels_train)
prediction_Tree_Regression = tree_reg.predict(housing_Model_test)
final_mse_Tree_Regression = mean_squared_error(housing_labels_test, prediction_Tree_Regression)
final_rmse_Tree_Regression = np.sqrt(final_mse_Tree_Regression)
print("Prediction Root Mean Square Error: ", final_rmse_Tree_Regression)
print("Prediction Avarage: ", np.average(prediction_Tree_Regression))
print("")

print("** Best Regression Model ** Random Forest on Test Set")
forest_reg.fit(housing_Model_train, housing_labels_train)
prediction_Forest_Regression = forest_reg.predict(housing_Model_test)
final_mse_Forest_Regression = mean_squared_error(housing_labels_test, prediction_Forest_Regression)
final_rmse_Forest_Regression = np.sqrt(final_mse_Forest_Regression)
print("Prediction Root Mean Square Error: ", final_rmse_Forest_Regression)
print("Prediction Avarage: ", np.average(prediction_Forest_Regression))
print("")



################################################################
#   Part 2: Scene Dataset Classification Problem using         #
#           Ski learn liibraries                               #
################################################################
print("")
print(" ******************************************************* ")
print(" ************** PART 2 - Classification **************** ")
print(" ******************************************************* ")
print("")
# Part a) Load the dataset in dataframe
scene_dataframe = pd.read_csv('./scene.csv')

# Part b) Inspect loaded dataframe using describe (), Head () and Info ().
    # scene_dataframe.head(5)
    # scene_dataframe.info()

# Part c) Apply feature selection technique[optional]
scene_dataframe["Temp"] = np.ceil(scene_dataframe["Urban"]/ 1.5)
scene_dataframe["Temp"].where(scene_dataframe["Temp"] < 5, 5.0, inplace=True)

# Part d) Split the train and test sets with ratio 80:20
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(scene_dataframe, scene_dataframe['Temp']):
    train_set = scene_dataframe.loc[train_index]
    test_set = scene_dataframe.loc[test_index]

train_set.drop(["Temp"], axis=1, inplace=True)
test_set.drop(["Temp"], axis=1, inplace=True)

X_train = train_set.iloc[:, train_set.columns != 'Urban' ]
Y_train = train_set['Urban']

X_test = test_set.iloc[:, test_set.columns != 'Urban' ]
Y_test = test_set['Urban']

"""
# Part e) Apply atleast 4 different algorithms such as
        Logistic Regression, Naïve Bias,Random forest, SVM, ANN and  etc. and train your models.
# Part f) Evaluate your 4 models on test set.
# Part g) Compare accuracy and find the best model.
"""

'''
Model 1: Logistic Regression
'''
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, Y_train)
score_LR = logisticRegression.score(X_test, Y_test)
print("Score -- Logistic Regression Model: ", score_LR)

'''
Model 2: K-Nearest Neighbors Algorithm
'''
from sklearn.neighbors import KNeighborsClassifier
K_Neighbors_Classifier = KNeighborsClassifier(n_neighbors=5)
K_Neighbors_Classifier.fit(X_train, Y_train)
score_KNN = K_Neighbors_Classifier.score(X_test, Y_test)
print("Score -- K Neighbors Classifier Model: ", score_KNN)

'''
Model 3: Support Vector Machine
'''
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)
score_SVM = svclassifier.score(X_test, Y_test)
print("Score -- SVM: ", score_SVM)

'''
Model 4: Random Forest Classifier
'''
from sklearn.ensemble import RandomForestClassifier
randomForestClassifier = RandomForestClassifier(n_estimators=20, random_state=0)
randomForestClassifier.fit(X_train, Y_train)
score_RFR = randomForestClassifier.score(X_test, Y_test)
print("Score -- Random Forest Classifier ", score_RFR)

# Part h) Present classification report of the best model[Do Prediction(with testset) and print its classification report)
print("")
print("")

print("Best Model is Support Vector Machine (SVM) with.....")
print("")
from sklearn.metrics import classification_report
prediction_SVM = svclassifier.predict(X_test)
print(classification_report(Y_test, prediction_SVM))
