'''
Program Name: Lab02busenium.py
Class: ENSE496AD Machine Learning

Name: Mckenzie Busenius
SID: 200378076

Professor: Dr. Kin-Choong Yow
Lab Professor: Usman Munawar

Acknowledgments:
    [1]  MachineLearningMastery.com --> Naive Bayes Classifier by Jason Brownlee
        https://machinelearningmastery.com/naive-bayes-classifier-scratch-python

    [2] Shayan Khan from University of Regina --> Helped with Part 3 of the Lab
'''

from math import sqrt
from math import pi
from math import exp
import pandas as pd
import numpy as np


trainData = [[3.39,2.33,0], [7.42,4.69,1],[5.74,3.53,1],
             [9.17,2.51,1], [3.11,1.78,0],[1.34,3.36,0],
             [3.58,4.67,0],[2.28,2.86,0],[7.79,3.42,1],
             [7.93,0.79,1]]

testData = [[1.39,2.33,0], [3.42,2.69,1],[4.74,5.53,1],
            [4.17,2.51,1],[2.11,3.78,0],[2.34,5.36,0],
            [2.58,3.67,0],[1.28,4.86,0],[6.79,2.42,1],
            [6.93,2.79,1]]

################################################################
#   Part 1: Implementation of Naive Bayes for classification   #
################################################################

'''
Function Name: seperate_dataset_by_class
Parameters: dataset
Description: Separate the given dataset by class and save them to dictionary
Returns: separted dictionary with two list to store the two different classes
'''
def seperate_dataset_by_class(dataset):
    separatedList = dict()
    for i in range(len(dataset)):
        # Iterating over the dataset and obtaining the label value for each data line in the dataset
        dataLine = dataset[i]
        LabelValue = dataLine[-1]

        # Taking the dataset, checking if it is in the dictionary, then creating a list to store the two classes in differnt list
        if (LabelValue not in separatedList):
            separatedList[LabelValue] = list()
        separatedList[LabelValue].append(dataLine)

    return separatedList


'''
Function Name: mean
Parameters: values
Description: calculates the mean
Returns: mean of the values passed
'''
def meanCalc(values):
    mean = 0
    for x in values :
        mean = mean + float(x)
    mean = mean / float(len(values))

    return mean


'''
Function Name: standard_deviation
Parameters: values
Description: calculates the mean
Returns: standard deviation of the values passed
'''
def standard_deviation(values):
    N = len(values)
    if N <= 1:
        return 0.0

    mean, stdDev = meanCalc(values), 0.0

    for x in values:
        stdDev += (float(x) - mean)**2

    stdDev = sqrt(stdDev / float(N-1))

    return stdDev


'''
Function Name: summarize_dataset_by_class
Parameters: dataset
Description: performs calculations on the dataset for mean, stdDev, and lenght
Returns: summaries

**** USED the function from Acknowledgments [1] ***
    "The first trick is the use of the zip() function that will aggregate elements from each provided argument. We pass in the dataset to the zip() function with the * operator that separates the dataset (that is a list of lists) into separate lists for each row. The zip() function then iterates over each element of each row and returns a column from the dataset as a list of numbers. A clever little trick."
'''
def summarize_dataset(dataset):
    summaries = [(meanCalc(column), standard_deviation(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])

    return summaries


'''
Function Name: summarize_by_class
Parameters: dataset
Description: Finds the mean and std deviation for each class
Returns: summaries

**** USED the function from Acknowledgments [1] ***
'''
def summarize_by_class(dataset):
    separated = seperate_dataset_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)

    return summaries


'''
Function Name: gaussian_probability
Parameters: inputSample, mean, stdDeviation
Description: Function to estimate the Gaussian Probability Distribution
             Function of Samples.
Returns: gaussianProbability
'''
def gaussian_probability(inputSample, mean, stdDeviation):
    ## WHERE ##
        # input = sample
        # Pi = Pi
        # Sigma = STD Deviation
        # U = mean
    leftSide = (1 / sqrt(2 * pi * (stdDeviation)**2))
    rightSide = exp(-(inputSample - mean)**2 / (2 * stdDeviation**2))
    gaussianProbability = leftSide * rightSide

    return gaussianProbability


'''
Function Name: probabilities_of_classes
Parameters: summaries, row
Description: Finds the probabilities of classes for a sample(test data provided)
Returns: probability for a row

**** USED the function from Acknowledgments [1] ***
'''
def probabilities_of_classes(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)

        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= gaussian_probability(row[i], mean, stdev)

    return probabilities



#################################################
#   Part 2: Logistic Regression Implementation  #
#################################################

'''
Function Name: sigmoid_function
Parameters: x, weight
Description: Calculates the Z factor alone with the sigmoid function
Returns: sigmoid function
'''
def sigmoid_function(x, weight):
    linearRegression = weight[0]
    for i in range(len(x) -1):
        linearRegression += weight[i + 1] * x[i]

    sigmoidFunction = 1 / (1 + exp(-linearRegression))

    return sigmoidFunction


'''
Function Name: maximum_log_likelihood_estimation
Parameters: input, output, weight
Description: Calculates the MLE
Returns: MLE
'''
def maximum_log_likelihood_estimation(input, output, weight):
    z = sigmoidFunction(x, weight)
    logLikelihoodEstimate = y * z - math.log10(1 + math.exp(z))
    return logLikelihoodEstimate


'''
Function Name: gradient_ascent
Parameters: input, output, prediction
Description: Calculates the gradiant ascent
Returns: gradiant ascent
'''
def gradient_ascent(input, output, prediction):
    gradientAscent = input.dot((output - prediction))
    return gradientAscent


'''
Function Name: update_weights
Parameters: weights, learningRate, gradient
Description: Updates weights acording to the learning rate
Returns: weights
'''
def update_weights(weights, learningRate, gradient):
    weights[0] = weights[0] + learningRate * gradient[0][0]
    weights[1] = weights[0] + learningRate * gradient[0][1]

    return weights


######################################################################
#   Part 3: Implementation of Telco dataset for logistic regression  #
######################################################################

'''
Function Name: logistic_regression
Parameters: None
Description: The main
Returns: None

**** USED the function from Acknowledgments [2] ***
'''
def logistic_regression():
    # Load the Dataset
    telcoDataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", low_memory=False)

    # Take any 2(numeric) important features as input for this problem from dataset such as
    # monthly charges and tenure.
    inputFeatureMonthlyCharges = telcoDataset["MonthlyCharges"]
    inputFeatureTenure = telcoDataset["tenure"]

    # Find the Target Label (output) from the dataset. Which will be considered as output(Y)
    telcoDataset["Churn"] = telcoDataset["Churn"].replace(["Yes", "No"], [1, 0])

    # Initilization of the weights and set the learning rate of 0.01
    weights = [1.0, 1.0]
    learningRate = 0.01
    print("Starting with weights of ", weights)

    # Set the number of Iteration = 200 and apply the Logistic regression using above functions
    # and predict the output and accuracy of model.
    for i in range(200):
        inputs = [inputFeatureMonthlyCharges[i], inputFeatureTenure[i]]
        inputs = np.array([inputs])
        outputs = telcoDataset["Churn"][i]
        prediction = sigmoid_function(inputs, weights)
        gradient = gradient_ascent(inputs, outputs, prediction)
        weights = update_weights(weights, learningRate, gradient)

    print("After 200 iterations the new weights are: ", weights)


# Main function
def main():
    logistic_regression()

if __name__ == '__main__':
    main()
