'''
Program Name: Lab00busenium.py
Class: ENSE496AD Machine Learning

Name: Mckenzie Busenius
SID: 200378076

Professor: Dr. Kin-Choong Yow
Lab Professor: Usman Munawar

Acknowledgments:
    [1] https://github.com/tuanavu/coursera-university-of-washington/blob/master/
        machine_learning/2_regression/assignment/week1/week-1-simple-regression-assignment-exercise.ipynb

    [2] Thanks to Lab Professor, Usman Munawar
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################################################
#   Part 1: Write user defined generic functions which accept  #
#           two parameters (numbers x and y) and return        #
#           results.                                           #
################################################################
print (' ***** QUESTION 1 *****')
calculationValues = []

def add(value1, value2):
 sumTemp = value1 + value2
 calculationValues.append(sumTemp)
 return str(sumTemp)


def subtract(value1, value2):
  difference = value1 - value2
  calculationValues.append(difference)
  return str(difference)


def multiply(value1, value2):
  product = value1 * value2
  calculationValues.append(product)
  return str(product)



def divide(value1, value2):
  divided = value1 / value2
  calculationValues.append(divided)
  return str(divided)


#Calling Fuctions
print(add(2,3))
print(subtract(3,2))
print(multiply(3,1))
print(divide(3,1))

#Print the array of values
print(calculationValues)


##Function to find the smallest number within the array.
def smallestNumber(someArray):
	someArray.sort()
	print(someArray)
	print("Smallest Value = ", someArray.pop(0))
	print("Largest Value = ", someArray.pop(len(someArray) - 1))

smallestNumber(calculationValues)


################################################################
#   Part 2: Write a user defined generic function within
#           range (0-N) to print
################################################################
print ('\n')
print (' ***** QUESTION 2 *****')

array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def forLoopOdd(array):
    odd=[]
    even=[]

    for i in array:
        if(i%2!=0):
            odd.append(i)
        else:
            even.append(i)
    return str(odd)

def forLoopEven(array):
    odd=[]
    even=[]

    for i in array:
        if(i%2!=0):
            odd.append(i)
        else:
            even.append(i)
    return str(even)

print("List of odd number is:")
print(forLoopOdd(array))

print("List of even number is:")
print(forLoopEven(array))


################################################################
#   Part 3: File handling
################################################################
print ('\n')
print (' ***** QUESTION 3 *****')

file = open("m_learn.txt", "w")
file.write("Question 1: ")
file.write('\n')

file.write(add(2,3))
file.write('\n')

file.write(subtract(3,2))
file.write('\n')

file.write(multiply(3,1))
file.write('\n')

file.write(divide(3,1))
file.write('\n')
file.write('\n')
file.close()

file = open("m_learn.txt", "a")
file.write("Question 2 : ")
file.write('\n')

file.write(forLoopOdd(array))
file.write('\n')

file.write(forLoopEven(array))
file.close()


print("File Reading and File Printing: ")
file = open("m_learn.txt", "r")
fileContents = file.read()
print(fileContents)
file.close()


################################################################
#   Part 4: Use the dataset “weight-height.csv” and do below
#           activities. Dataset is available in assignment
#           package.
################################################################
print ('\n')
print (' ***** QUESTION 4 *****')

weight_Height = pd.read_csv('weight-height.csv', low_memory=False)
#weight_Height.info()

height = weight_Height['Height']
weight = weight_Height['Weight']

plt.scatter(height, weight, color='r')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('scatter plot - Weight as Function of Height')
plt.show()


male_height = weight_Height['Height'][0:5000]
female_Height = weight_Height['Height'][5000:10000]

plt.hist(male_height, bins=7, alpha=0.5, label='Male', color='r')
plt.hist(female_Height, bins=7, alpha=0.5, label='Female', color='b' )
plt.axvline(linewidth=4, color='g')
plt.legend(loc='upper right')
plt.xlabel('Height for males and for females')
plt.ylabel('Frequency')
plt.title('Histogram of the heights for males and for females')
plt.show()


################################################################
#   Part 5: Use dataset [kc_house_data.csv].Apply closed form
#           solution to find the intercept and slope for simple
#           linear regression.
#
#   ***** USED help from [1] in Acknowledgments ******
################################################################
print ('\n')
print (' ***** QUESTION 5 *****')

# Output = price
# Input = yr_renovated


house_Data = pd.read_csv('kc_house_data.csv', low_memory=False)
#house_Data.info()

def slope_Calc(input_Feature, output):
    x = input_Feature
    y = output
    N = len(x)

    SumY_X = (y * x).sum()
    X_Sq = (x * x).sum()
    Y_X_ByN = (y.sum() * x.sum()) / N
    X_X_ByN = (x.sum() * x.sum()) / N

    slope = (SumY_X - Y_X_ByN) / (X_Sq - X_X_ByN)

    return slope

def intercept_Calc(input_Feature, output):
    x = input_Feature
    y = output
    N = len(x)

    xMean = x.mean()
    yMean = y.mean()

    intercept = yMean - (slope_Calc(input_Feature, output) * xMean)

    return intercept

def prediction_Calc(input_Feature, intercept, slope):
    prediction = intercept + (slope * input_Feature)
    return prediction

def error_Calc(input_Feature, output, intercept, slope):
    prediction = intercept + (slope * input_Feature)
    residuals = output - prediction_Calc(input_Feature, intercept, slope)
    error = (residuals * residuals).sum()

    return error

input_Feature = house_Data['yr_renovated']
output = house_Data['price']

shuffle_df = house_Data.sample(frac=1)
train_size = int(0.8 * len(house_Data))

train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]
train_setInput = train_set['yr_renovated']
train_setOutput = train_set['price']
test_setInput = train_set['yr_renovated']
test_setOutput = train_set['price']


slope = slope_Calc(input_Feature, output)
intercept = intercept_Calc(input_Feature, output)
prediction = prediction_Calc(input_Feature, intercept, slope)

errorTrain = error_Calc(train_setInput, train_setOutput, intercept, slope)
errorTest = error_Calc(test_setInput, test_setOutput, intercept, slope)


# Y' = predictoion ---> Y axis
# X = Input_feture ---> X axis
plt.plot(input_Feature, prediction, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.xlabel('Year Renovated')
plt.ylabel('Prediction')
plt.title('Linear Regresion')
plt.show()


xAxis = ["Error Train", "Error Test"]
yAxis = [errorTrain, errorTest]
plt.bar(xAxis, yAxis)
plt.show()

