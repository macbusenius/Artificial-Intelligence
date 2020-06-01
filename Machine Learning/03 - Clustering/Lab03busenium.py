'''
Program Name: Lab03busenium.py
Class: ENSE496AD Machine Learning

Name: Mckenzie Busenius
SID: 200378076

Professor: Dr. Kin-Choong Yow
Lab Professor: Usman Munawar

Acknowledgments:
    [1] https://mmuratarat.github.io/2019-07-23/kmeans_from_scratch

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from copy import deepcopy


################################################################
#   Part 1: Experimental Setup                                 #
################################################################

# Part a) Load the dataset in the dataframe
clients_Dataset = pd.read_csv('Company_Client.csv', low_memory=False)

# Part b) Inspect loaded dataframe using describe(), Head() and Info(). Give your observation
clients_Dataset.describe()
    # Obervations:
        # Tell us the basic statisics for the dataset Mean, std, min and max.

# Part c) Plot the Income Annually and Score of Spending with clients and explain if there is any similarity in plot.
plot = clients_Dataset.plot(x =' Income Annually (k$)', y=' Score of Spending (1-100)', kind = 'scatter')
#plt.show()
    # We clearly see 5 clusters of data, giving us a K=5. The data simililarity is what provides the region for analysis.

# Part d) Separate the male and female and present the count of both ‘Male’ and ‘Female’ in histogram.


# Part e) Finally, take any two features like income and spending score. Store these features as input ‘X’
X = clients_Dataset.iloc[:, [3, 4]].values
#print(X)


#######################################################################
#   Part 2: Implementation of K Means Clustering for classification   #
#######################################################################

# Part a) Find number of training samples and number of features based on your input ‘X’ and store them into ‘m’ and ‘n’ respectively.
#Traning Samples
m = X.shape[0]

#Freatures
n = X.shape[1]

# Part b) Set the number of iterations arbitrarily. It is better to take least one and then keep increasing till the solution becomes converged.
number_Of_Iterations = 100

# Part c) Set the intended clusters ‘you would like to develop’ in your study. Such as k=3, 4, 5 etc.
K = 5


'''
****
**** Heavely USED the code from Acknowledgments [1] for parts d - g ****
****
'''


# Part d) Initialize the centroids randomly with dimension (nxk) from the data points.
mean = np.mean(X, axis = 0)
std = np.std(X, axis = 0)
centers = np.random.randn(K,n)*std + mean

# Part e) Compute the distance of each data point (m) in the dataset from the centroids initialized in part d.
centers_old = np.zeros(centers.shape)
centers_new = deepcopy(centers)
clusters = np.zeros(m)
distances = np.zeros((m,K))
error = np.linalg.norm(centers_new - centers_old)

# Part g)
while error != 0:
    # Measure the distance to every center
    for i in range(K):
        distances[:,i] = np.linalg.norm(X - centers_new[i], axis=1)

    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    centers_old = deepcopy(centers_new)

    # Calculate mean for every cluster and update the center
    for i in range(K):
        centers_new[i] = np.mean(X[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
centers_new


# Part h) Plot the unlabeled data using ‘X’ (defined earlier).
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(X[:, 0], X[:,1], label=labels[k])


# Part i) Plot the clustered data using different colors for each cluster. In addition, plot clusters on the same plot as well.
plt.scatter(centers_new[:, 0],centers_new[:, 1],s=300,c='Red',label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()
