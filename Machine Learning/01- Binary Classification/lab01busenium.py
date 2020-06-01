#Mckenzie Busneius
#200378076
#Lab1
# ENSE496AD

#Help from: https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%205/Implementing%2Bbinary%2Bdecision%2Btrees.md

import numpy as np
import pandas as pd
import random 


				#############################
				#			Part 1          #
				#############################

#
# Experimental Setup and Feature Engineering for Binary Classification
#

##############
# Load the Dataset
#############
# Read in the data from the lending clud
loans = pd.read_csv('lending-club-data.csv', low_memory=False)
#print(loans.columns)


##############
# Identify the target label
#############
# Identify the target label for binary classification Problem and separate the ‘risky’ and Safe loan category from Target Label.
# The target label is the bad_loans header
    # We are going to seperate the bad loans from the safe loans by creating a new colunms
    # safe_loans =  1 => safe
    # safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis=1)
#print(loans['safe_loans'])


##############
# Imput feature selection
#############
# Assign the target and select the Input features 
# Prediction target (y) 
target = 'safe_loans'

# Features --> Creadit, Income, Term
features = ['grade', 'annual_inc', 'term']

# Extract the feature columns and target column
loans = loans[features + [target]]


##############
# Class Balance
#############
safe_loans_PreBalance = loans[loans[target] == +1]
risky_loans_PreBalance = loans[loans[target] == -1]
# print("Number of safe loans(Pre Balance)  : %s" % len(safe_loans_PreBalance))
# print("Number of risky loans(Pre Balance) : %s" % len(risky_loans_PreBalance))

# print("Percentage of safe loans(Pre Balance)  :", len(safe_loans_PreBalance) / float(len(safe_loans_PreBalance) + len(risky_loans_PreBalance))) 
# print("Percentage of risky loans(Pre Balance) :", len(risky_loans_PreBalance) / float(len(safe_loans_PreBalance) + len(risky_loans_PreBalance)))

risky_loans = risky_loans_PreBalance
safe_loans = safe_loans_PreBalance[:15122]
loans_data = risky_loans.append(safe_loans)

# print("Number of safe loans = ", len(safe_loans))
# print("Number of risky loans = ", len(risky_loans))
# print("This is the combined list:")
# print((loans_data))


##############
# Hot Encoding for required labels
#############
categorical_variables = []
for feat_name, feat_type in zip(loans_data.columns, loans_data.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)
        
for feature in categorical_variables:
    
    loans_one_hot_encoded = pd.get_dummies(loans_data[feature],prefix=feature)
    loans_one_hot_encoded.fillna(0)
    #print loans_one_hot_encoded
    
    loans_data = loans_data.drop(feature, axis=1)
    for col in loans_one_hot_encoded.columns:
        loans_data[col] = loans_one_hot_encoded[col]
    
# print(loans_data.head(30244))        
# print(loans_data.columns)


##############
# Suffle, split the dataset into train and test
#############
loans_data = loans_data.sample(frac=1).reset_index(drop=True)

train_data = loans_data[:24195]
test_data = loans_data[:6048]


				#############################
				#			Part 2          #
				#############################

#
# Functions to apply the Decision Tree Implementation. Implement these all functions on Loan dataset provided.
#

##############

# Name: 		intermediate_node_num_mistakes
# Paramaters: 	labels_in_node
# Returns: 		Return the number of mistakes that the majority classifier makes.
# Descriptions:	Calculate the misclassified examples while predicting majority class to determine best feature split.

#############			
def intermediate_node_num_mistakes(labels_in_node):
    if len(labels_in_node) == 0:
        return 0    
    # Count the safe loans
    safe_loan = (labels_in_node == 1).sum()
    # Count risky loans               
    risky_loan = (labels_in_node == -1).sum()

    # Return min number of mistakes between the different loans
    return min(safe_loan, risky_loan)



##############

# Name: 		best_splitting_feature	
# Paramaters: 	data, features, target
# Returns: 		Best feature
# Descriptions:	This function find best splitting feature between the features, and data.

#############	
def best_splitting_feature(data, features, target):
	#target_values = data[target]
  
    # Loop over each feature in the feature list
    for feature in features:
        
		# Split the data into two groups:
        # The data has feature value 0 or False (we will call this the left split)	
        left_split = data[data[feature] == 0]
        
        # The data has feature value 1 or True (we will call this the right split).
        right_split = data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in both groups of data, using the function earier in program
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            
        right_mistakes = intermediate_node_num_mistakes(right_split[target])  
            
        # Ccompute the classification error
        	# Classifcation Error = Total number of mistackes(Left and Right) / Total examples
        error = (left_mistakes + right_mistakes) / float(len(data))  

        # If the computed error is smaller than the best error found so far, store this feature and its error.
        best_feature = None
        best_error = 10

        if error < best_error:
            best_feature = feature
            best_error = error
    
    return best_feature



##############

# Name: 		create_leaf	
# Paramaters: 	target_values
# Returns: 		A created leaf
# Descriptions:	This creates a leaf for the decision tree

#############
def create_leaf(target_values):    
    leaf = {
    		# True/ False
    		'is_leaf'			: True,
    		# Prediction ad the leaf node 
    		'prediction '		: None,
            # Coresponding to left tree
            'left' 				: None,
            # Coresponding to right tree
            'right' 			: None,
            # The feature that this node splits on
            'splitting_feature' : None,
    }  
   
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    if num_ones > num_minus_ones:
        leaf['prediction'] = 1        
    else:
        leaf['prediction'] = -1            

    return leaf 



##############

# Name: 		decision_tree_create
# Paramaters: 	data, features, target, current_depth, max_depth
# Returns: 		A Leaf node
# Descriptions:	Create a decision tree which follow these three stopping conditions for creating trees.

#############
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    target_values = data[target]
    print("###########################################################################")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    
    # Stopping condition 1: All data points in a node are from the same class.
    if intermediate_node_num_mistakes(target_values) == 0:
        print("Stopping condition 1 reached.")    
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    remaining_features = features[:] # Make a copy of the features.
    if remaining_features == []:  
        print("Stopping condition 2 reached.")    
        return create_leaf(target_values)    
    
    # Additional stopping condition: The max_depth of the tree. By not letting the tree grow too deep, we will save computational effort in the learning process.
    if current_depth >= max_depth:
        print("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values)


				#############################
				#			Part 3          #
				#############################

#
# Write a user defined function and apply on loan Dataset
#



#Got stuck with the instructions, didnt not know how to continues