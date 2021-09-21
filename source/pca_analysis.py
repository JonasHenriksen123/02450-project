#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 09:32:34 2021

@author: filippo
"""
# %%
#
# CREATE MATRIX DATA 
#

import numpy as np
import xlrd
# from sklearn.preprocessing import OneHotEncoder


# Load xls sheet with data
# doc = xlrd.open_workbook('../../first_assignment/forestfires1.xls').sheet_by_index(0)
doc = xlrd.open_workbook('/home/filippo/Documents/machine_learning/first_assignment/forestfires1.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(0, 0, 12)
# print(attributeNames)

# ONE-OUT-OF-k_ENCODINGS
month = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
monthDict = dict(zip(month,range(1, len(month)+1)))
week = ['mon','tue','wed','thu','fri','sat','sun']
weekDict = dict(zip(week, range(1, len(week)+1)))
# enc = OneHotEncoder(handle_unknown='ignore')


# month
data = doc.col_values(2,1)
values = np.array(data)
monthEncoded = []
for value in values:
    monthEncoded.append(monthDict[value])
monthEncoded = np.array(monthEncoded)
# week
data = doc.col_values(3,1)
values = np.array(data)
weekEncoded = []
for value in values:
    weekEncoded.append(weekDict[value])
weekEncoded = np.array(weekEncoded)

X_first = np.empty((517, 2))
for i, col_id in enumerate(range(2)):
    X_first[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
X_second = np.stack((monthEncoded, weekEncoded), axis= 1)

# Extract class names to python list,
# then encode with integers (dict)
labelsX = doc.col_values(0,1)
labelsY = doc.col_values(1,1)
classLabels = []
for el in range(len(labelsX)):
    classLabels.append(tuple([labelsX[el], labelsY[el]]))
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(labelsX))))

# Extract vector y, convert to NumPy array 
# y indicates to which label each line in the table coorespond o
y = np.asarray([classDict[value] for value in classLabels]) 

# # Preallocate memory, then extract excel data to matrix X
X = np.empty((517, 8))
for i, col_id in enumerate(range(4, 12)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 518))

# Unify the two matrices
X = np.hstack((X_first, X_second, X))

# # Compute values of N, M and C.
N,M = X.shape
C = len(classNames)

# %%
#
# VISUALIZE DATA
#
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Data attributes to be plotted
i = 0
j = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line
# plot(X[:, i], X[:, j], 'o')

# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
title('ForestFires data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

# legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()

# %%
#
# VISUALIZE VARIANCE
#

import matplotlib.pyplot as plt
from scipy.linalg import svd

# Subtract mean value from data
#  axis = 0 : calculate the mean of the column
X_tilda = X - np.ones((N,1)) * X.mean(axis=0)
X_tilda = X_tilda * (1 / np.std(X_tilda, 0))
# PCA by computing SVD of Y
U, S, Vh = svd(X_tilda, full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# %%
#
# VISUALIZE PCA
#

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = X @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('ForestFires data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
# legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


# %%
#
# VISUALIZE COEFFICIENTS

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2,3,4,5,6,7,8]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()

# Inspecting the plot, we see that the 2nd principal component has large
# (in magnitude) coefficients for attributes A, E and H. We can confirm
# this by looking at it's numerical values directly, too:
print('PC2:')
print(V[:,1].T)

# How does this translate to the actual data and its projections?
# Looking at the data for water:

# Projection of water class onto the 2nd principal component.
all_water_data = X_tilda[y==4,:]

print('First water observation')
print(all_water_data[0,:])

# Based on the coefficients and the attribute values for the observation
# displayed, would you expect the projection onto PC2 to be positive or
# negative - why? Consider *both* the magnitude and sign of *both* the
# coefficient and the attribute!

# You can determine the projection by (remove comments):
print('...and its projection onto PC2')
print(all_water_data[0,:] @ V[:,1])
# Try to explain why?
