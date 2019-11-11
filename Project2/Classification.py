# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:11:52 2019

@author: peter
"""

from Data import *
from Functions_and_classes import *

import numpy as np


from sklearn import model_selection, tree
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
import sklearn.linear_model as lm


"""
  Below is a list of the methods used for classification.
  For each method models with different complexity parameter is chosen.
  2-Layer cross validation is used to estimate the generalization error of the
  model. The inner loop choses the optimal complexity parameter, the outer 
  calculates the test error for that optimal model on the rest of the data set
"""
lambdas = lambdas = np.arange(0.1,50,0.1) # lige ændret fra (0,20,0.1)
model_types = [
  ("Logistic Regression", [
    ("Logistic Lambda = {}".format(l),l, lm.logistic.LogisticRegression(solver='liblinear', C = 1/l ))
    for l in lambdas
  ]),
  ("Nearest Neighbour", [
    ("{}NN".format(k),k, KNeighborsClassifier(n_neighbors=k)) 
      for k in np.arange(1, 20)
  ]),
  ("Base Line", [("Base Line", None, BaseLine_Classification())])

]
  # Add baseline model

parameter_types = ["Lambda_vals","Number of neighbours", "None"] # The names of the parameters, for later convenience

def class_error_fn(y_hat, y):
  return (y_hat != y).mean()

test_errors, hats, tests = twoLayerCrossValidation(model_types, parameter_types, X, y, error_fn = class_error_fn)