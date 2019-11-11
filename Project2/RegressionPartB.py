# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:21:04 2019

@author: peter
"""

""" 
For regression part b) two-layer Cross validation is used - Algorithm 6.
"""
from Data import *
from Functions_and_classes import *
import sklearn.linear_model as lm
import numpy as np

lambdas = np.power(10., range(-6, 6))
hidden_units = np.arange(1, 5)

model_types = [
  ("Linear Model", [("Linear Model, lambda = {}".format(l), l, lm.Ridge(alpha=l)) for l in lambdas]),
  ("ANN", [("ANN, h_u = {}", h_u, ANN(hidden_units = h_u)) for h_u in hidden_units]),
  ("Base Line", [("Base Line", None, BaseLine())])
]

parameter_types = ["Lambda", "Hidden units", "Nothing"] # The names of the parameters, for later convenience

def reg_error_fn(y_hat, y):
  return np.power(y - y_hat, 2).mean()

test_errors_regr, hats_regr, tests_regr = twoLayerCrossValidation(model_types, parameter_types, X_regr, y_regr, error_fn = reg_error_fn, K1=10, K2 = 10)
