# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:07:17 2019

@author: peter
"""
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as st 




# Create two Layer Cross Validation function
def twoLayerCrossValidation(model_types, parameter_types, X, y, error_fn, K1 = 10, K2 = 10):
  test_errors = np.zeros((K1,len(model_types)*2))
#  best_parameters = np.zeros((K1,len(model_types)))
  y_hat_list = []
  y_test_list = []
  for i in range(len(model_types)):
    y_hat_list.append([])

  # The two layer cross-validation algorithm
  K1fold = model_selection.KFold(n_splits=K1, shuffle = True)
  for i, (par_index, test_index) in enumerate(K1fold.split(X, y)):
    print("Outer fold {} of {}".format(i+1,K1))

    K2fold = model_selection.KFold(n_splits=K2, shuffle = True)

    # Saves D_par and D_test to allow later statistical evaluation
    X_par = X[par_index, :]
    y_par = y[par_index]

    X_test = X[test_index, :]
    y_test = y[test_index]
    y_test_list.append(y_test)
    for m, (model_type, models) in enumerate(model_types): # Iterate over the three methods chosen for classification    
      val_errors = np.zeros((K2, len(models)))

      # Inner cross validation loop
      for j, (train_index, val_index) in enumerate(K2fold.split(X_par, y_par)):
        X_train = X_par[train_index, :]
        y_train = y_par[train_index]

        X_val = X_par[val_index, :]
        y_val = y_par[val_index]

        # Test modeltype and calculate validation error for each model of the three methods
        for k, (name, parameter, model) in enumerate(models):
          model.fit(X_train, y_train)

          y_hat = model.predict(X_val)
          val_errors[j, k] = len(X_val) / len(X_par) * error_fn(y_hat, y_val)

      # Finds the optimal model
      inner_gen_errors = val_errors.sum(axis=0)
      best_model_index = np.argmin(inner_gen_errors)
      best_model_name, best_model_parameter, best_model = models[best_model_index] # Determines optimal model
      if name == 'Base Line':
        model.fit(X_par,y_par)
        y_hat = np.ones(len(y_test)) * model.predict(X_test)
        

      else:

        best_model.fit(X_par, y_par)
        y_hat = best_model.predict(X_test)
      y_hat_list[m].append(y_hat.squeeze())
      
      test_errors[i,m*2+1] = error_fn(y_hat, y_test)  # Lists test_erros for each method and each outer fold
      test_errors[i,m*2] = best_model_parameter # List the best parametertype belonging to test-error


  test_errors_folds = pd.DataFrame.from_records(data = test_errors, 
                                                columns=sum([[parameter_types[i],model_types[i][0]] for i in range(len(model_types))],[]))

  return test_errors_folds, y_hat_list, y_test_list

"""
Baseline for classification and regression
"""
# Baseline for classification
class BaseLine_Classification:
  def fit(self, X, y):
    self.bincount = np.bincount(y.astype(int)).argmax()
  
  def predict(self, X):
    return self.bincount

# Baseline for regression
class BaseLine:
  def fit(self, X, y):
    self.mean = y.mean()
  
  def predict(self, X):
    return self.mean

"""
Python class ANN for regression and CV
"""

class ANN:
  def __init__(
    self,
    hidden_units, 
    criterion = nn.MSELoss(),
    tolerance = 1e-6,
    optimizer = lambda params: torch.optim.SGD(params, lr = 1e-2),
    
  ):
    self.criterion = criterion
    self.optimizer = optimizer
    self.max_iter = 1000
    self.tolerance = tolerance
    self.hidden_units = hidden_units

  def fit(self, X, y):
    X = torch.Tensor(X)
    y = torch.Tensor(y).reshape((-1, 1))

    self.model = nn.Sequential(
      #nn.Linear(X.shape[1], y.shape[1])
      nn.Linear(X.shape[1], self.hidden_units),
      nn.Tanh(),
      nn.Linear(self.hidden_units, y.shape[1])
    )

    print("Starting training.")
    optimizer = self.optimizer(self.model.parameters())
    old_loss = math.inf
    loss_history = []
    for i in range(self.max_iter):
      optimizer.zero_grad()

      y_hat = self.model(X)
      loss = self.criterion(y_hat, y)
      loss.backward()
      loss_value = loss.item()
      loss_history.append(loss_value)

      p_delta_loss = np.abs(loss_value - old_loss) / old_loss
      if p_delta_loss < self.tolerance: break
      old_loss = loss_value
      
      optimizer.step()
    print("Training done.")
    plt.plot(loss_history)
    plt.show()
    
  def predict(self, X):
    X = torch.Tensor(X)
    y = self.model.forward(X)
    return y.detach().numpy()

def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1) * (Q-1)
    q = (1-Etheta) * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in st.beta.interval(1-alpha, a=p, b=q) )

    p = 2*st.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    # print("Result of McNemars test using alpha=", alpha)
    # print("Comparison matrix n")
    # print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    # print("Approximate {} confidence interval of theta: [thetaL,thetaU] = ".format(1-alpha), CI)
    # print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    # thetahat = 2*thetahat-1
    return thetahat, CI, p