# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:54:49 2019

@author: peter
"""
from sklearn.linear_model import LogisticRegression
from Data import *
from sklearn.metrics import accuracy_score
## OPGAVE 5 ##
opt_lambda = 100
# Training logistic regression model with optimal lambda
mdl = LogisticRegression(penalty='l2', C=opt_lambda)    
mdl.fit(X, y)
w_est = mdl.coef_[0]
predicted_classes = mdl.predict(X)
accuracy = accuracy_score(y.flatten(),predicted_classes)

print(accuracy)
print(w_est)