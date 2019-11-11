# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:23:50 2019

@author: peter
"""

from Data import *
from Functions_and_classes import *
import numpy as np


print(test_errors)
predicted_log = np.concatenate(hats[0])
predicted_KNN = np.concatenate(hats[1])
predicted_BL = np.concatenate(hats[2])
true_class = np.concatenate(tests)
alpha = 0.05


z_1, CI_log_vs_KNN, p_log_vs_KNN = mcnemar(true_class, predicted_log, predicted_KNN, alpha = 0.05)
z_2, CI_log_vs_BL, p_log_vs_BL = mcnemar(true_class, predicted_log, predicted_BL, alpha = 0.05)
z_3, CI_KNN_vs_BL, p_KNN_vs_BL = mcnemar(true_class, predicted_KNN, predicted_BL, alpha = 0.05)

print("P_value for the null hypothesis: Log = KNN: ",p_log_vs_KNN)
print(1-alpha, "% Confidence interval for difference in accuracy between log and KNN: ", CI_log_vs_KNN)
print("")
print("P_value for the null hypothesis: Log = BL: ",p_log_vs_BL)
print(1-alpha, "% Confidence interval for difference in accuracy between log and BL: ", CI_log_vs_BL)
print("")
print("P_value for the null hypothesis: KNN = BL: ",p_KNN_vs_BL)
print(1-alpha, "% Confidence interval for difference in accuracy between KNN and BL ", CI_KNN_vs_BL)

print("Test_statistic, theta1: ",z_1)
print("Test_statistic, theta2: ",z_2)
print("Test_statistic, theta3: ",z_3)

print(("Mean Accuracy log: ", ((predicted_log == true_class)*1).mean()))
print(("Mean Accuracy KNN: ",((predicted_KNN == true_class)*1).mean()))
print(("Mean Accuracy BL: ", ((predicted_BL == true_class)*1).mean()))
