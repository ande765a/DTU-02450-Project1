# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:28:08 2019

@author: peter
"""
from Data import *
from Functions_and_classes import *
import numpy as np
import scipy as sp
import scipy.stats as st 

print(test_errors_regr)
predicted_LM = np.concatenate(hats_regr[0])
predicted_ANN = np.concatenate(hats_regr[1])
predicted_BL = np.concatenate(hats_regr[2])
true_class = np.concatenate(tests_regr)

# Test objects...
z_LM = (predicted_LM-true_class)**2
z_ANN = (predicted_ANN-true_class)**2
z_BL = (predicted_BL-true_class)**2

alpha = 0.05 # significance

# Compare the ANN with the linear model: M_ann - M_lm
# NullHypothesis: the two models are equal: M_ann - M_lm = 0
# If M_ann is better z < 0
# If M_lm is better z > 0

z_1 = z_ANN - z_LM
CI_ann_vs_lm = st.t.interval(1-alpha, len(z_1)-1, loc=np.mean(z_1), scale=st.sem(z_1)) # confidence interval
p_ann_vs_lm = st.t.cdf( -np.abs( np.mean(z_1) )/st.sem(z_1), df=len(z_1)-1)  # p-value


# Compare the ANN with the Baseline: M_ann - M_bl
# NullHypothesis: the two models are equal: M_ann - M_bl = 0
# If M_ann is better z < 0
# If M_bl is better z > 0

z_2 = z_ANN - z_BL
CI_ann_vs_bl = st.t.interval(1-alpha, len(z_2)-1, loc=np.mean(z_2), scale=st.sem(z_2)) # confidence interval
p_ann_vs_bl = st.t.cdf( -np.abs( np.mean(z_2) )/st.sem(z_2), df=len(z_2)-1)  # p-value

# Compare the Linear model with baseline: M_lm - M_bl
# NullHypothesis: the two models are equal: M_lm - M_bl = 0
# If M_lm is better z < 0
# If M_bl is better z > 0

z_3 = z_LM - z_BL
CI_lm_vs_bl = st.t.interval(1-alpha, len(z_3)-1, loc=np.mean(z_3), scale=st.sem(z_3)) # confidence interval
p_lm_vs_bl = st.t.cdf( -np.abs( np.mean(z_3) )/st.sem(z_3), df=len(z_3)-1)  # p-value

print("P_value for the null hypothesis: ANN = LM: ",p_ann_vs_lm)
print(1-alpha, "% Confidence interval for Z = E_ANN - E_LM: ", CI_ann_vs_lm)
print("")
print("P_value for the null hypothesis: ANN = BL: ",p_ann_vs_bl)
print(1-alpha, "% Confidence interval for Z = E_ANN - E_BL: ", CI_ann_vs_bl)
print("")
print("P_value for the null hypothesis: LM = BL: ",p_lm_vs_bl)
print(1-alpha, "% Confidence interval for Z = E_LM - E_BL: ", CI_lm_vs_bl)

print(z_1.mean())
print(z_2.mean())
print(z_3.mean())