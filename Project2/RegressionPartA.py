# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:19:03 2019

@author: peter
"""
from Data import *
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
import matplotlib.pyplot as plt


K1 = 10
lambdas = np.power(10., range(-5, 9))

K1fold = model_selection.KFold(n_splits=K1, shuffle = True)

errors = np.zeros((K1, len(lambdas)))
weights = np.zeros((K1, len(lambdas), X_regr.shape[1] + 1))

for i, (train_index, test_index) in enumerate(K1fold.split(X, y)):
  X_regr_train = X_regr[train_index]
  y_regr_train = y_regr[train_index]

  X_regr_test = X_regr[test_index]
  y_regr_test = y_regr[test_index]

  
  for j, l in enumerate(lambdas):
    model = lm.Ridge(alpha=l)
    model.fit(X_regr_train, y_regr_train)
    y_hat = model.predict(X_regr_test)
    error = ((y_regr_test - y_hat) ** 2).mean()
    errors[i, j] = (X_regr_train.shape[0] / X_regr.shape[0]) * error
    weights[i, j, 0] = model.intercept_
    weights[i, j, 1:] = model.coef_


gen_errors = errors.sum(0)
best_gen_error_index = np.argmin(gen_errors)
best_lambda = lambdas[best_gen_error_index]
mean_weights = weights.mean(0)
best_mean_weights = mean_weights[best_gen_error_index]
best_fold = np.argmin(errors[:, best_gen_error_index])
best_weights = weights[best_fold, best_gen_error_index]


plt.figure(2, figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.grid()
plt.semilogx(lambdas, mean_weights, ".-")
plt.xlabel('Regularization factor')
plt.ylabel('Mean coefficient values')
plt.legend(["Intercept", "sbp", "tobacco", "ldl", "adiposity",	"famhist", "obesity", "alcohol", "age", "chd"])


plt.subplot(1, 2, 2)
plt.loglog(lambdas, gen_errors, ".-")
plt.xlabel('Regularization factor')
plt.ylabel('Estimated generalization error')
plt.grid()
plt.show()


# pd.DataFrame(data=np.array([lambdas, gen_errors]).T, columns=["Lambda", "Gen. error"])


print("Best mean weights")
print(best_mean_weights)

print("Best weights")
print(best_weights)
print(np.min(gen_errors))