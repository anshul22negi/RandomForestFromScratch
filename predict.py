import matplotlib.pyplot as plt 
import numpy as np
from random_forest_bagging import random_forest
from hard_voting_prediction import hard_voting_prediction
from data.data_processing_titanic import X_train, y_train, X_test, y_test
from utils.error_score import error_score

n_estimators = [1, 10, 50, 100, 250, 500, 750, 1000]
errors = list()
for n_estimator in n_estimators:
    forest = random_forest(X_train, y_train, n_estimator)
    predictions = hard_voting_prediction(X_test, forest)
    errors.append(error_score(predictions, y_test))


plt.plot(n_estimators, errors)
plt.title("No. of trees vs test error")
plt.show()