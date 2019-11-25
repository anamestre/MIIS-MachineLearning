# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:07:24 2019

@author: Ana
"""

"""
Support Vector Machines: 
Run SVM to train a classifier, using radial basis as kernel function. 
Apply cross-validation to evaluate different combinations of values of the model hyper-parameters 
(box constraint C and kernel parameter Y). 
How sensitive is the cross-validation error to changes in C and Y? 
Choose the combination of C and Y that minimizes the cross-validation error, 
train the SVM on the entire dataset and report the total classification error.
Advice: use a logaritmic range
"""
# https://chrisalbon.com/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel/

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import numpy as np


def get_data(file):
    #x, y = load_svmlight_file(file)
    #return train_test_split(x, y, test_size = test_size)
    return load_svmlight_file(file)


def train_svm(x, y, C, gamma):
    svm = SVC(kernel = 'rbf', C = C, gamma = gamma)
    svm.fit(x, y)
    return svm


def test_error(svm, x, y):
    y_pred = svm.predict(x)
    print("Accuracy:", accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))


def build_svm(file, x, y, C, gamma):
    svm = train_svm(x, y, C, gamma)
    test_error(svm, x, y)


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
def apply_cross_validation(x, y):
    #C_range = np.geomspace(.01, 1000, 20)
    C_range = [1, 10, 100, 1000]
    #gamma_range = np.geomspace(.00001, 1000, 20)
    gamma_range = [1e-3, 1e-4]
    param_grid = dict(gamma = gamma_range, C = C_range)
    cross_validate = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 20)
    grid = GridSearchCV(SVC(), param_grid = param_grid, cv = cross_validate, verbose = 10)
    grid.fit(x, y)
    print("The best parameters", (grid.best_params_, grid.best_score_))
    return grid.best_params_

file = "cod-rna.txt"
x, y = load_svmlight_file(file)
C, gamma = apply_cross_validation(x, y)
build_svm(file, x, y, C, gamma)