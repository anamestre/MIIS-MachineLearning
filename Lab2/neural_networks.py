# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:18:00 2019

@author: Ana
"""

"""
Neural Networks: 
Train a Multi-Layer perceptron using the cross-entropy loss with l-2 regularization (weight decay penalty). 
In other words, the activation function equals the logistic function. 
Plot curves of the training and validation error as a function of the penalty strength alpha. 
How do the curves behave? Explain why.
Advice: use a logaritmic range for hyper-parameter alpha. 
Experiment with different sizes of the training/validation sets and different model parameters (network layers).
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

# https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-scratch.html

def get_data(file, test_size):
  x, y = load_svmlight_file(file)
  return train_test_split(x, y, test_size = test_size)


def plot(alpha, training, testing):
  plt.semilogx(alpha, training)
  plt.semilogx(alpha, testing)
  plt.legend(['training', 'testing'])
  plt.ylabel("Accuracy")
  plt.xlabel("Alpha")
  plt.title("Overfitting of MLP")


def train_neural_networks(x_train, y_train, x_test, y_test):
  training, testing = [], []
  alpha = np.logspace(-4, 0, num = 100, base = 10)
  alpha = np.concatenate((np.array([0.0]), alpha))
  
  for a in alpha:
    clf = MLPClassifier(activation = 'logistic', max_iter = 1000, 
                        n_iter_no_change = 10,
                        learning_rate_init=0.001,
                        alpha = a, hidden_layer_sizes = (50, 5))
    fit = clf.fit(x_train, y_train)

    training.append(clf.score(x_train, y_train))
    testing.append(clf.score(x_test, y_test))
  
  return training, testing, alpha


def build_nn(file, test_size):
    x_train, x_test, y_train, y_test = get_data(file, test_size)
    training, testing, alpha = train_neural_networks(x_train, y_train, x_test, y_test)
    plot(alpha, training, testing)    



file = "mushrooms.txt"
build_nn(file, 0.3)