# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:36:53 2019

@author: Ana
"""

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

def initialize():
    Qf2, Qf10 = 2, 10
    N = 100
    X = np.random.uniform(-1, 1, N)
    X.sort()
    #_, std = norm.fit(X)
    sigma = 0.3
    epsilon = np.random.standard_normal(N)
    return Qf2, Qf10, N, sigma, X, epsilon

def norm_aqs(aqs, Qf):
    summ = sum([aqs[q]**2/(2 * q + 1) for q in range(Qf + 1)])
    #summ = sum([1/(2*q + 1) for q in range(Qf + 1)])
    return [aqs / math.sqrt(2 * summ)]

def fun_x(x, Qf, aqs):
    return sum([aqs[0][q] * legendre(q, x) for q in range(Qf)])

def legendre(k, x):
    if k == 0: 
        return 1
    elif k == 1:
        return x
    else: 
        return((2 * k - 1) / k) * x * legendre(k - 1, x) - ((k - 1) / k) * legendre(k - 2, x)

def plot_function(vals, f, label = " ", color = "-r"):
    plt.plot(vals, f, color, label = label)
    plt.show()

# Y_n = f(x_n) + sigma * epsilon_n
# f(x) = sum(from q = 0 to Qf) (a_q) * L_q(x)
def get_y(fx, sigma, epsilon):
    y = [fx[i] + ((sigma**2) * epsilon[i]) for i in range(len(fx))]
    return y

def polRegression(x, y, Qf):
    poly_feat = PolynomialFeatures(degree = Qf, include_bias = True)
    lin_reg = LinearRegression()
    x_ = poly_feat.fit_transform(x.reshape(-1, 1))
    lin_reg.fit(x_, y)
    y_pred = lin_reg.predict(x_)
    plt.scatter(x, y)
    plt.plot(x, y_pred, c = "r")
    plt.show()
    

Qf2, Qf10, N, sigma, X, epsilon = initialize()

# Select aq independently from a standard Normal distribution
aq2 = np.random.standard_normal(Qf2 + 1)
norm_aq2 = norm_aqs(aq2, Qf2)
aq10 = np.random.standard_normal(Qf10 + 1)
norm_aq10 = norm_aqs(aq10, Qf10)

# ... rescaling so E[f^2] = 1
#norm_aq2 = preprocessing.normalize([aq2] , norm = 'l2')
#norm_aq10 = preprocessing.normalize([aq10] , norm = 'l2')

g2 = fun_x(X, Qf2, norm_aq2) # H2
g10 = fun_x(X, Qf10, norm_aq10) # H10
#plot_function(X, g2, "H2", "-r")
#plot_function(X, g10, "H10", "-b")

Y2 = get_y(g2, sigma, epsilon)
Y10 = get_y(g10, sigma, epsilon)

polRegression(X, Y2, Qf2)
polRegression(X, Y10, Qf10)