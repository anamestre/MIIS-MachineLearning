# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:36:53 2019

@author: Ana
"""

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

def initialize():
    Qf2, Qf10 = 2, 10
    N = 100
    X = np.random.uniform(-1, 1, N)
    X.sort()
    #_, std = norm.fit(X)
    sigma = 0.1
    epsilon = np.random.standard_normal(N)
    return Qf2, Qf10, N, sigma, X, epsilon

def poly(x, Qf):
    leg_pol = legendre(Qf)
    Lk = leg_pol(x)
    return Lk

def plot_function(vals, f, label = " ", color = "-r"):
    plt.plot(vals, f, color, label = label)
    plt.legend()

def get_y(N, X, sigma, epsilon):
    y = [X[i] + (sigma * epsilon[i]) for i in range(N)]
    return y
    

Qf2, Qf10, N, sigma, X, epsilon = initialize()

g2 = poly(X, Qf2) # H2
g10 = poly(X, Qf10) # H10

plot_function(X, g2, "H2", "-r")
plot_function(X, g10, "H10", "-b")

# Select aq independently from a standard Normal distribution
aq2 = np.random.standard_normal(Qf2)
aq10 = np.random.standard_normal(Qf10)

# ... rescaling so E[f^2] = 1
norm_aq2 = preprocessing.normalize([aq2] , norm = 'l2')
norm_aq10 = preprocessing.normalize([aq10] , norm = 'l2')

Y = get_y(N, X, sigma, epsilon)