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
import statistics

def initialize():
    Qf2, Qf10 = 2, 10
    N = 100
    X = np.random.uniform(-1, 1, N)
    X.sort()
    #_, std = norm.fit(X)
    sigma = 0.2
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
    #y = [fx + ((sigma) * epsilon[i]) for i in range(Qx)]
    y = fx + sigma * epsilon
    # y = [fun_x(x[i], Qf, aqs) + ((sigma) * epsilon[i]) for i in range(Qf)]
    return y

def polRegression(x, y, Qf, color = "r", title=""):
    x_reshape = x.reshape(-1, 1)
    poly_feat = PolynomialFeatures(degree = Qf, include_bias = True)
    lin_reg = LinearRegression()
    x_ = poly_feat.fit_transform(x_reshape)
    lin_reg.fit(x_, y)
    y_pred = lin_reg.predict(x_)
    plt.scatter(x_reshape, y, label = "points")
    plt.plot(x, y_pred, c = color, label = "h" + str(Qf))
    if title != "":
        plt.title(title, fontsize = 16)
    plt.legend()
    plt.show()
    return x, y_pred, lin_reg, poly_feat
    
def polRegression_models(x, Qf, sig, eps, n):
    aq = np.random.standard_normal(Qf + 1)
    norm_aq = norm_aqs(aq, Qf)
    
    g = fun_x(x, Qf, norm_aq)
    Y = get_y(g, sig, eps)
    
    title = "N = " + str(N) + ", Sigma = " + str(sig) + ", Qf = " + str(Qf)
    title2 = "H2 - " + title
    title10 = "H10 - " + title

    x2, y_pred2, lin_reg2, poly_feat2 = polRegression(x, Y, 2, "r", title2)
    x10, y_pred10, lin_reg10, poly_feat10 = polRegression(x, Y, 10, "g", title10)
    
    # E_out:
    x_ = np.random.uniform(-1, 1, n)
    x_.sort()
    y_ = fun_x(x_, Qf, norm_aq)
    x_reshape = x_.reshape(-1, 1)
    y2_hat = lin_reg2.predict(poly_feat2.fit_transform(x_reshape))
    y10_hat = lin_reg10.predict(poly_feat10.fit_transform(x_reshape))
    
    plt.scatter(x_, y_, label = "points")
    plt.title(title, fontsize = 16)
    plt.plot(x_, y2_hat, c = "r", label = "h2")
    plt.plot(x_, y10_hat, c = "g", label = "h10")
    plt.legend()
    plt.show()
    
    E_out2  = np.power(y_ - y2_hat, 2).mean()
    E_out10 = np.power(y_ - y10_hat, 2).mean()
    
    return E_out2, E_out10
    

def print_different_errors():
    Ns = np.arange(50, 121, 50)
    sigmas = np.arange(0.5, 1.05, 0.3)
    Qfs = np.arange(5, 16, 5)
    it = 1
    for n in Ns:
        for qf in Qfs:
            for sig in sigmas:
                err2_total, err10_total = [], []
                for i in range(it):
                    x = np.random.uniform(-1, 1, n)
                    x.sort()
                    eps = np.random.standard_normal(n)
                    err2, err10 = polRegression_models(x, qf, sig, eps, n)
                    err2_total.append(err2)
                    err10_total.append(err10)
                err2_mean = statistics.mean(err2_total)
                err10_mean = statistics.mean(err10_total)
                overfit = err10_mean - err2_mean
                print("For N =", n, "qf =", qf, "sigma =", sig, "error2 =", err2_mean, "error10=", err10_mean, "overfit =", overfit)
                
                    

Qf2, Qf10, N, sigma, X, epsilon = initialize()

# Select aq independently from a standard Normal distribution
aq2 = np.random.standard_normal(Qf2 + 1)
norm_aq2 = norm_aqs(aq2, Qf2)
aq10 = np.random.standard_normal(Qf10 + 1)
norm_aq10 = norm_aqs(aq10, Qf10)

g2 = fun_x(X, Qf2, norm_aq2) # H2
g10 = fun_x(X, Qf10, norm_aq10) # H10

Y2 = get_y(g2, sigma, epsilon)
Y10 = get_y(g10, sigma, epsilon)

polRegression(X, Y2, Qf2, "r")
polRegression(X, Y10, Qf10, "g")

print_different_errors()