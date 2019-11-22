# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:36:53 2019

@author: Ana
"""

import numpy as np
import matplotlib.pyplot as plt
import math, csv, os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statistics

# Get X and X.reshape(-1, 1)
def get_x(n):
    x_ = np.random.uniform(-1, 1, n)
    x_.sort()
    x_reshape = x_.reshape(-1, 1)
    return x_, x_reshape


def initialize():
    Qf2, Qf10 = 2, 10
    N = 100
    X, _ = get_x(N)
    sigma = 0.2
    epsilon = np.random.standard_normal(N)
    return Qf2, Qf10, N, sigma, X, epsilon


# Get rescaled aq
def norm_aqs(Qf):
    aqs = np.random.standard_normal(Qf + 1)
    summ = sum([aqs[q]**2/(2 * q + 1) for q in range(Qf + 1)])
    return [aqs / math.sqrt(2 * summ)]


# Obtain function: F(x) = Summation (aq*Lq(x))
def fun_x(x, Qf, aqs):
    return sum([aqs[0][q] * legendre(q, x) for q in range(Qf)])


# Get legendre polynomial of degree k at x
def legendre(k, x):
    if k == 0: 
        return 1
    elif k == 1:
        return x
    else: 
        return((2 * k - 1) / k) * x * legendre(k - 1, x) - ((k - 1) / k) * legendre(k - 2, x)


# Get y = f(x) + sigma * epsilon(x)
def get_y(fx, sigma, epsilon):
    y = fx + sigma * epsilon
    return y


def plotResults(x, y, y_pred, Qf, label, title = "", color = "r"):
    plt.figure()
    plt.scatter(x, y, label = "points")
    plt.plot(x, y_pred, c = color, label = "h" + str(Qf))
    plt.title(title, fontsize = 16)
    plt.legend()
    #plt.show()
    directory = "plots_experiments"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + label + '.png')


def plotResults_h2_h10(x, y, y2, y10, label, title = ""):
    plt.figure()
    plt.scatter(x, y, label = "points")
    plt.title(title, fontsize = 16)
    plt.plot(x, y2, c = "r", label = "h2")
    plt.plot(x, y10, c = "g", label = "h10")
    plt.legend()
    #plt.show()
    directory = "plots_experiments"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + label + '.png')


def polRegression(x, y, Qf, color, label, title="", plot = True):
    x_reshape = x.reshape(-1, 1)
    
    poly_feat = PolynomialFeatures(degree = Qf, include_bias = True)
    lin_reg = LinearRegression()
    x_ = poly_feat.fit_transform(x_reshape)
    lin_reg.fit(x_, y)
    y_pred = lin_reg.predict(x_)
    
    if plot:
        plotResults(x, y, y_pred, Qf, label, title, color)
    
    return x, y_pred, lin_reg, poly_feat


def polRegression_models(x, Qf, sig, eps, n, label, plot):
    norm_aq = norm_aqs(Qf)
    
    g = fun_x(x, Qf, norm_aq)
    Y = get_y(g, sig, eps)
    
    title = "N = " + str(n) + ", Sigma = " + str(sig) + ", Qf = " + str(Qf)
    title2 = "H2 - " + title
    title10 = "H10 - " + title

    x2, y_pred2, lin_reg2, poly_feat2 = polRegression(x, Y, 2, "r", "H2_" + label, title2, plot)
    x10, y_pred10, lin_reg10, poly_feat10 = polRegression(x, Y, 10, "g", "H10_" + label, title10, plot)
    
    x_, x_reshape = get_x(n)
    y_ = fun_x(x_, Qf, norm_aq)
    
    # E_out:
    new_y2 = lin_reg2.predict(poly_feat2.fit_transform(x_reshape))
    new_y10 = lin_reg10.predict(poly_feat10.fit_transform(x_reshape))
    E_out2  = ((y_ - new_y2)**2).mean()
    E_out10 = ((y_ - new_y10)**2).mean()
    
    if plot:
        plotResults_h2_h10(x_, y_, new_y2, new_y10, label, title)
    
    return E_out2, E_out10


def write_results_csv(result):
    print(result)
    with open('experiments.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(result)
    csvFile.close()
    

def print_different_errors(plot):
    Ns = np.arange(50, 121, 50)
    sigmas = np.arange(0.5, 1.05, 0.3)
    Qfs = np.arange(5, 16, 5)
    it = 1
    results = [["N", "Qf", "Sigma", "Mean E2", "Mean E10", "Overfit"]]
    for n in Ns:
        for qf in Qfs:
            for sig in sigmas:
                err2_total, err10_total = [], []
                for i in range(it):
                    x, _ = get_x(n)
                    eps = np.random.standard_normal(n)
                    label = "N_" + str(n) + "_Qf_" + str(qf) + "_Sigma_" + str(sig)
                    err2, err10 = polRegression_models(x, qf, sig, eps, n, label, plot)
                    err2_total.append(err2)
                    err10_total.append(err10)
                err2_mean = statistics.mean(err2_total)
                err10_mean = statistics.mean(err10_total)
                overfit = err10_mean - err2_mean
                results.append([n, qf, sig, err2_mean, err10_mean, overfit])
    
    write_results_csv(results)


def basic_problem(plot):
    Qf2, Qf10, N, sigma, X, epsilon = initialize()
    # Select aq independently from a standard Normal distribution
    norm_aq2 = norm_aqs(Qf2)
    norm_aq10 = norm_aqs(Qf10)
    
    g2 = fun_x(X, Qf2, norm_aq2) # H2
    g10 = fun_x(X, Qf10, norm_aq10) # H10
    
    Y2 = get_y(g2, sigma, epsilon)
    Y10 = get_y(g10, sigma, epsilon)
    
    title = "N = " + str(N) + ", Sigma = " + str(sigma)
    title2 = "H2 - " + title + ", Qf = " + str(Qf2)
    title10 = "H10 - " + title + ", Qf = " + str(Qf10)
    
    label2 = "basic_N_" + str(N) + "_Qf_" + str(Qf2) + "_Sigma_" + str(sigma)
    label10 = "basic_N_" + str(N) + "_Qf_" + str(Qf10) + "_Sigma_" + str(sigma)
    
    polRegression(X, Y2, Qf2, "r", label2, title2, plot)
    polRegression(X, Y10, Qf10, "g", label10, title10, plot)


# Select what part to run: (comment what's not needed)
# the basic_problem is just a simple example for a given N, Sigma and Qf
# Mark plot as True or False, depending on whether the graphic plots want to be shown or not.
np.random.seed(8)
basic_problem(plot = True) # First part of the assignment
print_different_errors(plot = True) # Part d & e