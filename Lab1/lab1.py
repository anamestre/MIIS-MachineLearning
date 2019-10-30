# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:13:16 2019

@author: Ana Mestre
"""

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
import random, time
from statistics import mean 
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def get_x_y(dataset):
    return dataset[0], dataset[1]

def get_dataset(dataset, N):
    # N = size of the sample
    length = dataset[0].shape[0]
    i = random.randint(0, length - N)
    x, y = get_x_y(dataset)
    return x[i : i + N], y[i : i + N]

def linear_regression(x, y, N):
    time0 = timer()
    model = LinearRegression().fit(x, y)
    pred_y = model.predict(x)
    mse = mean_squared_error(y, pred_y)
    time1 = timer()
    times = time1 - time0
    return model.coef_, mse, times

def plot_values(ns, values, text = "", labelx = "", labely = ""):
    plt.figure()
    plt.suptitle(text, fontsize = 15)
    plt.xlabel(labelx, fontsize = 10)
    plt.ylabel(labely, fontsize = 10)
    plt.plot(ns, values)
    plt.show()

def plot_stem_weights(coef_total, n_total):
	for coef, n in zip(coef_total, n_total):
		plt.figure()
		markerline, stemlines, _ = plt.stem(coef)
		plt.setp(stemlines, 'linestyle', 'dotted')
		plt.title("Coefficients stem plot with N = " + str(n), fontsize = 24)
		plt.show()
    

    
dataset = load_svmlight_file("cadata.txt")
length = dataset[0].shape[0]
mse_total, N_total, times_total, coef_total = [], [], [], []
random_values = random.sample(range(1, length), 10)
random_values.sort()
for N in random_values:
    x, y = get_dataset(dataset, N)
    coef, mse, times = linear_regression(x, y, N)
    print(coef)
    ns = list(range(1, N + 1))
    times_total.append(times)
    mse_total.append(mse)
    N_total.append(N)
    coef_total.append(coef)
print(coef_total)
#plot_values(N_total, mse_total, "Mean squared error", "N", "Error")
#plot_values(N_total, times_total, "CPU time", "N", "Time")
#plot_values(N_total, coef_total, "Weights", "N", "Weights")
#plot_stem_weights(coef_total, N_total)

# TODO: fer també un sol plot del mean sq error per N

# 1 - Linear Regression with regression dataset
# Plot the approximation error
# Plot the cpu-time
# Explore and comment results

