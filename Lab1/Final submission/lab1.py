# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:13:16 2019

@author: Ana Mestre
"""

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random, os
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Generate a dataset of size N starting at a random position
def get_dataset(dataset, N):
    x, _, y, _ = train_test_split(dataset[0], dataset[1], train_size = N)
    return x, y

# Get weights, accuracy and time for linear regression 
def linear_regression(x, y):
    time0 = timer()
    model = LinearRegression().fit(x, y)
    pred_y = model.predict(x)
    mse = mean_squared_error(y, pred_y)
    time1 = timer()
    times = time1 - time0
    return model.coef_, mse, times

# Plot values with values at the Y-axis and ns at the X-axis
def plot_values(ns, values, text = "", labelx = "", labely = ""):
    plt.figure()
    plt.suptitle(text, fontsize = 15)
    plt.xlabel(labelx, fontsize = 10)
    plt.ylabel(labely, fontsize = 10)
    plt.plot(ns, values)
    directory = "linear_" + str(len(ns))
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/plot_' + labely + '.png')

# Show a weights plot per every N that has been calculated
def plot_stem_weights(coef_total, n_total):
    i = 0
    for coef, n in zip(coef_total, n_total):
        plt.figure()
        markerline, stemlines, _ = plt.stem(coef)
        plt.setp(stemlines, 'linestyle', 'dotted')
        plt.title("Coefficients stem plot with N = " + str(n), fontsize = 15)
        plt.xlabel("Sample size", fontsize = 10)
        plt.ylabel("Weight", fontsize = 10)
        i += 1
        directory = "linear_" + str(len(n_total))
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + "/n_" + str(n) + "__it_" + str(i) + ".png")


# --------------------------------------------------- #
# --------------------------------------------------- #
# --------------------------------------------------- #    
dataset = load_svmlight_file("cadata.txt")
length = dataset[0].shape[0]
mse_total, N_total, times_total, coef_total = [], [], [], []
random_values = random.sample(range(1, length), 50)
random_values.sort()
for N in random_values:
    x, y = get_dataset(dataset, N)
    coef, mse, times = linear_regression(x, y)
    ns = list(range(1, N + 1))
    times_total.append(times)
    mse_total.append(mse)
    N_total.append(N)
    coef_total.append(coef)

# Plotting values as functions of N
plot_values(N_total, mse_total, "Mean squared error", "N", "Error")
plot_values(N_total, times_total, "CPU time", "N", "Time")
plot_values(N_total, coef_total, "Weights", "N", "Weights")
plot_stem_weights(coef_total, N_total)
