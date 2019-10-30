# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:52:34 2019

@author: Ana
"""

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import random
from timeit import default_timer as timer
from sklearn.metrics import classification_report, accuracy_score
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def get_x_y(dataset):
    return dataset[0], dataset[1]

def get_dataset(dataset, N):
    # N = size of the sample
    length = dataset[0].shape[0]
    i = random.randint(0, length - N)
    x, y = get_x_y(dataset)
    return x[i : i + N], y[i : i + N]

def logistic_regression(x, y, N):
    time0 = timer()
    model = LogisticRegression().fit(x, y)
    pred_y = model.predict(x)
    mse = accuracy_score(y, pred_y)
    time1 = timer()
    times = time1 - time0
    return model.coef_, mse, times

def plot_values(ns, values, text = "", labelx = "", labely = ""):
    plt.figure()
    plt.suptitle(text, fontsize = 15)
    plt.xlabel(labelx, fontsize = 10)
    plt.ylabel(labely, fontsize = 10)
    plt.plot(ns, values)
    

    
dataset = load_svmlight_file("shuttle.txt")
length = dataset[0].shape[0]
mse_total, N_total, times_total, coef_total = [], [], [], []
random_values = random.sample(range(1, length), 10)
random_values.sort()
for N in random_values:
    x, y = get_dataset(dataset, N)
    coef, mse, times = logistic_regression(x, y, N)
    ns = list(range(1, N + 1))
    times_total.append(times)
    mse_total.append(mse)
    N_total.append(N)
    coef_total.append(coef)

plot_values(N_total, mse_total, "Accuracy score", "N", "Error")
plot_values(N_total, times_total, "CPU time", "N", "Time")
#plot_values(N_total, coef_total, "Weights", "N", "Weights")

# TODO: fer també un sol plot del mean sq error per N

# 1 - Linear Regression with regression dataset
# Plot the approximation error
# Plot the cpu-time
# Explore and comment results

# 2 - Logistic Regression with classification dataset