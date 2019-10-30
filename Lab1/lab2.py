# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:52:34 2019

@author: Ana

Coefs shuttle: 10 arrays (num N) de 7 llistes (7 classes) de 9 elements (features)

"""

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import random
from timeit import default_timer as timer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
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
    plt.show()

def plot_stem_weights(coef_total, n_total):
	#for class in coef_total:
		for coef, n in zip(coef_total, n_total):
			plt.figure()
			markerline, stemlines, _ = plt.stem(coef)
			plt.setp(stemlines, 'linestyle', 'dotted')
			plt.title("Coefficients stem plot with N = " + str(n), fontsize = 24)
			plt.show()

#flatten = lambda l: [item for sublist in l for item in sublist]    
dataset = load_svmlight_file("cod-rna.txt")
length = dataset[0].shape[0]
mse_total, N_total, times_total, coef_total = [], [], [], []
random_values = random.sample(range(1, length), 10)
random_values.sort()
for N in random_values:
    x, y = get_dataset(dataset, N)
    coef, mse, times = logistic_regression(x, y, N)
    #print(coef.shape)
    #print("-----------")
    #print(coef[0].shape)
    #print("----------------------------------")
    ns = list(range(1, N + 1))
    times_total.append(times)
    mse_total.append(mse)
    N_total.append(N)
    coef_total.append(coef)
#new_cofs = [c[0] for c in coef_total]
new_coefs = []
for sublist in coef_total:
        for item in sublist:
            new_coefs.append(item)
#print(new_coefs)
#print("........")
#print(coef_total)
#print(coef_total)
#plot_values(N_total, mse_total, "Accuracy score", "N", "Error")
#plot_values(N_total, times_total, "CPU time", "N", "Time")
plot_values(N_total, coef_total, "Weights", "N", "Weights")
#print(flatten(coef_total))
#plot_stem_weights(flatten(coef_total), N_total)
