#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Fri Nov 15 12:32:41 2019

@author: jony
"""

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Thu Nov 14 13:40:58 2019

@author: jony
"""
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy.stats import norm

N = 100
X = np.random.uniform(-1,1,N)
ε = np.random.standard_normal(N)
X.sort()


# Get Lq and Qf 
def poly(x,Qf):
    l=legendre(Qf)
    Lq = l(x)
    return Lq , Qf

# def g2 , g10
g2 , Qf = poly(X , 2)
g10 , Qf2 = poly(X , 10)

# check polynoms by plotting them and compare with ML book if ok
plt.plot(X , g2 ,'r--', label = " H2")
plt.plot(X , g10 ,'g--', label = " H10")
plt.legend()
plt.grid()
plt.show()


# Generate aq data - Normal standard distribution
mu, sigma = 1, 0.1 # mean and standard deviation
# Define aq for polynoms
aq = np.random.standard_normal(Qf)
aq2 = np.random.standard_normal(Qf2)

# normalize aq
normalized_aq = preprocessing.normalize([aq] , norm='l2')
normalized_aq2 = preprocessing.normalize([aq2], norm='l2')

# Check Expectation - it should give us E(f^2) = 1
print("\n\nCHECK for E(f^2)=1  ---------> ")
Esperanza = 0 
for i in range(Qf):
    Esperanza += normalized_aq[0][i]**2
    
print("\nEsperanza para poly=2  =  " , Esperanza)

Esperanza2 = 0 
for i in range(Qf2):
    Esperanza2 += normalized_aq2[0][i]**2
    
print("Esperanza para poly=10  =  " , Esperanza2)

# This will return the mean and standard deviation 
# the combination of which define a normal distribution.
mu, std = norm.fit(X)
print("\nSome Information ----->")
print()
print("estimated σ = " , std)

# Define Y(n) = f(x) + σ*ε
Y = []
for i in range(len(X)):
    Y.append(X[i]+((std**2) *ε[i]))

print()
print("Max X Value = ",max(X))
print("Min X Value = ",min(X))
print()
print("Max Y Value = ",max(Y))
print("Min Y Value = ",min(Y))