#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:00:44 2019

@author: Ana Mestre
"""
import random
import matplotlib.pyplot as plt

def gen_random_point(dim, x0, x1):
	point = ()
	for d in range(dim):
		point += (random.uniform(x0, x1),)
	return point

def gen_points(dim, x0, x1, size):
	return [gen_random_point(dim, x0, x1) for _ in range(size)]


size = 20
points = gen_points(2, -100, 100, size)
print(points)
plt.scatter(*zip(*points))
plt.show()
