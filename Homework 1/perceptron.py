# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 20:14:49 2019

@author: Ana
"""
import random
import matplotlib.pyplot as plt
import time

"""
Class Point(N)
coords = [x_0, x_1, ... x_N-1], with N >= 2
label = our target function
"""
class Point:
    #x: float
    #y: float
    coords = [] # len >= 2
    label: int

    def __init__(self, N):
        #self.x = random.uniform(-20,20)
        #self.y = random.uniform(-20,20)
        self.coords = [random.uniform(-1, 1) for _ in range(N)]
        if self.coords[0] > self.coords[1]:
            self.label = 1
        else:
            self.label = -1

#######################################

def guess(inputs):
    sum_ = 0
    for i in range(len(weights)):
        sum_ += inputs[i] * weights[i]
    return sign(sum_)

# The activation function
def sign(value):
    if value >= 0:
        return 1
    else:
        return -1

# Training the data
def train(inputs, target):
    guess_ = guess(inputs)
    error = target - guess_
    for i in range(len(weights)):
        weights[i] += error * inputs[i] * learning_rate

# Ploting the points
def print_points(points):
    for p in points:
        #pt = [p.x, p.y]
        pt = p.coords
        if guess(pt) == -1:
            color = "g"
        else:
            color = "r"
        plt.scatter(*pt, color=color)
    plt.plot([-dims, dims], [-dims, dims], "-b") # line 
    plt.show()
 
# Looking for misclassifications
def check_points(points):
    mis_classification = []
    for p in points:
        #pt = [p.x, p.y]
        pt = p.coords
        if guess(pt) != p.label:
            mis_classification.append(p)
    return mis_classification



# Initializations
# ###############
size = 1000
dims = 1
N = 2
random.seed(36) # Changed for every test
weights = [random.randint(-1,1) for _ in range(N)]
points = [Point(N) for _ in range(size)]
learning_rate = 0.1



# Print initial points
# ####################
#print("\n Dataset of size:", size)
#inputs = [(p.x, p.y) for p in points]
inputs = [p.coords for p in points]

# Comment these three lines for N = 10
plt.scatter(*zip(*inputs))
plt.plot([-dims, dims], [-dims, dims], "-b") # line 
plt.show()
# ##############################################

start_time = time.time()
i = 0
while True:
    i += 1
    print("\nIteration number: " + str(i))
    print_points(points) # Comment theis line for N = 10
    mis_class = check_points(points)
    if len(mis_class) == 0:
        break
    else:
        point = random.choice(mis_class)
        #inp = [point.x, point.y]
        inp = point.coords
        train(inp, point.label)
print("\nTime needed for execution:")
print(time.time() - start_time)

random.seed(37) # Changed for every test
testing_points = [Point(N) for _ in range(size)]
num_errors = len(check_points(testing_points))
accuracy = 100 * (size - num_errors) / size
print ("Accuracy: " + str(accuracy))