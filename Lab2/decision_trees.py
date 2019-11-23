# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:33:30 2019

@author: Ana
"""

"""
Decision Trees: 
Partition the dataset into a training and a testing set.
Run a decision tree learning algorithm using the training set. 
Test the decision tree on the testing dataset and report the total classi
cation error
(i.e. 0=1 error). Repeat the experiment with a diferent partition. Plot
the resulting trees. Are they very similar, or very diferent? Explain why.
Advice: it can be convenient to set a maximum depth for the tree. 
"""

# https://www.datacamp.com/community/tutorials/decision-tree-classification-python

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

def get_data(file, test_size):
    x, y = load_svmlight_file(file)
    return train_test_split(x, y, test_size = test_size)
    

def test_error(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    
def plot(clf, name):
    dot_data = StringIO()
    export_graphviz(clf, out_file = dot_data,  
                    filled = True, rounded = True,
                    special_characters = True, class_names = ['1','-1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(name)
    Image(graph.create_png())


def train_tree(x_train, y_train, max_depth):
    clf = DecisionTreeClassifier(max_depth = max_depth)
    clf.fit(x_train, y_train)
    return clf


def build_decision_tree(file, test_size, max_depth, image_name):
    x_train, x_test, y_train, y_test = get_data(file, test_size)
    clf = train_tree(x_train, y_train, max_depth)
    test_error(clf, x_test, y_test)
    plot(clf, image_name)
    

file = "cod-rna.txt"
build_decision_tree(file, 0.2, 20, "first_tree.png")
build_decision_tree(file, 0.5, 20, "second_tree.png")