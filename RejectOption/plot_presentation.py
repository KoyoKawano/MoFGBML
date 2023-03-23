# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 06:02:45 2023

@author: kawano
"""

from MyDataset import DatasetMaker as DM
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from ThresholdBaseRejection import SingleThreshold, SecondStageRejectOption
from Cilab_Classifier import Rule, FuzzyClassifier
from RejectorBaseRejection import RejectorBasedRejectOption 
from RejectorModel import Rejector, ResampledTransformer
from sklearn.neighbors import KNeighborsClassifier

 

def all_rules(X_train, y_train):
    
    fuzzyset_ID = np.arange(6)
    
    ruleset = [Rule([i, j]).culc_conseqent(X_train, y_train) for i in fuzzyset_ID for j in fuzzyset_ID]
    
    ruleset = list(filter(lambda x : x.CF > 0, ruleset))
    
    fuzzyClassifier = FuzzyClassifier(ruleset)
    
    return fuzzyClassifier

def get_rotation_matrix(X, deg):
    """
    指定したradの回転行列を返す
    """
    rad = deg * np.pi / 180
    rot = np.array([[np.cos(rad), -np.sin(rad)],
                    [np.sin(rad), np.cos(rad)]])
    
    
    return preprocessing.MinMaxScaler().fit_transform(X @ rot)

def base_scatter(X, y, fig, axes):
    
    clf = all_rules(X, y)
    
    n_div = 100
    
    X_grid = DM().grid_dataset(n_div)
    
    fig, axes = dataset.plot_2d_dataset(X, y, fig = fig, ax = axes)
    
    axes.contour(X_grid[:,0].reshape((n_div+1, n_div+1)),
                 X_grid[:,1].reshape((n_div+1, n_div+1)),
                 clf.predict(X_grid).reshape((n_div+1, n_div+1)),
                 colors = ["k"],
                 linestyles = ["--"],
                 alpha = 0.4
                 )
    
    return fig, axes


def second_single(X_train, y_train):
    
    threshold = 0.55
    
    Single = SingleThreshold()
    
    second = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
    
    clf = all_rules(X_train, y_train)

    n_div = 100
    X_grid = DM().grid_dataset(n_div)
    
    proba = clf.predict_proba(X_grid)
    
    reject = Single.isReject(proba, threshold) & (second.predict(X_grid) != clf.predict(X_grid))
    
    X_reject = X_grid[reject]
        
    fig, axes = plt.subplots(1, 1, figsize=(7, 7), dpi = 300)
    
    axes.scatter(X_reject[:,0], X_reject[:,1], c = "gray", marker = "s")

    fig, axes = base_scatter(X_train, y_train, fig, axes)


def single(X_train, y_train):
    
    threshold = 0.55
    
    Single = SingleThreshold()
    
    clf = all_rules(X_train, y_train)

    n_div = 100
    X_grid = DM().grid_dataset(n_div)
    
    proba = clf.predict_proba(X_grid)
    
    reject = Single.isReject(proba, threshold)
    
    X_reject = X_grid[reject]
        
    fig, axes = plt.subplots(1, 1, figsize=(7, 7), dpi = 300)
    
    axes.scatter(X_reject[:,0], X_reject[:,1], c = "gray", marker = "s")

    fig, axes = base_scatter(X_train, y_train, fig, axes)


def rejector(X_train, y_train):
    
    clf = all_rules(X_train, y_train)
    
    rejector_clf = KNeighborsClassifier(n_neighbors = 3)
    
    transformer = ResampledTransformer(clf)
    
    X_error, y_error = transformer.R(X_train, y_train)

    fig, axes = dataset.plot_2d_dataset(X_error, y_error)
    
    X_r, y_r = ResampledTransformer(clf).transform(X_train, y_train)
    
    fig, axes = dataset.plot_2d_dataset(X_r, y_r)
    
    rm = Rejector(rejector_clf).fit(X_r, y_r)
    
    n_div = 100
    
    X_grid = DM().grid_dataset(n_div)
    
    reject = rm.isReject(X_grid)
    
    X_reject = X_grid[reject]
    
    fig, axes = plt.subplots(1, 1, figsize=(7, 7))
    
    axes.scatter(X_reject[:,0], X_reject[:,1], c = "gray", marker = "s")

    fig, axes = base_scatter(X_train, y_train, fig, axes)


dataset = DM()

X, y = dataset.make_dataset(n_samples = 200, class_sep = 1.5, scale = 1)

X = get_rotation_matrix(X, 40)

fig, axes = plt.subplots(1, 1, figsize=(7, 7), dpi = 300)

fig, axes = base_scatter(X, y, fig, axes)

# fig, axes = DM().plot_2d_dataset(X, y)
single(X, y)