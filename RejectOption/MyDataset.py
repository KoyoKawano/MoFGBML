# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 15:36:09 2022

@author: kawano
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn import preprocessing
import numpy as np
import itertools

class DatasetMaker():
    
    def __init__(self):
        
        pass
        
        
    def make_dataset(self, n_samples, class_sep, scale, random_state = 8,
                 n_features = 2,
                 n_clusters_per_class = 1,
                 n_classes = 2,
                 weights = [0.5, 0.5],
                 flip_y = 0):
        
        # X, y = make_classification(n_features=100,n_informative=1,n_redundant=0,n_clusters_per_class=1,weights=[0.9,0.1],random_state=12)
        
        X, y = make_classification(n_samples = n_samples,
                                   class_sep = class_sep,
                                   n_features = n_features, 
                                   n_clusters_per_class = n_clusters_per_class,
                                   n_classes = n_classes,
                                   weights = weights,
                                   scale = scale,
                                   random_state = random_state,
                                   n_informative = 1,
                                   n_redundant = 0,
                                   hypercube = True, 
                                   flip_y = flip_y)
        
        X = preprocessing.MinMaxScaler().fit_transform(X)
        
        return X, y
    
    
    def plot_2d_dataset(self, X, y, setting = None, fig = None, ax = None):
        
        if setting == None:
            setting = {"0" : {"color" : "tab:blue", "size" : 200, "marker" : "o"},
                       "0.0" : {"color" : "tab:blue", "size" : 200, "marker" : "o"},
                       "1" : {"color" : "tab:orange", "size" : 200, "marker" : "o"},
                       "1.0" : {"color" : "tab:red", "size" : 200, "marker" : "o"},
                       "-1.0" : {"color" : "gray", "size" : 200, "marker" : "x"}
                       }
            
        if fig == None and ax == None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi = 300)
        
        ax.set_aspect('equal', adjustable='box')
        
        color = [setting[str(label)]["color"] for label in y]
        linewidths = 0.5
        edgecolors = 'black'
        alpha = 0.8
        plt.rcParams['font.family'] = 'Times New Roman'
        for x0, x1, label in zip(X[:,0], X[:,1], y):
            
            marker = "o"
            size = 200
            color = "tab:blue"
            if label == -1.0:
                marker = "X"
                color = "royalblue"
                size = 400
                
            if str(label) == "1.0":
                color = "tab:red"
                
            if str(label) == "1":
                color = "tab:orange"
                
            
                
            ax.scatter(x0,
                       x1,
                       s = size,
                       c = color,
                       linewidths = linewidths,
                       edgecolors = edgecolors,
                       alpha = alpha,
                       marker = marker)
            
            plt.xlim(0,1)
            plt.ylim(0,1)
            
        
        return fig, ax
    
    def grid_dataset(self, n_div):
        
        width = 1 / n_div
        
        X = np.arange(0, 1 + width, width)
    
        X = np.array([np.array(x) for x in itertools.product(X, repeat = 2)])
        
        return X
    
if __name__ == "__main__":
    
    dataset = DatasetMaker()
    
    X, y = dataset.make_dataset(n_samples = 200, class_sep = 1.5, scale = 1)
    
    dataset.plot_2d_dataset(X, y)
    