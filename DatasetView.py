# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:49:36 2021

@author: kawano
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

sep = '\\'
dataset = 'pima'
folder = "dataset" + sep + dataset + sep

train = folder + "a0_0_" + dataset + "-10tra.dat"
test = folder + "a0_0_" + dataset + "-10tst.dat"


def readData(FileName):
    # ----Get Attribute Num
    df = pd.read_csv(FileName)        
    #----Data Frame
    df = pd.read_csv(FileName, header = None, skiprows = 1)
    df = df.drop(columns = df.columns[-1])

    return df


def Scatter_plot_matrix(df):
    sns.pairplot( df, hue = df.columns[-1])


def PCA_plot(df):
    fig = plt.figure(figsize=(12, 8))
    
    manifolders = {
        'PCA': PCA(),
        'MDS': MDS(),
        'Isomap': Isomap(),
        'LLE': LocallyLinearEmbedding(),
        'Laplacian Eigenmaps': SpectralEmbedding(),
        't-SNE': TSNE(),
    }
    
    for i, (name, manifolder) in enumerate(manifolders.items()):
        
        ax = fig.add_subplot(2, 3, i + 1)
    
        X_transformed = manifolder.fit_transform(df.drop(columns = df.columns[-1]))
    
        sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1], alpha=0.8, hue=list(df.iloc[:, -1]))
        ax.legend()
        
    plt.show()


def main():
    # ----DataFrame of train + test
    df = pd.concat([readData(train), readData(test)])
    Scatter_plot_matrix(df)
    PCA_plot(df)

if __name__ == '__main__':
    main()