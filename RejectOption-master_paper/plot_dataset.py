# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:24:08 2023

@author: kawano
"""

import seaborn as sns
from CIlab_function import CIlab
from matplotlib import pyplot as plt

class plotDataset():
    
    def __init__(self):
        pass
    
    def pairplot(fname, fout):
        
        df = CIlab.load_cilab_style_dataset(fname)
        
        axis = sns.pairplot(df, hue = "target")
        
        axis.savefig(fout, dpi = 300)
        
        return
    
    def hist(fname):
        
        df = CIlab.load_cilab_style_dataset(fname)
        
        num_attribute = len(df.columns) - 1

        v = 5
        h = 1
        fontsize = 10
        cols = df.columns[:-1]
        
        fig, axes = plt.subplots(h, v, dpi = 300)
        
        plt.subplots_adjust(wspace=0.45, hspace=0.3)
        
        axes = axes.ravel()

        for col, ax in zip(cols, axes):
            
            sns.histplot(df, x = col, hue = "target", ax = ax, legend = None)
            ax.set_xlabel(col, fontsize = fontsize)
            ax.set_ylabel("count", fontsize = fontsize)
            ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            ax.tick_params(direction = "in", length = 2, labelsize = 8)
    
        # for i in range(num_attribute):
            
        #     ax[int(i/v), i%v] = sns.histplot(data = df, x = f"attribute{i}", hue = "target")

        # fig.tight_layout()
        return
    
    def kdeplot(data, fname = None):
        
        df = CIlab.load_cilab_style_dataset(data)
        
        
        num_attribute = len(df.columns) - 1

        v = num_attribute
        h = 1
        fontsize = 1
        cols = df.columns[:-1]
    
        for i, col in enumerate(cols):
            
            fig, axes = plt.subplots(nrows=1, ncols=1, dpi = 300)
            
            sns.kdeplot(data = df, x = col, hue = "target", legend = None)
            axes.set_xlim(0, 1)
            # axes.set_ylim(-0.02, 1.02)
            axes.set_box_aspect(1)
            # axes.get_xaxis().set_visible(False)
            # axes.get_yaxis().set_visible(False)
        
            if fname != None:
                plt.savefig(f"{fname}_attribute{i}.png", dpi = 300)
 
        else:
            plt.show()
            
            
        # for col, ax in zip(cols, axes):
        #     sns.kdeplot(data = df, x = col, hue = "target", ax = ax, legend = None)
        #     # ax.set_xlabel(col, fontsize = fontsize)
        #     # ax.set_ylabel("Denticy", fontsize = fontsize)
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        #     ax.set_xlim(-2, 2)
        #     ax.set_ylim(-0.02, 3)
        #     ax.set_aspect(2)
            
        #     ax.tick_params(direction = "in", length = 2, labelsize = 8)
    
        # for i in range(num_attribute):
            
        #     ax[int(i/v), i%v] = sns.histplot(data = df, x = f"attribute{i}", hue = "target")

        # fig.tight_layout()
        return

class plotRule():
    
    def plot(clf):
        
        pass

if __name__ == "__main__" :

    dataset = "phoneme"
    
    rr = 0
    cc = 0
    
    fname_train = f"../dataset/{dataset}/a{rr}_{cc}_{dataset}-10tra.dat"
    
    # fname_resampled = f"../results/resampled_dataset/MoFGBML_Basic/{dataset}/a{rr}_{cc}_{dataset}-10tra.dat"
    
    fname_ensemble = f"../results/ensemble_dataset/{dataset}/a{rr}_{cc}_{dataset}-10tra.dat"
    
    # fname = f"../results/plots/pop/{dataset}/{dataset}"
    # plotDataset.kdeplot(fname_train, fname)
    
    # fname = f"../results/plots/pop/{dataset}/{dataset}_noError"
    # plotDataset.kdeplot(fname_no_error, fname)
    
    plotDataset.pairplot(fname_train, f"../results/plots/dataset/{dataset}_train.png") 
   
    # plotDataset.pairplot(fname_ensemble, f"../results/plots/dataset/{dataset}_ensemble.png")