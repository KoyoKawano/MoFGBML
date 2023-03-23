# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 08:07:46 2022

@author: kawano
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def make_setting(dataset):
    
    setting = {"australian" : {"train" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1},
                               "test" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1}},
               
               "pima" : {"train" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1},
                         "test" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1}},
               
               "vehicle" : {"train" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1},
                            "test" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1}},
               
               "heart" : {"train" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1},
                           "test" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1}},
               
               "phoneme" : {"train" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1},
                            "test" : {"x_min" : 0, "x_max" : 0.6, "y_min": 0.6, "y_max" : 1}}}
    
    
    return setting[dataset]


def file_setting(fname):
    
    setting = {"train-single.csv" : {"color" : "tab:blue", "linestyle" : "dashed"},
               "train-cwt.csv" : {"color" : "tab:orange", "linestyle" : "dashed"},
               "train-rwt.csv" : {"color" : "tab:green", "linestyle" : "dashed"},
               "second-train-single.csv" : {"color" : "tab:blue", "linestyle" : "solid"},
               "second-train-cwt.csv" : {"color" : "tab:orange", "linestyle" : "solid"},
               "second-train-rwt.csv" : {"color" : "tab:green", "linestyle" : "solid"}}
    
    return setting[fname]
               
               
    

def fig_setting(x_min, x_max, y_min, y_max):
    
    fontsize = 15
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.grid(color='k', linestyle='dotted', linewidth=1)
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"               
    plt.rcParams["xtick.minor.visible"] = True           
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.major.width"] = 1.5              
    plt.rcParams["ytick.major.width"] = 1.5              
    plt.rcParams["xtick.minor.width"] = 1.0              
    plt.rcParams["ytick.minor.width"] = 1.0
    plt.rcParams["xtick.major.size"] = 8                
    plt.rcParams["ytick.major.size"] = 8                
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    
    return plt

def result(result_pass, dataset, fname):
    
    def result_one_files(result_files):
        
        df = pd.concat([read_csv(file) for file in result_files])
        
        return step_mean(df)
        
    
    files = [f"{result_pass}/{dataset}/trial{rr}{cc}/{fname}" for rr in range(3) for cc in range(10)]
    
    
    return pd.concat(result_one_files(files))
                        
    

def step_mean(df, step = 0.02):
    
    def df_one_step(t_min, t_max):
        
        culc_df = df.query(f"rejectrate > {t_min}").query(f"rejectrate <= {t_max}")
        
        if len(culc_df) > 0:
            
            return culc_df
        
        return None
    
    df = df.drop_duplicates()
    
    x_min = df["rejectrate"].min() - step
    x_max = df["rejectrate"].max() + step
    
    df_mean = [df_one_step(t, (t + step)) for t in np.arange(x_min, x_max, step)]
    
    return [df.mean() for df in df_mean if df is not None]


def plot_ARCs(df, setting, plot_setting):
    
    plt = fig_setting(setting["x_min"], setting["x_max"], setting["y_min"], setting["y_max"])
    
    x = df["rejectrate"]
    
    y = df["accuracy"]
    
    plt.plot(x, y, color = plot_setting["color"], linestyle = plot_setting["linestyle"])
    
       

def read_csv(fname):
    
    df = pd.read_csv(fname)
    
    return df[["accuracy", "rejectrate"]]


df = read_csv("../results/MoFGBML_Basic/kNN/australian/trial00/train-single.csv")

df_step = step_mean(df)

setting = make_setting("australian")

setting["df"] = df_step

train_files = ["train-single.csv", "train-cwt.csv"]

second_train_files = ["second-train-single.csv", "second-train-cwt.csv"]


# results = result("../results/MoFGBML_Basic/kNN", "australian", "train-single.csv")

dataset = "australian"

for file in train_files:
    
    results = result("../results/MoFGBML_Basic/kNN", dataset, file)
    
    plot_ARCs(results, setting["train"], file_setting(file))
    
for file in second_train_files:
    
    results = result("../results/MoFGBML_Basic/kNN", dataset, file)
    
    plot_ARCs(results, setting["train"], file_setting(file))
    


