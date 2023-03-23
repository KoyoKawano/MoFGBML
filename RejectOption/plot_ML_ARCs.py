# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 08:59:27 2023

@author: kawano
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:16:01 2023

@author: kawano
"""
from CIlab_function import CIlab
from FuzzyClassifierFromCSV import FileInput
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os

def make_setting(dataset):
    
    return {"dataset" : dataset, "marker":"o", "color":"tab:orange", "size":150, "algorithmID" : "MoFGBML_Basic"}
            

def make_threshold_setting():
    
    setting = {"single" : {"linestyle" : "dotted"},
               "cwt" : {"linestyle" : "dashed"},
               "rwt" : {"linestyle" : "dashdot"}}
    
    return setting


class plotMLModelARC():
    
    def __init__(self, measure = ["acc_train", "reject_train", "acc_test", "reject_test"]):
        
        self.measure = measure
        
    
    def figSetting(self, xMin, xMax, yMin, yMax):
        
        # figsizeで図のサイズを指定 横長にしたいなら左の数値を大きくする．
        fig, axes = plt.subplots(1, 1, figsize=(7, 7), dpi = 300)
        
        # X ticks (= Number of rule)
        xH = 0.1
        xticks = np.arange(0, xMax + 0.1, 0.1)
    
        xMax = xticks[len(xticks) - 1]
        
        xH = 0.02
        axes.set_xlim(xMin - xH, xMax + xH)
        axes.set_xticks(xticks)
        
        # Y ticks
        yH = 0.05
        yMin = (int)(yMin / yH)
        yMin = yMin * yH
        yMax = (int)((yMax + yH) / yH)
        yMax = yMax * yH  
        
        yticks = np.arange(yMin, yMax + 0.02, 0.05)
        yMin = yticks[0]
        yMax = yticks[-1]
        
        yH = 0.01
        axes.set_ylim(yMin - yH, yMax + yH)
        axes.set_yticks(yticks)
        
        axes.set_xlabel("Reject rate", fontsize = 26)
        axes.set_ylabel("Accuracy", fontsize = 26)
        
        fig.subplots_adjust(left = 0.2)
        axes.grid(linewidth=0.4)
        axes.yaxis.set_minor_locator(AutoMinorLocator(3))
        axes.tick_params(which = 'major', length = 8, color = 'black', labelsize = 25)
        axes.tick_params(which = 'minor', length = 5, color = 'black', labelsize = 25)
        
        return fig, axes
        
    
    def _get_result(self, setting):
        
        algorithmID = "../results/rejector/" + setting["algorithmID"]
        
        dataset = setting["dataset"]
        
        RR = 3
        CC = 10
        
        files = [f"{algorithmID}/{dataset}/trial{rr}{cc}/result.csv" for rr in range(RR) for cc in range(CC)]
        
        results_list = [pd.read_csv(file, index_col = 0) for file in files]
        
        models = results_list[0].index
        
        measures = ["acc_train", "reject_train", "acc_test", "reject_test"]
        
        mean_dict = {}
        std_dict = {}
        
        for model in range(len(models)):
            
            result = pd.concat([results_list[i].iloc[model] for i in range(30)])
            
            measure_mean = {}
            measure_std = {}
            
            for measure in measures:
                
                measure_mean[measure] = result[measure].mean()
                measure_std[measure] = result[measure].std()
            
            mean_dict[models[model]] = measure_mean
            std_dict[models[model]] = measure_std
        
        return mean_dict, std_dict
    
    
    def _get_threshold_base(self, setting, model, num_stage, base = "single", y_measure = "train"):
        
        def get_one_trial(files):
        
            dfs = [pd.read_csv(file) for file in files]
            
            dfs = [df.drop_duplicates() for df in dfs]
            
            return step_mean(dfs)
        
        def step_mean(dfs, step = 0.02):
    
            def df_one_step(t_min, t_max):
        
                culc_df = [df.query(f"rejectrate > {t_min}").query(f"rejectrate <= {t_max}") for df in dfs]
                culc_df = [df for df in culc_df if not df.empty]
                
                criteria = 16
                
                if len(culc_df) >= criteria:
                    
                    return pd.concat(culc_df)
                
                return None
            
            x_min = min([df["rejectrate"].min() for df in dfs]) - step
            x_max = max([df["rejectrate"].max() for df in dfs]) + step

            df_mean = [df_one_step(t, (t + step)) for t in np.arange(x_min, x_max, step)]
    
            return [df.mean() for df in df_mean if df is not None]


        if y_measure == "acc_train":
            y_measure = "train"
            
        if y_measure == "acc_test":
            y_measure = "test"
            
        algorithmID = "../results/threshold_base/" + setting["algorithmID"]
        
        dataset = setting["dataset"]
        
        RR = 3
        CC = 10
        
        files = [f"{algorithmID}/{dataset}/trial{rr}{cc}/{y_measure}-{base}.csv" for rr in range(RR) for cc in range(CC)]
        
        if num_stage == 2:
            
            files = [f"{algorithmID}/{dataset}/trial{rr}{cc}/{model}/second-{y_measure}-{base}.csv" for rr in range(RR) for cc in range(CC)]
        
        return pd.concat(get_one_trial(files))
 
    
    def plot_threshold_base(self, setting, y_measure, model, fig, axes, bases = ["single", "cwt", "rwt"]):
        
        setting_threshold = make_threshold_setting()
        
        for num_stqage in [1, 2]:
            
            for base in bases:
                
                results = self._get_threshold_base(setting, model, num_stqage, base, y_measure)
                
                color = "tab:blue"
                if num_stqage == 2:
                    color = "tab:orange"
                    
                plt.plot(results["rejectrate"], 
                         results["accuracy"],
                         color = color,
                         linestyle = setting_threshold[base]["linestyle"])
                
    
    def plot_ARC(self, setting, x_measures = ["reject_train", "reject_test"], y_measures = ["acc_train", "acc_test"]):
        
        plt.rcParams["font.family"] = "Times New Roman"     #全体のフォントを設定
        plt.rcParams["font.size"] = 14                      #フォントの大きさ
        plt.rcParams["xtick.minor.visible"] = False         #x軸補助目盛りの追加
        plt.rcParams["ytick.direction"] = "out"             #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["ytick.major.size"] = 10               #y軸主目盛り線の長さ
        plt.rcParams['figure.subplot.bottom'] = 0.15
        
        mean, std = self._get_result(setting)
        
        plot_result = {k: v for k, v in mean.items() if k != 'base'}
        
        models = plot_result.keys()
        
        markers = ["$AB$", "$DT$", "$NB$", "$GP$", "$kNN$", "$MLP$", "$RF$", "$SVM$"]
        
        for marker, model in zip(markers, models):
            
            for x_measure, y_measure in zip(x_measures, y_measures):
                
                xMin = 0.0
                xMax = 0.6
                # xMax = plot_result[model][x_measure]
                yMin = mean["base"][y_measure]
                yMax = 0.95
    
                fig, axes = self.figSetting(xMin, xMax, yMin, yMax)
                    
                color = setting['color']
                size = setting['size']
                linewidths = 1
                edgecolors = 'black'
                alpha = 0.8
                plt.rcParams['font.family'] = 'Times New Roman'
                
                x = mean[model][x_measure]
                y = mean[model][y_measure]
                
                axes.scatter(x,
                             y,
                             s = size,
                             marker = "o",
                             color = color,
                             linewidths = linewidths,
                             edgecolors = edgecolors,
                             alpha = alpha)
                

            
                self.plot_threshold_base(setting, y_measure, model, fig, axes)     
                
                dataset = setting["dataset"]
                
                output_dir = f"../results/plots/ARC_ML_thesis/{dataset}/{model}"
    
                if not os.path.exists(output_dir):
    
                    os.makedirs(output_dir)
        
                fig.savefig(f"{output_dir}/{dataset}_{model}_{y_measure}.pdf", dpi = 300)
            
        return fig, axes
                
    
def run_dataset(dataset):
    
    setting = make_setting(dataset)
    # fig, axes = plt.subplots(1, 1, figsize=(7, 7))
    # plotRejectorARC().plot_threshold_base(setting[0], "acc_test", fig, axes)
    
    plotMLModelARC().plot_ARC(setting)
    
    return f"finished_{dataset}"
       
    
if __name__ == "__main__":
    
    # datasets = ["pima", "australian", "vehicle", "heart", "phoneme", "mammographic", "ring", "spambase", "sonar", "segment", "bupa", "bal", "ecoli", "page-blocks", "wisconsin"]
    
    # datasets = ["spambase"]
    datasets = ["vehicle"]
    
    for dataset in datasets:
        
        print(run_dataset(dataset))