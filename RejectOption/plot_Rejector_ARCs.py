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
    
    return [
            {"dataset" : dataset, "marker":"o", "color":"tab:blue", "size":200, "algorithmID" : "MoFGBML_Basic_28set"},
            {"dataset" : dataset, "marker":"o", "color":"tab:orange", "size":200, "algorithmID" : "MoFGBML_Basic_ensemble"}
            ]

def make_threshold_setting():
    
    setting = {"single" : {"color" : "tab:blue", "linestyle" : "dotted"},
               "cwt" : {"color" : "tab:blue", "linestyle" : "dashed"},
               "rwt" : {"color" : "tab:blue", "linestyle" : "dashdot"},
               "second-single" : {"color" : "tab:blue", "linestyle" : "solid"},
               "second-cwt" : {"color" : "tab:orange", "linestyle" : "solid"},
               "second-rwt" : {"color" : "tab:green", "linestyle" : "solid"}}
    
    return setting

class plotRejectorARC():
    
    def __init__(self, measure = ["acc_train", "reject_train", "acc_test", "reject_test"]):
        
        self.measure = measure
        
    
    def figSetting(self, xMin, xMax, yMin, yMax):
        
        # figsizeで図のサイズを指定 横長にしたいなら左の数値を大きくする．
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))
        
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
        
        axes.grid(linewidth=0.4)
        axes.yaxis.set_minor_locator(AutoMinorLocator(3))
        axes.tick_params(which = 'major', length = 8, color = 'black', labelsize = 25)
        axes.tick_params(which = 'minor', length = 5, color = 'black', labelsize = 25)
        
        return fig, axes
        
    
    def _get_result(self, setting):
        
        algorithmID = "../results/ensemble_rejector/" + setting["algorithmID"]
        
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
    
    def _get_threshold_base(self, setting, base = "single", y_measure = "train"):
        
        def get_one_trial(files):
        
            dfs = [pd.read_csv(file) for file in files]
            
            dfs = [df.drop_duplicates() for df in dfs]
            
            return step_mean(dfs)
        
        def step_mean(dfs, step = 0.02):
    
            def df_one_step(t_min, t_max):
        
                culc_df = [df.query(f"rejectrate > {t_min}").query(f"rejectrate <= {t_max}") for df in dfs]
                culc_df = [df for df in culc_df if not df.empty]
                
                criteria = 16
                
                if len(culc_df) > criteria:
                    
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
        
        files = [f"{algorithmID}/kNN/{dataset}/trial{rr}{cc}/{y_measure}-{base}.csv" for rr in range(RR) for cc in range(CC)]
        
        # result = get_one_trial(files)
        
        # if all(np.array(result) is None):
        #     columns = ["accuracy", "rejectrate"]
        #     x = np.arange(0, 1, 0.01)
        #     y = np.zeros(len(x)) + min([df["rejectrate"].min() for df in dfs])
        
        return pd.concat(get_one_trial(files))
 
    def plot_threshold_base(self, setting, y_measure, fig, axes, bases = ["single", "cwt", "rwt"]):
        
        setting_threshold = make_threshold_setting()
        
        for base in bases:
            
            results = self._get_threshold_base(setting, base, y_measure)
            
            plt.plot(results["rejectrate"], 
                     results["accuracy"],
                     color = setting_threshold[base]["color"],
                     linestyle = setting_threshold[base]["linestyle"])
            
    
    def plot_ARC(self, settings, x_measures = ["reject_train", "reject_test"], y_measures = ["acc_train", "acc_test"]):
        
        plt.rcParams["font.family"] = "Times New Roman"     #全体のフォントを設定
        plt.rcParams["font.size"] = 14                      #フォントの大きさ
        plt.rcParams["xtick.minor.visible"] = False         #x軸補助目盛りの追加
        plt.rcParams["ytick.direction"] = "out"             #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["ytick.major.size"] = 10               #y軸主目盛り線の長さ
        plt.rcParams['figure.subplot.bottom'] = 0.15
        
        results = [self._get_result(setting) for setting in settings]

        models = results[0][0].keys()
        
        # markers = ["$AB$", "$DT$", "$GP$", "$kNN$", "$MLP$", "$NB$", "$RF$", "$SVM$"]
        
        
        for x_measure, y_measure in zip(x_measures, y_measures):
            
            xMin = 0.0
            xMax = max([max([v[x_measure] for v in result[0].values()]) for result in results])
            yMin = min([min([v[y_measure] for v in result[0].values()]) for result in results])
            yMax = 0.95

            fig, axes = self.figSetting(xMin, xMax, yMin, yMax)
            
            for result, setting in zip(results, settings):  
                
                mean, std = result
                
                color = setting['color']
                size = setting['size']
                # label = setting['label']
                linewidths = 0.5
                edgecolors = 'black'
                alpha = 0.8
                plt.rcParams['font.family'] = 'Times New Roman'
                
                x = [v[x_measure] for v in mean.values()]
                y = [v[y_measure] for v in mean.values()]
                
                for model in range(len(models)):
                    
                    axes.scatter(x[model],
                                 y[model], 
                                 s = size,
                                 marker = 'o',
                                 color = color,
                                 # label = label,
                                  linewidths = linewidths,
                                  edgecolors = edgecolors,
                                 alpha = alpha)
                
                if setting["algorithmID"] == "MoFGBML_Basic_NoError":
                    axes.axhline(y = y[0], color = color, linestyle = "dashed")
                
                if setting["algorithmID"] == "MoFGBML_Basic":
                    self.plot_threshold_base(setting, y_measure, fig, axes)     
            # axes.set_xlabel("Number of rules", fontsize = 25)
            # axes.set_ylabel(ytickMeasure, fontsize = 25)
            # plt.legend()
            
            dataset = settings[0]["dataset"]
            
            out_dir = f"../results/plots/ensemble_rejector/{dataset}"
            
            if not os.path.exists(out_dir):

                os.makedirs(out_dir)
    
            fig.savefig(f"{out_dir}/{y_measure}.png", dpi = 300)
        
        return fig, axes
                
    
def run_dataset(dataset):
    
    setting = make_setting(dataset)
    # fig, axes = plt.subplots(1, 1, figsize=(7, 7))
    # plotRejectorARC().plot_threshold_base(setting[0], "acc_test", fig, axes)
    
    plotRejectorARC().plot_ARC(setting)
    
    return f"finished_{dataset}"
       
    
if __name__ == "__main__":
    
    datasets = ["pima", "australian","vehicle", "heart", "phoneme", "segment", "wisconsin", "page-blocks"]
    
    # datasets = []
    # datasets = ["heart"]
    
    for dataset in datasets:
        
        print(run_dataset(dataset))