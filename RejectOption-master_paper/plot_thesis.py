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
    
    setting = {
               # "single" : {"linestyle" : "dotted"},
               # "cwt" : {"linestyle" : "dashed"},
               "rwt" : {"linestyle" : "solid"}
               }
    
    return setting


def make_model_setting():

    setting = {"Adaboost" : {"color" : ""},
               "DecisionTree" : {},
               "NaiveBayes" : {},
               "GaussianProcess" : {},
               "kNN" : {},
               "MLP" : {},
               "RF" : {},
               "SVM" : {}
               }
    
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
 
    
    def _get_threshold_mean(self, setting, y_measure, base):
        
        def step_mean(dfs, step = 0.02):
    
            def df_one_step(t_min, t_max):
                
                columns = ["accuracy", "rejectrate"]
                
                df_model = [pd.DataFrame(np.array([list(df["accuracy"]), list(df["rejectrate"])]).T, columns = columns) for df in dfs if len(df) > 3]
        
                culc_df = [df.query(f"rejectrate > {t_min}").query(f"rejectrate <= {t_max}") for df in df_model]

                culc_df = [df for df in culc_df if not df.empty]
                
                criteria = 5
                
                if len(culc_df) >= criteria:
                    
                    return pd.concat(culc_df)
                
                return None
            
            x_min = min([df["rejectrate"].min() for df in dfs]) - step
            x_max = max([df["rejectrate"].max() for df in dfs]) + step

            df_mean = [df_one_step(t, (t + step)) for t in np.arange(x_min, x_max, step)]
    
        
            return [[(df["accuracy"].min(), df["accuracy"].mean(), df["accuracy"].max()),
                    (df["rejectrate"].min(), df["rejectrate"].mean(), df["rejectrate"].max())] for df in df_mean if df is not None]
        
        model_setting = make_model_setting()
        
        models = model_setting.keys()
        
        result_threshold = [self._get_threshold_base(setting, model, 2, base, y_measure) for model in models]
        
        return step_mean(result_threshold)

            
        
    def plot_threshold_base(self, setting, y_measure, num_stage, fig, axes):
        
        setting_threshold = make_threshold_setting()
        
        bases = setting_threshold.keys()
        
        if num_stage == 1:
            color = "tab:blue"
        
            for base in bases:
                
                results = self._get_threshold_base(setting, None, num_stage, base, y_measure)
                
                plt.plot(results["rejectrate"], 
                         results["accuracy"],
                         color = color,
                         linestyle = setting_threshold[base]["linestyle"])
                
            return fig, axes
            
        color = "tab:orange"
        
        for base in bases:
            
            results = self._get_threshold_mean(setting, y_measure, base)
            
            reject_mean = [result[1][1] for result in results]

            axes.plot(reject_mean,
                      [result[0][1] for result in results], 
                      color = color,
                      linestyle = setting_threshold[base]["linestyle"])
            
            plt.fill_between(reject_mean,
                              [result[0][0] for result in results],
                              [result[0][2] for result in results],
                              facecolor = color,
                              alpha = 0.2)
            
        return fig, axes
            
    def _get_rejector_mean(self, setting):
        
        def culc_one_measure(measure):
            
            list_ = np.array([mean[model][measure] for model in models])
            
            return np.min(list_), np.mean(list_), np.max(list_) 
        
        mean, std = self._get_result(setting)
        
        model_setting = make_model_setting()
        
        models = model_setting.keys()
        
        measures = ["acc_train", "reject_train", "acc_test", "reject_test"]
        
        return [culc_one_measure(measure) for measure in measures]

                
    
    def plot_ARC(self, setting, x_measures = ["reject_train", "reject_test"], y_measures = ["acc_train", "acc_test"]):
        
        plt.rcParams["font.family"] = "Times New Roman"     #全体のフォントを設定
        plt.rcParams["font.size"] = 14                      #フォントの大きさ
        plt.rcParams["xtick.minor.visible"] = False         #x軸補助目盛りの追加
        plt.rcParams["ytick.direction"] = "out"             #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["ytick.major.size"] = 10               #y軸主目盛り線の長さ
        plt.rcParams['figure.subplot.bottom'] = 0.15
        
        mean, std = self._get_result(setting)
        
        result_rejector = self._get_rejector_mean(setting)
        
        i = 0
        
        for x_measure, y_measure in zip(x_measures, y_measures):
            
            xMin = 0.0
            xMax = result_rejector[i+1][2]
            # xMax = 0.95
            yMin = mean["base"][y_measure]
            # yMin = 0.0
            yMax = 0.95

            fig, axes = self.figSetting(xMin, xMax, yMin, yMax)
            
            self.plot_threshold_base(setting, y_measure, 1, fig, axes)
            color = setting['color']
            size = setting['size']
            linewidths = 1
            edgecolors = 'black'
            alpha = 0.8
            plt.rcParams['font.family'] = 'Times New Roman'
            
            x_min, x_mean, x_max = result_rejector[i+1]
            y_min, y_mean, y_max = result_rejector[i]
            
            axes.plot([x_mean, x_mean],
                      [y_min, y_max],
                      # s = size,
                      marker = "_",
                      color = "k",
                      zorder = 1
                      # linewidths = linewidths
                      )

            axes.plot([x_min, x_max],
                      [y_mean, y_mean],
                      # s = size,
                      marker = "|",
                      color = "k",
                      zorder = 1
                      # linewidths = linewidths
                      )
            
            axes.scatter(x_mean,
                         y_mean,
                         s = size,
                         marker = "o",
                         color = color,
                         linewidths = linewidths,
                         edgecolors = edgecolors,
                         # alpha = alpha,
                         zorder = 2)
            
        
            self.plot_threshold_base(setting, y_measure, 2, fig, axes)     
            
            dataset = setting["dataset"]
            
            output_dir = f"../results/plots/thesis/{dataset}/"

            if not os.path.exists(output_dir):

                os.makedirs(output_dir)
    
            fig.savefig(f"{output_dir}/{dataset}_{y_measure}.pdf")
        
            i = i + 2

        return fig, axes
                
    
def run_dataset(dataset):
    
    setting = make_setting(dataset)
    # fig, axes = plt.subplots(1, 1, figsize=(7, 7))
    # plotRejectorARC().plot_threshold_base(setting[0], "acc_test", fig, axes)
    
    plotMLModelARC().plot_ARC(setting)
    
    return f"finished_{dataset}"
       
    
if __name__ == "__main__":
    
    datasets = ["pima", "australian", "vehicle", "heart", "phoneme", "mammographic", "ring", "spambase", "sonar", "segment", "bupa", "bal", "ecoli", "page-blocks", "wisconsin"]
    
    # datasets = ["mammographic"]
    # datasets = ["vehicle"]
    
    for dataset in datasets:
        
        print(run_dataset(dataset))