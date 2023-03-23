# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:14:24 2023

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
            {"dataset" : dataset, "marker":"o", "color":"tab:blue", "size":200, "algorithmID" : "MoFGBML_Basic", "train_dir" : f"../dataset/{dataset}/", "test_dir" : f"../dataset/{dataset}/"}
            # {"dataset" : dataset, "marker":"o", "color":"tab:orange", "size":200, "algorithmID" : "MoFGBML_Basic_ensemble", "train_dir" : f"../results/ensemble_dataset/{dataset}/", "test_dir" : f"../dataset/{dataset}/"}
            ]
    

class plotMoFGBML():
    
    def __init__(self):
        
        pass
    
    def figSetting(xMin, xMax, yMin, yMax):
        
        # figsizeで図のサイズを指定 横長にしたいなら左の数値を大きくする．
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))
        
        # X ticks (= Number of rule)
        xH = 2
        xticks = np.arange(0, xMax + xH, xH)
    
        if xMin < 2:
            xticks[0] = 2
        else:
            xticks[0] = xMin
        
        xMin = xticks[0]
        xMax = xticks[len(xticks) - 1]
        axes.set_xlim(xMin - 2, xMax + 2)
        axes.set_xticks(xticks)
        
        # Y ticks
        yH = 0.05

            
        yMin = (int)(yMin / yH)
        yMin = yMin * yH
        yMax = (int)((yMax + yH) / yH)
        yMax = yMax * yH  
        
        yH = 0.1
        
        if yMax > 1:
            yH = 1
            
        yticks = np.arange(yMin, yMax+yH, yH)
        yMin = yticks[0]
        yMax = yticks[-1]
        axes.set_ylim(yMin - 0.02, yMax + 0.02)
        axes.set_yticks(yticks)
        
        axes.grid(linewidth=0.4)
        axes.yaxis.set_minor_locator(AutoMinorLocator(3))
        axes.tick_params(which = 'major', length = 8, color = 'black', labelsize = 25)
        axes.tick_params(which = 'minor', length = 5, color = 'black', labelsize = 25)
        
        return fig, axes

    # =============================================================================
    # getAvaragePopulation_Results:
    # 1つのMOP・データセットの結果を集計し，プロットに使うdictionaryを返す．
    # =============================================================================
    
    def _get_avarage_population(setting):
        
        dataset = setting["dataset"]
        
        train_dir = setting["test_dir"]
        test_dir = setting["test_dir"]
        algorithmID = "../results/" + setting["algorithmID"]
        
        
        RR = 3
        CC = 10
        
        files = [f"{algorithmID}/{dataset}/trial{rr}{cc}/VAR-0000600000.csv" for rr in range(RR) for cc in range(CC)]
        
        trains = [f"{train_dir}/a{rr}_{cc}_{dataset}-10tra.dat" for rr in range(RR) for cc in range(CC)]
        
        tests = [f"{test_dir}/a{rr}_{cc}_{dataset}-10tst.dat" for rr in range(RR) for cc in range(CC)]
        
        data_list = [CIlab.load_train_test(train, test, type_ = "numpy") for train, test in zip(trains, tests)]
            
        fuzzy_clf_list = [FileInput.input_VAR(file) for file in files]
        
        df_list  = [FileInput.to_dataFrame(fuzzy_clf, data[0], data[1], data[2], data[3]) for fuzzy_clf, data in zip(fuzzy_clf_list, data_list)]
        
        max_rule_num = 60

        min_rule_num = 2
        
        values = {str(i + 1) : [] for i in range(max_rule_num)}
    
        for df in df_list:
            
            for rule_num in range(min_rule_num, max_rule_num, 1):
                
                if(any(df['rule_num'].isin([rule_num]))):
                    
                    values[str(rule_num)].append(df[df['rule_num'] == rule_num].mean())
            
        criteria = 16
        
        plotIndividual = filter(lambda x : len(x) >= criteria, values.values())
        plotIndividual = list(map(lambda x : pd.concat(x), plotIndividual))
        
        measures = df_list[0].columns.values
        
        return {measure : list(map(lambda x : x[measure].mean(), plotIndividual)) for measure in measures}
        
    
    # =============================================================================
    # plotAveragePopulation_Results:
    # settingList内の結果を集計し，プロットを行う．
    # =============================================================================
    def plot_average_population(settings, x_measure = "rule_num", y_measures = ["acc_train", "acc_test", "rule_length"]):
        
        results = [plotMoFGBML._get_avarage_population(setting) for setting in settings]   
        
        plt.rcParams["font.family"] = "Times New Roman"     #全体のフォントを設定
        plt.rcParams["font.size"] = 14                      #フォントの大きさ
        plt.rcParams["xtick.minor.visible"] = False         #x軸補助目盛りの追加
        plt.rcParams["ytick.direction"] = "out"             #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["ytick.major.size"] = 10               #y軸主目盛り線の長さ
        plt.rcParams['figure.subplot.bottom'] = 0.15
        
        for y_measure in y_measures:
            
            xMin = min([min(result[x_measure]) for result in results])
            xMax = max([max(result[x_measure]) for result in results])
            yMin = min([min(result[y_measure]) for result in results])
            yMax = max([max(result[y_measure]) for result in results])
                
            fig, axes = plotMoFGBML.figSetting(xMin, xMax, yMin, yMax)
            
            for result, setting in zip(results, settings):   
                
                marker = setting['marker']
                color = setting['color']
                size = setting['size']
                linewidths = 1
                edgecolors = 'black'
                alpha = 0.8
                plt.rcParams['font.family'] = 'Times New Roman'
                
                axes.scatter(result[x_measure],
                             result[y_measure], 
                             s = size,
                             marker = marker,
                             color = color,
                             # label = label,
                             linewidths = linewidths,
                             edgecolors = edgecolors,
                             alpha = alpha)
                        
            # axes.set_xlabel("Number of rules", fontsize = 25)
            # axes.set_ylabel(ytickMeasure, fontsize = 25)
            # plt.legend()
            
            dataset = settings[0]["dataset"]
            algorithm0 = settings[0]["algorithmID"]
            
            output_dir = f"../results/plots/pop_/{algorithm0}/{dataset}/"
            
            if not os.path.exists(output_dir):

                os.makedirs(output_dir)
    
            fig.savefig(f"{output_dir}{y_measure}.png", dpi = 300)
        
        return fig, axes

def run_dataset(dataset):
    
    settings = make_setting(dataset)
    
    plotMoFGBML.plot_average_population(settings)
    
    return f"finished_{dataset}"

if __name__ == "__main__":

    # datasets = ["pima", "vehicle", "heart", "phoneme", "wisconsin", "segment"]
    # datasets = ["page-blocks"]
    datasets = ["pima", "australian", "vehicle", "heart", "phoneme", "mammographic", "ring", "spambase", "sonar", "segment", "bupa", "bal", "ecoli", "page-blocks", "wisconsin"]

    for dataset in datasets:
        
        print(run_dataset(dataset))
    
    
    
        
        