# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:56:24 2023

@author: kawano
"""

from CIlab_function import CIlab
import numpy as np
# from MyDataset import DatasetMaker
from GridSearchParameter import GridSearch
from FuzzyClassifierFromCSV import FileInput
import pandas as pd
import os
import sys
from RejectorModel import Rejector


class RejectorBasedRejectOption():
    
    def __init__(self, base_model, rejector):
        
        self.base_model = base_model
        
        self.rejector = rejector
        
        
    def accuracy(self, X, y):
        
        isReject = self.rejector.isReject(X)
        
        predict = self.base_model.predict(X)
        
        len_accept = np.count_nonzero(~isReject)
        
        if len_accept == 0:
            
            return 1.0
        
        return np.count_nonzero((predict == y) & (~isReject)) / len_accept
    
    
    def rejectrate(self, X):
        
        isReject = self.rejector.isReject(X)
        
        return np.count_nonzero(isReject) / len(isReject)
    
    
    def fit(self, X_train, y_train):
        
        self.base_model.fit(X_train, y_train)
        
        return self
    
    
    def score(self, X, y):
        
        return {"accuracy" : self.accuracy(X, y), "rejectrate" : self.rejectrate(X)}


    def output_result(dict_, output_dir, fname):
        
        columns = ["acc_train", "reject_train", "acc_test", "reject_test"]
        
        algorithm = dict_.keys()
        
        data = list(dict_.values())
        
        df = pd.DataFrame(data = data, columns = columns, index = algorithm)
    
        if not os.path.exists(output_dir):
    
            os.makedirs(output_dir)
        
        df.to_csv(output_dir + fname)
        
    
    
def main_mofgbml():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test, clf_name = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    base_model = FileInput.best_classifier(clf_name, X_train, y_train)

    resampled_algorithm = "MoFGBML_Basic"
    
    trial = list(experimentID)
    
    fname_resampled = f"../results/resampled_dataset/{resampled_algorithm}/{dataset}/a{trial[5]}_{trial[6]}_{dataset}-10tra.dat"
    
    X_resampled, y_resampled = CIlab.load_dataset(fname_resampled, type_ = "numpy")
     
    algorithms = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "SVM"]
    
    dict_ = {}
    
    dict_["base"] = [base_model.score(X_train, y_train), 0.0, base_model.score(X_test, y_test), 0.0]
    
    for algorithm in algorithms:
        
        rejector_model = GridSearch.run_grid_search(algorithm, X_train, y_train, f"../results/rejector/{algorithmID}/{dataset}/{experimentID}/", f"gs_result_{algorithm}.csv", cv = 5)
        
        rejector = Rejector(rejector_model).fit(X_resampled, y_resampled)
        
        rejectorBasedRejectOption = RejectorBasedRejectOption(base_model, rejector).fit(X_train, y_train)
        
        result_train = list(rejectorBasedRejectOption.score(X_train, y_train).values())
        
        result_test = list(rejectorBasedRejectOption.score(X_test, y_test).values())
        
        dict_[algorithm] = result_train + result_test
    
    RejectorBasedRejectOption.output_result(dict_, f"../results/rejector/{algorithmID}/{dataset}/{experimentID}/", "result.csv")

    
if __name__ == "__main__" :
    
    main_mofgbml()
    
    # dataset = "pima"
    
    # algorithmID = "MoFGBML_Basic"
    
    # rr = 0
    # cc = 0
    
    # experimentID = f"trial{rr}{cc}"
    
    # fname_train = f"..\\dataset\\{dataset}\\a{rr}_{cc}_{dataset}-10tra.dat"
                 
    # fname_test = f"..\\dataset\\{dataset}\\a{rr}_{cc}_{dataset}-10tst.dat"

    # clf_name = f"../results/{algorithmID}/{dataset}/trial{rr}{cc}/VAR-0000600000.csv"
    
    # X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    
    # base_model = FileInput.best_classifier(clf_name, X_train, y_train)
    
    # resampled_algorithm = "MoFGBML_Basic"
    
    # fname_resampled = f"../results/resampled_dataset/{resampled_algorithm}/{dataset}/a{rr}_{cc}_{dataset}-10tra.dat"
    
    # X_resampled, y_resampled = CIlab.load_dataset(fname_resampled, type_ = "numpy")
     
    # algorithms = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "SVM"]
    
    # dict_ = {}
    
    # dict_["base"] = [base_model.score(X_train, y_train), 0.0, base_model.score(X_test, y_test), 0.0]
    
    # for algorithm in algorithms:
        
    #     rejector_model = GridSearch.run_grid_search(algorithm, X_train, y_train, f"../results/rejector_test/{algorithmID}/{dataset}/{experimentID}/", f"gs_result_{algorithm}.csv", cv = 5)
        
    #     rejector = Rejector(rejector_model).fit(X_resampled, y_resampled)
        
    #     rejectorBasedRejectOption = RejectorBasedRejectOption(base_model, rejector).fit(X_train, y_train)
        
    #     result_train = list(rejectorBasedRejectOption.score(X_train, y_train).values())
        
    #     result_test = list(rejectorBasedRejectOption.score(X_test, y_test).values())
        
    #     dict_[algorithm] = result_train + result_test
    
    # RejectorBasedRejectOption.output_result(dict_, f"../results/rejector_test/{algorithmID}/{dataset}/{experimentID}/", "result.csv")