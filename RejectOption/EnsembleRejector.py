# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:12:54 2023

@author: kawano
"""
from functools import reduce 
import numpy as np
from RejectorModel import Rejector, ResampledTransformer 
from CIlab_function import CIlab
from GridSearchParameter import GridSearch
import sys
from RejectorBaseRejection import RejectorBasedRejectOption
from FuzzyClassifierFromCSV import FileInput


class EnsembleRejectorModel():
    
    def __init__(self, models):
        
        self.models = models
        
        
    def fit(self, X, y):
        
        for model in self.models:
            
            X_resampled, y_resampled = ResampledTransformer(model).transform(X, y)
            
            model.fit(X_resampled, y_resampled)
            
        return self

    
    def isReject(self, X):
        
        reject_index = np.array([model.predict(X) == -1 for model in self.models])
        
        reject_index = np.sum(reject_index, axis = 0)
            
        reject_index = reject_index >= ((len(self.models) + 1) / 2)
        
        return reject_index
            
    def transform(self, X, y):
        
        reject_index = self.isReject(X)
        
        return X[~reject_index], y[~reject_index]
        
        
    
    

def make_dataset():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    models_name = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "LinearSVC", "RBFSVC"]
    
    output_dir = f"../results/{algorithmID}/{dataset}/"
    
    models = [GridSearch.run_grid_search(model, X_train, y_train, f"{output_dir}/{model}/", f"gs_result_{experimentID}.csv") for model in models_name]
    
    rejector = EnsembleRejectorModel(models)
    
    rejector.fit(X_train, y_train)
    
    X_new, y_new = rejector.transform(X_train, y_train)
    
    trial = list(experimentID)
    
    fname = f"a{trial[5]}_{trial[6]}_{dataset}-10tra.dat"
    
    CIlab.output_cilab_style_dataset(X_new, y_new, output_dir, fname)

    
def main():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test, fname_clf = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    base_model = FileInput.best_classifier(fname_clf, X_train, y_train)
    
    models_name = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "LinearSVC", "RBFSVC"]
    
    output_dir = f"../results/ensemble_rejector/{algorithmID}/{dataset}/{experimentID}/"
    
    models = [GridSearch.run_grid_search(model, X_train, y_train, f"{output_dir}/{model}/", f"gs_result_{experimentID}.csv") for model in models_name]
    
    rejector = EnsembleRejectorModel(models)
    
    rejector.fit(X_train, y_train)
    
    rejector_based = RejectorBasedRejectOption(base_model, rejector)
    
    result = {"ensemble" : list(rejector_based.score(X_train, y_train).values()) + list(rejector_based.score(X_test, y_test).values())}
    
    RejectorBasedRejectOption.output_result(result, output_dir, "result.csv")

    
if __name__ == "__main__" :
    
    main()
    
    # dataset = "australian"
    
    # rr = 0
    # cc = 2
    
    # fname_train = f"..\\dataset\\{dataset}\\a{rr}_{cc}_{dataset}-10tra.dat"
                 
    # fname_test = f"..\\dataset\\{dataset}\\a{rr}_{cc}_{dataset}-10tst.dat"
    
    # output = "../results/test/"
    # X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    # models_name = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "LinearSVC", "RBFSVC"]
    
    # models = [GridSearch.run_grid_search(model, X_train, y_train, f"{output}/{model}/", f"gs_result_{model}.csv") for model in models_name]
    
    # rejector = EnsembleRejectorModel(models)
    
    # rejector.fit(X_train, y_train)
    
    # X_new, y_new = rejector.transform(X_train, y_train)
    
    # CIlab.output_cilab_style_dataset(X_new, y_new, output, f"a{rr}_{cc}_{dataset}-10tra.dat")

