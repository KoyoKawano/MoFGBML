# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:02:22 2023

@author: kawano
"""

from CIlab_function import CIlab
from GridSearchParameter import GridSearch
import sys
from EnsembleRejector import EnsembleRejectorModel

def main():
    
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

if __name__ == "__main__" :
    
    main()