# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:04:41 2023

@author: kawano
"""

from CIlab_function import CIlab
from GridSearchParameter import GridSearch
import sys
from RejectorBaseRejection import RejectorBasedRejectOption
from FuzzyClassifierFromCSV import FileInput
from EnsembleRejector import EnsembleRejectorModel
    
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