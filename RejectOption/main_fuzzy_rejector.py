# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:45:31 2023

@author: kawano
"""
from CIlab_function import CIlab
from GridSearchParameter import GridSearch
from FuzzyClassifierFromCSV import FileInput
import sys
from RejectorModel import Rejector, ResampledTransformer
from RejectorBaseRejection import RejectorBasedRejectOption

def main():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test, clf_name = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    base_model = FileInput.best_classifier(clf_name, X_train, y_train)
    
    X_resampled, y_resampled = ResampledTransformer(base_model).transform(X_train, y_train)
     
    algorithms = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "LinearSVC"]
    
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
    
    main()
    