# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 18:22:16 2022

@author: kawano
"""

from ThresholdOptimization import predict_proba_transformer
from Runner import runner
from ThresholdBaseRejection import SingleThreshold, ClassWiseThreshold, RuleWiseThreshold
from FuzzyClassifierFromCSV import FileInput
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
import sys
from CIlab_function import CIlab
from GridSearchParameter import GridSearch


def main():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test, clf_name = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    run = runner(dataset, algorithmID, experimentID, fname_train, fname_test)
    
    thresh_param = {"kmax" : [400], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}

    fuzzy_clf = FileInput.best_classifier(clf_name, X_train, y_train)

    second_keys = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "LinearSVC"]
        
    output_dir = f"../results/threshold_base/{algorithmID}/{dataset}/{experimentID}/"

    second_values = [GridSearch.run_grid_search(key, X_train, y_train, f"{output_dir}/{key}/", f"gs_result_{key}.csv").fit(X_train, y_train) for key in second_keys]

    second_dict = { key: value for key, value in zip(second_keys, second_values)} 

    transfomer = predict_proba_transformer(fuzzy_clf).fit(X_train, y_train)


    pipe = Pipeline(steps = [('predict_proba_transform', transfomer),
                             ('estimator', SingleThreshold())])
    
    run.run_second_stage(pipe, ParameterGrid(thresh_param), second_dict, "train-single.csv", "test-single.csv")
    
    
    # pipe = Pipeline(steps = [('predict_proba_transform', transfomer),
    #                          ('estimator', ClassWiseThreshold())])

    # run.run_second_stage(pipe, ParameterGrid(thresh_param), second_dict, "train-cwt.csv", "test-cwt.csv")


    # pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(fuzzy_clf, base = "rule")),
    #                          ('estimator', RuleWiseThreshold(fuzzy_clf.ruleset))])

    # run.run_second_stage(pipe, ParameterGrid(thresh_param), second_dict, "train-rwt.csv", "test-rwt.csv")
    
    
if __name__ == "__main__":
    
    main()
    