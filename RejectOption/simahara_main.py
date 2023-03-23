# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:57:37 2022

@author: kawano
"""

# -*- coding: utf-8 -*-
"""
main class
Eric との比較実験用
ただし，LinearSVCに限り，predict_proba関数がないので比較出来ない．

results直下のディレクトリ名で手法を指定しているので使用する際には注意
"""


from ThresholdOptimization import predict_proba_transformer
from Runner import runner
from ThresholdBaseRejection import SingleThreshold, ClassWiseThreshold, RuleWiseThreshold
from Cilab_Classifier import FuzzyClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
import sys


def parameter(algorithm):
    
    if algorithm == "kNN":
        
        model = KNeighborsClassifier()
    
        gridserch_param = {"n_neighbors" : [2, 3, 4, 5, 6]}
    
        return model, gridserch_param

    if algorithm == "DecisionTree":
        
        model = DecisionTreeClassifier()
         
        gridserch_param = {"max_depth" : [5, 10, 20]}
    
        return model, gridserch_param
    
    if algorithm == "Adaboost":
        
        model = AdaBoostClassifier(algorithm = 'SAMME')
        
        gridserch_param = {}

        return model, gridserch_param

    if algorithm == "NaiveBayes":
        
        model = GaussianNB()
        
        gridserch_param = {}

        return model, gridserch_param
    
    if algorithm == "GaussianProcess":
        
        model = GaussianProcessClassifier()
        
        gridserch_param = {}

        return model, gridserch_param
    
    if algorithm == "MLP":
        
        model = MLPClassifier()
        
        gridserch_param = {"activation" : ["relu"], "alpha" : [1e-5, 1e2]}

        return model, gridserch_param
    
    if algorithm == "RF":
        
        model = RandomForestClassifier()
        
        gridserch_param = {"max_depth" : [5, 10, 20]}

        return model, gridserch_param
    
    if algorithm == "SVM":
        
        model = LinearSVC()
        
        gridserch_param = {"C" : [2 ** -5, 2 ** 15]}

        return model, gridserch_param

    return 

def main():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test, clf_name = args[1:]
    
    run = runner(dataset, algorithmID, experimentID, fname_train, fname_test)
    
    model, gridserch_param = parameter(algorithmID)
    
    thresh_param = {"kmax" : [400], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}
    
    run.output_const(dict(**gridserch_param, **thresh_param))
    
    second_model = model
    
    if gridserch_param != None:
        
        second_model = run.grid_search(model, gridserch_param)
        
        
    fuzzy_clf = FuzzyClassifier()
    
    fuzzy_clf.set_ruleset_csv(clf_name)
    
        

    pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(fuzzy_clf)),
                             ('estimator', SingleThreshold())])
    
    run.run_second_stage(pipe, ParameterGrid(thresh_param), second_model, "train-single.csv", "test-single.csv")
    
    
    pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(fuzzy_clf)),
                             ('estimator', ClassWiseThreshold())])

    run.run_second_stage(pipe, ParameterGrid(thresh_param), second_model, "train-cwt.csv", "test-cwt.csv")


    pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(fuzzy_clf, base = "rule")),
                             ('estimator', RuleWiseThreshold(fuzzy_clf.ruleset))])

    run.run_second_stage(pipe, ParameterGrid(thresh_param), second_model, "train-rwt.csv", "test-rwt.csv")
    
    
if __name__ == "__main__":
    
    main()
    