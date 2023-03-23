# -*- coding: utf-8 -*-
"""
main class
Eric との比較実験用
ただし，LinearSVCに限り，predict_proba関数がないので比較出来ない．

results直下のディレクトリ名で手法を指定しているので使用する際には注意
"""


from ThresholdOptimization import predict_proba_transformer
from Runner import runner
from CIlab_function import CIlab
from ThresholdBaseRejection import SingleThreshold, ClassWiseThreshold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
import sys


def parameter(algorithm):
    
    if algorithm == "kNN":
        
        model = KNeighborsClassifier()
    
        gridserch_param = {"n_neighbors" : [2, 3, 4, 5, 6]}
    
        thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}
        
        return model, gridserch_param, thresh_param

    if algorithm == "DecisionTree":
        
        model = DecisionTreeClassifier()
         
        gridserch_param = {"max_depth" : [5, 10, 20]}
    
        thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}
    
        return model, gridserch_param, thresh_param
    
    if algorithm == "Adaboost":
        
        model = AdaBoostClassifier(algorithm = 'SAMME')
        
        gridserch_param = {}

        thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}
   
        return model, gridserch_param, thresh_param

    if algorithm == "NaiveBayes":
        
        model = GaussianNB()
        
        gridserch_param = {}

        thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}

        return model, gridserch_param, thresh_param
    
    if algorithm == "GaussianProcess":
        
        model = GaussianProcessClassifier()
        
        gridserch_param = {}

        thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}

        return model, gridserch_param, thresh_param
    
    if algorithm == "MLP":
        
        model = MLPClassifier()
        
        gridserch_param = {"activation" : ["relu"], "alpha" : [1e-5, 1e2]}

        thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}

        return model, gridserch_param, thresh_param
    
    if algorithm == "RF":
        
        model = RandomForestClassifier()
        
        gridserch_param = {"max_depth" : [5, 10, 20]}

        thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}

        return model, gridserch_param, thresh_param
    
    # if algorithm == "SVM":
        
    #     model = LinearSVC()
        
    #     gridserch_param = {"C" : [2 ** -5, 2 ** 15]}

    #     thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.01]}

    #     return model, gridserch_param, thresh_param

    return 

def main():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test = args[1:]
    
    model, gridserch_param, thresh_param = parameter(algorithmID)
    
    run = runner(dataset, algorithmID, experimentID, fname_train, fname_test)
    
    run.output_const(dict(**gridserch_param, **thresh_param))
    
    best_model = model
    
    if gridserch_param != None:
        
        best_model = run.grid_search(model, gridserch_param)    

    pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(best_model)),
                             ('estimator', SingleThreshold())])
    
    run.run(pipe, ParameterGrid(thresh_param), "train-single.csv", "test-single.csv")
    
    
    pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(best_model)),
                             ('estimator', ClassWiseThreshold())])

    run.run(pipe, ParameterGrid(thresh_param), "train-cwt.csv", "test-cwt.csv")
 

if __name__ == "__main__":
    
    main()
    
    # model = AdaBoostClassifier(algorithm = 'SAMME')
        
    # gridserch_param = {}
  
    # thresh_param = {"kmax" : [700], "Rmax" : np.arange(0, 0.51, 0.01), "deltaT" : [0.001]}
    
    # dataset = "pima"
    
    # fname_train = f"..\\dataset\\{dataset}\\a0_0_{dataset}-10tra.dat"
                 
    # fname_test = f"..\\dataset\\{dataset}\\a0_0_{dataset}-10tst.dat"
   
    # X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, "numpy")
    
    
    # run = runner("pima-ada",
    #               "RO-test",
    #               "trial00-v2",
    #               fname_train,
    #               fname_test)
    
    # best_model = model
    
    # if gridserch_param != None:
        
    #     best_model = run.grid_search(model, gridserch_param)    
        
    # pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(best_model)),
    #                          ('estimator', ClassWiseThreshold())])

    # run.run(pipe, ParameterGrid(thresh_param), "train-cwt.csv", "test-cwt.csv")
    
