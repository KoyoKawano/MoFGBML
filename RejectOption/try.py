# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:09:28 2023

@author: kawano
"""
from CIlab_FuzzyFunction import FuzzyFunction
from CIlab_function import CIlab
from ThresholdOptimization import predict_proba_transformer
from Runner import runner
from sklearn.neighbors import KNeighborsClassifier
from ThresholdBaseRejection import SingleThreshold, ClassWiseThreshold, RuleWiseThreshold, SecondStageRejectOption
from ThresholdOptimization import ThresholdEstimator
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt


from Cilab_Classifier import FuzzyClassifier
from FuzzyClassifierFromCSV import FileInput
dataset = "pima"
    
fname_train = f"../dataset/{dataset}/a0_0_{dataset}-10tra.dat"
             

fname_test = f"../dataset/{dataset}/a0_0_{dataset}-10tst.dat"


X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, "numpy")

run = runner(dataset, "RO-test", "trial00-v3", fname_train, fname_test)

rr = 0
cc = 0

fuzzy_cl = f"../results/MoFGBML_Basic/{dataset}/trial{rr}{cc}/VAR-0000600000.csv"


best_model = FileInput.best_classifier(fuzzy_cl, X_train, y_train)

param = {"kmax" : [700], "Rmax" : [0, 0.5], "deltaT" : [0.001]}

pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(best_model)),
                          ('estimator', ClassWiseThreshold())])

# pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(best_model, base = "rule")),
#                           ('estimator', RuleWiseThreshold(best_model.ruleset))])


second_model = KNeighborsClassifier()

thresh_estimator = ThresholdEstimator(pipe, param)

run.run_second_stage(pipe, ParameterGrid(param), ["kNN"], "train-rule.csv", "test-rule.csv")