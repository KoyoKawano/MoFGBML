# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:08:07 2022

@author: kawano
"""
from sklearn.neighbors import KNeighborsClassifier
from CIlab_function import CIlab
from FuzzyClassifierFromCSV import FileInput
import numpy as np
import sys
from GridSearchParameter import GridSearch


class NoErrorDataMaker():
    
    def __init__(self, model):
        
        self.model = model
        
    def make(self, X, y):
        
        self.model.fit(X, y)
        
        true_index = self.model.predict(X) == y
        
        return X[true_index], y[true_index]



    
def main_mofgbml():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test, clf_name = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    fuzzy_clf = FileInput.best_classifier(clf_name, X_train, y_train)
    
    X_new, y_new = NoErrorDataMaker(fuzzy_clf).make(X_train, y_train)
    
    trial = list(experimentID)
    
    CIlab.output_cilab_style_dataset(X_new, y_new, f"../NoErrorDataset/{algorithmID}/{dataset}/", f"a{trial[5]}_{trial[6]}_{dataset}-10tra.dat")


def main_sklearn():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    clf = GridSearch.run_grid_search(algorithmID, X_train, y_train, f"../NoErrorDataset/{algorithmID}/{dataset}/", f"gs_result_{dataset}_{experimentID}.csv")

    X_new, y_new = NoErrorDataMaker(clf).make(X_train, y_train)
    
    
    trial = list(experimentID)
    
    CIlab.output_cilab_style_dataset(X_new, y_new, f"../NoErrorDataset/{algorithmID}/{dataset}/", f"a{trial[5]}_{trial[6]}_{dataset}-10tra.dat")
 

if __name__ == "__main__":
    
    main_mofgbml()
    
    # dataset = "pima"
    
    # fname_train = f"../dataset/{dataset}/a0_0_{dataset}-10tra.dat"
                 
    # fname_test = f"../dataset/{dataset}/a0_0_{dataset}-10tst.dat"
    
    # X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    # model = KNeighborsClassifier(n_neighbors = 3)

    # model.fit(X_train, y_train)

    # print(model.score(X_train, y_train))
    
    # X_new, y_new = NoErrorDataMaker(model).make(X_train, y_train)
    
    # CIlab.output_cilab_style_dataset(X_new, y_new, f"../newdata/{dataset}/trial00/", "test.dat")
    
    # print(model.score(X_new, y_new))
