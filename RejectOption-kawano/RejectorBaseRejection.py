# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:56:24 2023

@author: kawano
"""

import numpy as np
import pandas as pd
import os

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
        
    
    

