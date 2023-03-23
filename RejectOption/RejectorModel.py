# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:15:13 2022

@author: kawano
"""

"""
リサンプル後のデータセットを出力する必要あり．
リサンプル後のデータセットは従来も再学習も同様のリジェクターを使う
"""
from CIlab_function import CIlab
import numpy as np
from imblearn.over_sampling import ADASYN


class ResampledTransformer():
    
    def __init__(self, base_model, resample = ADASYN(random_state = 2022)):
        
        self.base_model = base_model
        
        self.resample = resample
        
    
    def fit(self, X, y):
        
        return self
    
    def R(self, X, y):
            
        reject_index = self.base_model.predict(X) != y
        
        X_reject = X[reject_index]
        
        X_accept = X[~reject_index]
        
        y_reject = np.ones(len(X_reject)) * -1
        
        y_accept = np.ones(len(X_accept))
        
        return np.r_[X_reject, X_accept], np.r_[y_reject, y_accept] 

    def transform(self, X, y):
        
        self.base_model.fit(X, y)
        
        X_new, y_new = self.R(X, y)
         
        X_resampled, y_resampled = X_new, y_new
        
        if len(set(y_new)) > 1:
            
            try:
                X_resampled, y_resampled = self.resample.fit_resample(X_new, y_new)
        
            except Exception:
                print("can't resampled")
                
            else:
                pass
            
        return X_resampled, y_resampled
    
    
    def output(self, X, y, output_dir, fname):
        
        X_resampled, y_resampled = self.transform(X, y)
        
        CIlab.output_cilab_style_dataset(X, y, output_dir, fname)
        
        return


class Rejector():
    
    def __init__(self, rejector_model):
        
        self.rejector_model = rejector_model
        
        
    def fit(self, X, y):
        
        self.rejector_model.fit(X, y)
        
        return self
    
    
    def predict(self, X):
        
        return self.rejector_model.predict(X)
    
    
    def isReject(self, X):
        
        return self.predict(X) == -1

    

        