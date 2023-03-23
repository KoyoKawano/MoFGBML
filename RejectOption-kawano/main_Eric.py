# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 19:22:01 2023

@author: kawano
"""

from RejectorModel import ResampledTransformer, Rejector
from RejectorBaseRejection import RejectorBasedRejectOption
from CIlab_function import CIlab
from GridSearchParameter import GridSearch
import sys

class Eric_method():
    
    def __init__(self, dataset, 
                       base_model,
                       experimentID,
                       fname_train,
                       fname_test):        

        self.dataset = dataset
        
        self.algorithmID = base_model
        
        self.experimentID = experimentID
        
        self.X_train, self.X_test, self.y_train, self.y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")

        self.base_model = GridSearch.run_grid_search(base_model, self.X_train, self.y_train, f"../results/rejector/{base_model}/{self.dataset}/{self.experimentID}/", f"gs_result_{base_model}.csv", cv = 5)
            
        self.base_model = self.base_model.fit(self.X_train, self.y_train)        
 
        self.X_resampled, self.y_resampled = ResampledTransformer(base_model = self.base_model).transform(self.X_train, self.y_train) 
        
        
    def run(self, algorithms = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "LinearSVC"]):
        
        def _run_one_rejector_model(rejector_model):
            
            rejector = Rejector(rejector_model).fit(self.X_resampled, self.y_resampled)
            
            rejectorBasedRejectOption = RejectorBasedRejectOption(self.base_model, rejector)
            
            result_train = list(rejectorBasedRejectOption.score(self.X_train, self.y_train).values())
            
            result_test = list(rejectorBasedRejectOption.score(self.X_test, self.y_test).values())
            
            return result_train + result_test
        
        dict_ = {}
        
        dict_["base"] = [self.base_model.score(self.X_train, self.y_train), 0.0, self.base_model.score(self.X_test, self.y_test), 0.0]
    

        for algorithm in algorithms:
            
            if dict_["base"][0] < 1.0:
                rejector_model = GridSearch.run_grid_search(algorithm, self.X_train, self.y_train, f"../results/rejector/{self.algorithmID}/{self.dataset}/{self.experimentID}/", f"gs_result_{algorithm}.csv", cv = 5)
                result = _run_one_rejector_model(rejector_model)
                dict_[algorithm] = result
                
            else:
                dict_[algorithm] = dict_["base"]
            
        
        RejectorBasedRejectOption.output_result(dict_, f"../results/rejector/{self.algorithmID}/{self.dataset}/{self.experimentID}/", "result.csv")
    
        return
            
def main():
    
    args = sys.argv
    
    dataset, base_model, experimentID, fname_train, fname_test = args[1:]
    
    eric = Eric_method(dataset,
                       base_model, 
                       experimentID,
                       fname_train,
                       fname_test)
    
    eric.run()
    
if __name__ == "__main__" :
    
    main()
    
    