# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 01:04:14 2022

@author: kawano
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor


def detection_async(request):
    
    result = subprocess.run(["python",
                             request["pythonFile"],
                             request["dataset"],
                             request["algroithmID"],
                             request["experimentID"],
                             request["trainFile"],
                             request["testFile"]])
    
    return result


def detection_async_parallel(requests):
    
    results = []
    
    with ThreadPoolExecutor(max_workers = 5) as executor:
        
        for result in executor.map(detection_async, requests):
            
            results.append(result)
            
    return results

def run(dataset, algorithm):
    
    requests = [{"dataset" : dataset, 
                 "pythonFile" : "NoErrorDataset.py", 
                 "algroithmID" : algorithm,
                 "experimentID" : f"trial{i}{j}",
                 "trainFile" : f"..\dataset\{dataset}\\a{i}_{j}_{dataset}-10tra.dat",
                 "testFile" : f"..\dataset\{dataset}\\a{i}_{j}_{dataset}-10tst.dat"} \
                 for i in range(3) for j in range(10)]
    
    for result in detection_async_parallel(requests):
        
        print(result)

if __name__ == "__main__":
    
    Datasets = ["pima","australian","vehicle", "heart", "phoneme", "penbased", "satimage"]

    algorithm = ["Adaboost", "DecisionTree", "NaiveBayes", "GaussianProcess", "kNN", "MLP", "RF", "SVM"]
    
    for dataset in Datasets:
        
        for algorithmID in algorithm:
        
            run(dataset, algorithmID)