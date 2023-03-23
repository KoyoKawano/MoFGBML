# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:35:19 2022

@author: kawano
"""


import subprocess
from concurrent.futures import ThreadPoolExecutor


def detection_async(_request):
    
    result = subprocess.run(["python",
                             _request["pythonFile"],
                             _request["dataset"],
                             _request["algroithmID"],
                             _request["experimentID"],
                             _request["trainFile"],
                             _request["testFile"]])
    
    return result


def detection_async_parallel(_requests):
    
    results = []
    
    with ThreadPoolExecutor(max_workers = 6) as executor:
        
        for result in executor.map(detection_async, _requests):
            
            results.append(result)
            
    return results

def run(_Dataset, algorithm):
    
    requests = [{"dataset" : _Dataset, 
                 "pythonFile" : "main.py", 
                 "algroithmID" : algorithm,
                 "experimentID" : f"trial{i}{j}",
                 "trainFile" : f"..\dataset\{_Dataset}\\a{i}_{j}_{_Dataset}-10tra.dat",
                 "testFile" : f"..\dataset\{_Dataset}\\a{i}_{j}_{_Dataset}-10tst.dat"} \
                 for i in range(3) for j in range(10)]
    
    for result in detection_async_parallel(requests):
        
        print(result)

if __name__ == "__main__":
    
    Datasets = ["pima","australian","vehicle", "vowel", "heart", "phoneme", "penbased", "satimage", "texture"]

    algorithm = ["kNN", "MLP", "RF"]
    
    for dataset in Datasets:
        
        for algorithmID in algorithm:
        
            run(dataset, algorithmID)