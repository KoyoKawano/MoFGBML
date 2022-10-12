# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 01:04:14 2022

@author: kawano
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor


def detection_async(_request):
    
    result = subprocess.run(["Java", "-jar",
                             _request["jarFile"],
                             _request["dataset"],
                             _request["algroithmID"],
                             _request["experimentID"],
                             _request["parallelCores"],
                             _request["trainFile"],
                             _request["testFile"]])
    
    return result


def detection_async_parallel(_requests):
    
    results = []
    
    with ThreadPoolExecutor(max_workers = 5) as executor:
        
        for result in executor.map(detection_async, _requests):
            
            results.append(result)
            
    return results


if __name__ == "__main__":
    
    Dataset = "iris"
    
    requests = [{"trial" : f"{i}_{j}",
                 "dataset" : Dataset, 
                 "jarFile" : "target\FAN2021.jar", 
                 "algroithmID" : "FAN2021_test3",
                 "parallelCores" : "4",
                 "experimentID" : f"trial{i}{j}",
                 "trainFile" : f"dataset\\{Dataset}\\a{i}_{j}_{Dataset}-10tra.dat",
                 "testFile" : f"dataset\\{Dataset}\\a{i}_{j}_{Dataset}-10tst.dat"} \
                 for i in range(3) for j in range(10)]
    
    for result in detection_async_parallel(requests):
        
        print(result)