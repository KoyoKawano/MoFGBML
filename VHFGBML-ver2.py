# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:51:34 2019

@author: koyo kawano
"""

import subprocess
import multiprocessing as mp
import os

CORE_USING = mp.cpu_count() - 3

#CORE_USING = 1
def function(name):
    subprocess.call(name)


def multi(list_console_command):
    #print("cpu_count =", mp.cpu_count())
    p = mp.Pool(CORE_USING)
    p.map(function, list_console_command)
    p.close()


if __name__ ==  "__main__":
    
    list_console_command = []
                    
    dataset = "iris"
    jarFile = "target\FAN2021.jar"
    algroithmID = "FAN2021_test2"
    parallelCores = "4"

    for i in range(3):
        for j in range(10):
            experimentID = "trial" + str(i) + str(j)
            trainFile = "dataset\\" + dataset + "\\a" + str(i) + "_" + str(j) + "_" + dataset + "-10tra.dat"
            testFile = "dataset\\" + dataset + "\\a" + str(i) + "_" + str(j) + "_" + dataset + "-10tst.dat"

            list_console_command.append(["Java", "-jar",
                                         jarFile,
                                         dataset,
                                         algroithmID,
                                         experimentID,
                                         parallelCores,
                                         trainFile,
                                         testFile])
        multi(list_console_command)
        list_console_command.clear()
    print("done")
    
    