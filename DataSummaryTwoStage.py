# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:59:46 2021

@author: kawano
"""

import pandas as pd
import statistics
from collections import defaultdict

sep = "\\"

Dataset = 'vehicle'

folder = 'results' + sep + 'TwoStage' + sep + Dataset + sep


"""
fuction SummaryOneTrial
this function return dictionary(max rule number, min error rate for train, min error rate for test)
for one trial.
@param Dataset : String
@param trial : String , it is the trial number. 
return dictionary of: 
           max the number of rule,
           min error rate for the training data,
           min error rate for the test data
"""
def SummaryResultsOneTrial(Dataset, trial):
    df_results = pd.read_csv(folder + Dataset + "_trial" + trial + sep + 'results.csv')
    
    errorRate_Dtra = min(df_results.errorRate_Dtra)
    
    errorRate_Dtst = min(df_results.errorRate_Dtst)
    
    maxRuleNum = max(df_results.ruleNum)
    
    maxRuleLength = max(df_results.ruleLength)
        
    return {"errorRate_Dtra" : errorRate_Dtra, "errorRate_Dtst" : errorRate_Dtst, "maxRuleNum" : maxRuleNum, "maxRuleLength" : maxRuleLength}

def SummaryFUNOneTrial(Dataset, trial):
    trialFolder = folder + sep + Dataset + "_trial" + trial + sep
    
    start = 3000
    end = 300000
    freq = 3000
    evaluationFile = [trialFolder + "FUN-" + str(i) + ".csv" for i in range(start, end + 1, freq)]
    return 0

def SummaryByEvaluation(Dataset, evaluate, RR, CC):
    
    #evaluation = [str(freq)]
    df_FUN = 1
    return df_FUN

trial = "00"
trialFolder = folder + sep + Dataset + "_trial" + trial + sep

start = 3000
end = 300000
freq = 3000
evaluationFile = [trialFolder + "FUN-" + str(i) + ".csv" for i in range(start, end + 1, freq)]
dfList = list(map(lambda x : pd.read_csv(x), evaluationFile))

countTrial = {str(i+1):0 for i in range(100)}
for df in dfList:
    for ruleNum in range(1, 100, 1):
        if(any(df['f1'].isin([ruleNum]))):
            countTrial[str(ruleNum)] += 1



#dfDicList = [dfList[i] for i in len(dfList)]



# #test SummaryOneTrial
# RR = 1
# CC = 10
# #make trial number rr = {0,1,2}, cc = {0,1,...9}
# trial = [str(rr) + str(cc) for rr in range(RR) for cc in range(CC)]

# results = list(map(lambda x : SummaryOneTrial(Dataset, x), trial))

# ruleNum = statistics.mean([results[trial]["maxRuleNum"] for trial in range(len(results))])


