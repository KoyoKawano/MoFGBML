# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:15:13 2022

@author: kawano
"""

"""
リサンプル後のデータセットを出力する必要あり．
リサンプル後のデータセットは従来も再学習も同様のリジェクターを使う
"""
# from sklearn.neighbors import KNeighborsClassifier
from CIlab_function import CIlab
import numpy as np
from imblearn.over_sampling import ADASYN
from MyDataset import DatasetMaker
from GridSearchParameter import GridSearch
from FuzzyClassifierFromCSV import FileInput
import sys
from sklearn.neighbors import KNeighborsClassifier

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

    
        
def main_mofgbml():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test, clf_name = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")

    base_model = FileInput.best_classifier(clf_name, X_train, y_train)

    resample_transformer = ResampledTransformer(base_model)
    
    X_resampled, y_resampled = resample_transformer.transform(X_train, y_train)
    
    output_dir = f"../results/resampled_dataset/{algorithmID}/{dataset}/"
    
    trial = list(experimentID)
    
    fname = f"a{trial[5]}_{trial[6]}_{dataset}-10tra.dat"
    
    resample_transformer.output(X_resampled, y_resampled, output_dir, fname)
    
    
def main_sklearn():
    
    args = sys.argv
    
    dataset, algorithmID, experimentID, fname_train, fname_test = args[1:]
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
    
    output_dir = f"../results/resampled_dataset/{algorithmID}/{dataset}/"
    
    base_model = GridSearch.run_grid_search(algorithmID, X_train, y_train, output_dir + experimentID + "/", f"gs_result_{algorithmID}.csv")

    resample_transformer = ResampledTransformer(base_model)
    
    X_resampled, y_resampled = resample_transformer.transform(X_train, y_train)
    
    trial = list(experimentID)
    
    fname = f"a{trial[5]}_{trial[6]}_{dataset}-10tra.dat"
    
    resample_transformer.output(X_resampled, y_resampled, output_dir, fname)


if __name__ == "__main__" :
        
    # main_mofgbml()
    
    X, y = DatasetMaker().make_dataset(n_samples = 200, class_sep = 1.5, scale = 1)
    
    DatasetMaker().plot_2d_dataset(X, y)
    
    rejector_model = KNeighborsClassifier(n_neighbors = 3)
    
    rejector = Rejector(rejector_model)
    
    base_model = KNeighborsClassifier(n_neighbors = 3)
    
    rejectorBasedRejectOption = RejectorBasedRejectOption(rejector, base_model).fit(X, y)
    
    # print(base_model.fit(X, y).score(X, y))

    # print(rejectorBasedRejectOption.accuracy(X, y))
    
    # print(rejectorBasedRejectOption.rejectrate(X))
    
    
    # dataset = "vehicle"
    
    # rr = 2
    # cc = 9
    
    # fname_train = f"..\\dataset\\{dataset}\\a{rr}_{cc}_{dataset}-10tra.dat"
                 
    # fname_test = f"..\\dataset\\{dataset}\\a{rr}_{cc}_{dataset}-10tst.dat"
    
    
    # X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")

    # base_model = FileInput.best_classifier(f"../results/MoFGBML_Basic/{dataset}/trial{rr}{cc}/VAR-0000600000.csv", X_train, y_train)
    
    # resample_transformer = ResampledTransformer(base_model)
    
    # X_resampled, y_resampled = resample_transformer.transform(X_train, y_train)
    
    # trial = list(f"trial{rr}{cc}")
    
    # fname = f"a{trial[5]}_{trial[6]}_{dataset}-10tra.dat"
    
    # output_dir = f"../results/resampled_dataset/MoFGBML_Basic/{dataset}/"
    
    # resample_transformer.output(X_resampled, y_resampled, output_dir, fname)
        