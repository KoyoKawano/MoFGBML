# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:06:50 2022

@author: kawano
"""
from joblib import Parallel, delayed
from CIlab_function import CIlab
from ThresholdOptimization import ThresholdEstimator, predict_proba_transformer
import file_output as output
from ThresholdBaseRejection import SecondStageRejectOption

class runner():
    
    def __init__(self,
                 dataset,
                 algorithmID,
                 experimentID,
                 fname_train,
                 fname_test):
        
        """
        コンストラクタ
        
        Parameter
        ---------
        MoFGBMLライブラリの仕様に合わせています．
        dataset : dataset name : string, ex. "iris"
        
        algorithmID : string
                      "result"直下のディレクトリ名 
        
        experimentID : string
                       出力ファイルのデイレクトリ名, 
                       出力ファイルは "result\\algorithmID\\experimentID に出力されます．
        
        file_train : string 
                     学習用データのファイル名
        
        file_test : string
                    評価用データのファイル名
        """
    
        self.dataset = dataset
        self.algorithmID = algorithmID
        self.experimentID = experimentID
        self.X_train, self.X_test, self.y_train, self.y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
        self.output_dir = f"../results/threshold_base/{self.algorithmID}/{self.dataset}/{self.experimentID}/"
    
    
    def run_second_stage(self, pipe, params, second_models, train_file, test_file, core = 5):
        
        """
        run function
        2段階棄却オプションのARCs(Accuracy-Rejection Curves)で必要なデータを出力する関数.
        
        Parameter
        ---------
        pipe : Pipeline module
               ステップ：predict_proba_transfomer，ThresholdBaseRejectOption
        
        params : パラメータ辞書のリスト, 
                 辞書のキーは，"kmax", "Rmax", "deltaT"にしてください．
        
        second_model : sklearn.ClassifierMixin
                       sklearnの識別器で使用される関数を実装したモデル
                       2段階目の判定で用いるモデル．
        
        train_file : file name of result for trainning data, result is accuracy, reject rate, threshold
        
        test_file : file name of result for test data
        """
        
        def _run_one_search_threshold(param):
            
            return ThresholdEstimator(pipe, param).fit(self.X_train, self.y_train)
 
        
        # 閾値の探索,ここだけ並列処理をしています．
        result_list = Parallel(n_jobs = core)(delayed(_run_one_search_threshold)(param) for param in params)
        
        # 学習用データの結果をまとめて出力
        train_result = [[result.accuracy, result.rejectrate, result.threshold] for result in result_list]
        
        output.to_csv(train_result, self.output_dir, train_file)
        
        proba_test = predict_proba_transformer(result_list[0].pipe[0].model).transform(self.X_test)
        base_predict_test = result_list[0].pipe[-1].predict(proba_test)
        isReject_list = [result.proba_isReject(proba_test) for result in result_list]
  
        # 評価用データの結果をまとめて出力
        test_result = [result_list[0].proba_score(self.y_test, base_predict_test, isReject) for isReject in isReject_list]
           
        output.to_csv(test_result, self.output_dir, test_file)
        
        proba_train = result_list[0].pipe[0].predict_proba
        base_predict_train = result_list[0].pipe[-1].predict(proba_train)
        
        # 2段階棄却オプション，やってることは上と同じ
        for key, model in second_models.items():
            
            model_predict = model.predict(self.X_train)
            
            RO_list = [SecondStageRejectOption(thresh_estimator, model) for thresh_estimator in result_list]
            
            isReject_list = [RO.isReject(proba_train, model_predict) for RO in RO_list]

            second_RO_train_result = [[RO.accuracy(self.y_train, base_predict_train, isReject),
                                       RO.rejectrate(isReject)] for RO, isReject in zip(RO_list, isReject_list)]
            
            output.to_csv(second_RO_train_result, f"{self.output_dir}/{key}/", "second-" + train_file)
            
            model_predict = model.predict(self.X_test)
            
            
            isReject_list = [RO.isReject(proba_test, model_predict) for RO in RO_list]

            second_RO_test_result = [[RO.accuracy(self.y_test, base_predict_test, isReject),
                                       RO.rejectrate(isReject)] for RO, isReject in zip(RO_list, isReject_list)]
          
            output.to_csv(second_RO_test_result, f"{self.output_dir}/{key}/", "second-" + test_file)
            
        
    def output_const(self, dict_):
        
        CIlab.output_dict(dict_, self.output_dir, "Const.txt")
    