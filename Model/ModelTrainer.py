import pandas as pd 
from Model import XGB
from Pretreatment.DataProcessor import DataProcessor
from Tools.tools import shifting,plot_prediction_results,eval_metrics
from pathlib import Path
import os
import json
from Pretreatment.ConfigLoader import ConfigLoader

class ModelTrainer(ConfigLoader):

    def applytransformations(self, df):
        for new_col, datetime_attr in self.transformations.items():
            df[new_col] = getattr(df.index, datetime_attr)
        return df
    
    def getfeatures(self, df):
        features = [col for col in df.columns if 'lag' in col]
        features += [self.transformations.get(attr, attr) for attr in ['hour', 'day_of_week', 'day_of_year']]
        return features   
    
    def gettraintestsplit(self, df):
        data_train = df.loc[self.date_config["start_date"]:self.date_config["end_date"]]
        data_test = df.loc[self.date_config["predict_start_date"]:self.date_config["predict_end_date"]]
        return data_train, data_test

    def preparetraindata(self, data_train):
        features = self.getfeatures(data_train)
        x_train = data_train[features]
        y_train = data_train[self.variables_config["target_variable"]]
        return x_train, y_train

    def preparetestdata(self, data_test):
        features = self.getfeatures(data_test)
        x_test = data_test[features]
        return x_test
    
    def process_data_and_train_model(self):

        df_final = self.launchdataprocessor()
        df_final = self.applytransformations(df_final)
        df_final = shifting(self.variables_config["variables_to_shift"], df_final, self.variables_config["time_to_shift"])

        data_train, data_test = self.gettraintestsplit(df_final)
        x_train, y_train = self.preparetraindata(data_train)
        x_test = self.preparetestdata(data_test)
        y_true = data_test[self.variables_config["target_variable"]]

        return x_train, y_train, x_test, y_true   
    