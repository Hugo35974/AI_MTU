import sys
from pathlib import Path

import pandas as pd

Project_Path = Path(__file__).parents[1]
sys.path.append(Project_Path)

from src.Pretreatment.ConfigLoader import ConfigLoader
from src.Tools.tools import (multi_step, remove_rows_hour_col, shifting,
                             shifting_by_day)


class ModelTrainer(ConfigLoader):
    def __init__(self):
        super().__init__()
        self.features = None
        self.df = None

    def apply_transformations(self, df):
        """
        Apply transformations to the dataframe based on the specified datetime attributes.
        """
        for new_col, datetime_attr in self.transformations.items():
            if datetime_attr == 'weekofyear':
                df[new_col] = df.index.to_series().apply(lambda x: x.isocalendar().week)
            else:
                df[new_col] = getattr(df.index, datetime_attr)
        return df

    def get_features(self, df):
        """
        Extract the relevant features for training from the dataframe.
        """
        features = [col for col in df.columns if 'lag' in col]
        features += list(self.transformations.keys())
        features += self.variables_config["features"]
        features.remove('hour')
        return features

    def get_train_test_split(self, df):
        """
        Split the dataframe into training and testing datasets.
        """
        split_index = int(0.8 * len(df))
        df_train = df.iloc[:split_index]
        df_test = df.iloc[split_index:]
        self.date_s = df_train.index[0].strftime('%Y-%m-%d')
        self.date_end = df_train.index[-1].strftime('%Y-%m-%d')
        self.predict_s = df_test.index[0].strftime('%Y-%m-%d')
        self.predict_e = df_test.index[-1].strftime('%Y-%m-%d')

        print(f"Trainning : from {self.date_s} to {self.date_end}")
        print(f"Test : from {self.predict_s} to {self.predict_e}")
        return df_train, df_test

    def prepare_data(self, data, is_train=True):
        """
        Prepare the training or testing data by selecting the features and the target variable.
        """
        x_data = data[self.features]
        if is_train:
            y_data = data[self.target]
            return x_data, y_data
        return x_data

    def get_final_df(self):
        """
        Launch the data processor and apply transformations to get the final dataframe.
        """
        df_final = self.launchdataprocessor()
        df_final = self.apply_transformations(df_final)
        return df_final
    
    def get_data_from_api(self, df):
        """
        Launch the data processor and apply transformations to get the final dataframe.
        """
        # Créer le DataFrame et définir l'index sur 'applicable_date'
        df = pd.DataFrame(df, columns=['applicable_date', 'elec_prices'])
        df.set_index('applicable_date', inplace=True)
        
        # Convertir l'index en datetime
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
        df = df.sort_index()
        # Appliquer les transformations et retourner le DataFrame final
        df_final = self.apply_transformations(df)
        return df_final

    def process_data_and_train_model(self,df= None,model_infos=None):
        """
        Process the data and train the model, returning the scaled training and testing data.
        """
        # Get the final processed dataframe
        if df:
            df_final = self.get_data_from_api(df)
            self.variables_config["horizon"] = model_infos["horizon"]
        else: 
            df_final = self.get_final_df()

        # Add more columns for machine learning
        df_final['expanding_window_price'] = df_final['elec_prices'].expanding().mean()
        df_final = shifting(self.variables_config["variables_to_shift"], df_final, self.variables_config["window_lookback(shift)"])
        df_final, namelist = multi_step(df_final, self.variables_config["target_variable"],self.variables_config["horizon"])
        
        # Split the datasest in train & test set
        data_train, data_test = self.get_train_test_split(df_final.dropna())
        

        if self.mode == 0 :
            data_train = remove_rows_hour_col(data_train,self.variables_config["hour"])
            data_test = remove_rows_hour_col(data_test,self.variables_config["hour"])

        self.target = [self.variables_config["target_variable"]] + namelist
        self.features = self.get_features(df_final)

        # Prepare the training and testing data
        x_train, y_train = self.prepare_data(data_train, is_train=True)
        x_test = self.prepare_data(data_test, is_train=False)
        y_true = data_test[self.target]

        print(f"x_names {self.features}")
        print(f"y_names {self.target}")

        print(f"Xtrain shape = {x_train.shape}")
        print(f"Xtest shape = {x_test.shape}")
        print(f"Ytrain shape = {y_train.shape}")
        print(f"Ytrue shape = {y_true.shape}")

        return x_train.values, y_train.values, x_test.values, y_true.values, self.features,self.target,self.config
