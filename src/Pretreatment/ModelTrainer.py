from sklearn.preprocessing import MinMaxScaler
import os
import sys
from pathlib import Path

Project_Path = Path(__file__).parents[2]
sys.path.append(Project_Path)

from src.Tools.tools import shifting
from src.Pretreatment.ConfigLoader import ConfigLoader

class ModelTrainer(ConfigLoader):

    def __init__(self):
        super().__init__()
        self.features = None

    def apply_transformations(self, df):
        """
        Apply transformations to the dataframe based on the specified datetime attributes.
        """
        for new_col, datetime_attr in self.transformations.items():
            df[new_col] = getattr(df.index, datetime_attr)
        return df
    
    def get_features(self, df):
        """
        Extract the relevant features for training from the dataframe.
        """
        features = [col for col in df.columns if 'lag' in col]
        features += [self.transformations.get(attr, attr) for attr in ['hour', 'day_of_week', 'day_of_year']]
        return features   
    
    # def get_train_test_split(self, df):
    #     """
    #     Split the dataframe into training and testing datasets based on the configured dates.
    #     """
    #     data_train = df.loc[self.date_config["start_date"]:self.date_config["end_date"]]
    #     data_test = df.loc[self.date_config["predict_start_date"]:self.date_config["predict_end_date"]]
    #     self.date_s = data_train.index[0].date().strftime('%Y-%m-%d')
    #     self.date_end = data_train.index[-1].date().strftime('%Y-%m-%d')
    #     self.predict_s = data_test.index[0].date().strftime('%Y-%m-%d')
    #     self.predict_e = data_test.index[-1].date().strftime('%Y-%m-%d')
    #     return data_train, data_test
    def get_train_test_split(self,df):
        # Filtrer le DataFrame en fonction de la plage de dates spécifiée
        df_filtered = df.loc[self.date_config["start_date"]:self.date_config["end_date"]]
        
        # Calculer l'index pour séparer les données en 80% et 20%
        split_index = int(len(df_filtered) * 0.8)
        
        # Diviser le DataFrame en ensembles d'entraînement et de test
        df_train = df_filtered.iloc[:split_index]
        df_test = df_filtered.iloc[split_index:]
        self.date_s = df_train.index[0].date().strftime('%Y-%m-%d')
        self.date_end = df_train.index[-1].date().strftime('%Y-%m-%d')
        self.predict_s = df_test.index[0].date().strftime('%Y-%m-%d')
        self.predict_e = df_test.index[-1].date().strftime('%Y-%m-%d')
        return df_train, df_test
    def prepare_data(self, data, is_train=True):
        """
        Prepare the training or testing data by selecting the features and the target variable.
        """
        x_data = data[self.features]
        if is_train:
            y_data = data[self.variables_config["target_variable"]]
            return x_data, y_data
        return x_data
    
    def get_final_df(self):
        """
        Launch the data processor and apply transformations to get the final dataframe.
        """
        df_final = self.launchdataprocessor()
        df_final = self.apply_transformations(df_final)
        return df_final
    
    def process_data_and_train_model(self):
        """
        Process the data and train the model, returning the scaled training and testing data.
        """
        df_final = self.get_final_df()
        df_final = shifting(self.variables_config["variables_to_shift"], df_final, self.variables_config["time_to_shift"])
        self.features = self.get_features(df_final)
        data_train, data_test = self.get_train_test_split(df_final)
        x_train, y_train = self.prepare_data(data_train, is_train=True)
        x_test = self.prepare_data(data_test, is_train=False)
        y_true = data_test[self.variables_config["target_variable"]]

        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, y_train, x_test_scaled, y_true
