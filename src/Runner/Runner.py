import time

from colorama import Fore, Style, init
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import (RandomizedSearchCV, cross_val_predict,
                                     cross_validate)
from tabulate import tabulate

from src.Model.CompositeModel import CompositeModel
from src.Pretreatment.ModelTrainer import ModelTrainer
from src.Tools.tools import (convert_dates, evaluate_model,
                             plot_prediction_results, save_best_model,
                             setup_pipeline_multi_output,
                             setup_pipeline_single_output, train_and_search)

init(autoreset=True)
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

Project_Path = Path(__file__).parents[2]
Main_Path = Path(__file__).parents[0]
sys.path.append(Project_Path)
Model_Path = os.path.join(Project_Path,'data','modelsave')
Contener_Path = os.path.join(Project_Path,'contener')

import numpy as np
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   RobustScaler, StandardScaler)

from src.Model.ML import regression_models


class Run:
    def __init__(self, models = None,plot =False):
        self.modeltrainer = ModelTrainer()
        if models :
            self.models = models
        else:
            self.models = self.modeltrainer.models
        self.plot = plot
        self.model_path = Model_Path

    def _prepare_results(self, model_name, mae, y_true, y_pred,y_train):

        sd, ed, sdt, edt = convert_dates(self.modeltrainer)
        result = [model_name, sd, ed, (ed - sd).days, sdt, edt, (edt - sdt).days, mae]
        return result

    def _print_results(self, results, headers):
        headers_colored = [Fore.CYAN + header + Style.RESET_ALL for header in headers]
        table = tabulate(sorted(results, key=lambda x: x[7]), headers=headers_colored, tablefmt='pretty')
        print(table)

    def run(self):
        results = []
        for model_name in self.models:
            model = os.path.join(self.model_path,f'{model_name}.pkl')
            with open(model, 'rb') as model_file:
                loaded_model = pickle.load(model_file)
                x_train, y_train, x_test, y_true, self.features,self.target,self.config= self.modeltrainer.process_data_and_train_model()
                y_pred = loaded_model.predict(x_test)   
                # Calculate the MAE
                mae = MAE(y_true, y_pred)
                results.append(self._prepare_results(model_name, mae, y_true, y_pred, y_train))
            self._print_results(results, ['Model', 'Start-Day-Train', 'End-Day-Train', 'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE'])
        return results  
    
    def build_pipeline(self, df = None, model_infos = None):
        # Préparation des données et des modèles
        x_train, y_train, x_test, y_test, features, target, config = self.modeltrainer.process_data_and_train_model(df,model_infos)
        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), QuantileTransformer()]
        results = []

        for model_name in self.models:
            model, model_space = regression_models[model_name]
            model_space_multi = {f'multioutput__estimator__{key}': value for key, value in model_space.items()}

            for scaler in scalers:
                print(f"Model : {model_name} -> Scaler : {scaler} ")

                if len(y_train.shape) == 1:
                    pipeline, distributions = setup_pipeline_single_output(
                        scaler, model, model_space, x_train
                    )
                    y_train = np.ravel(y_train)
                    y_test = np.ravel(y_test)
                else:
                    pipeline, distributions = setup_pipeline_multi_output(
                        scaler, model, model_space_multi, x_train
                    )

                # Recherche des meilleurs hyperparamètres et entraînement
                search, duration = train_and_search(pipeline, distributions, x_train, y_train)

                # Évaluation du modèle
                result = evaluate_model(
                    search.best_estimator_, x_train, y_train, x_test, y_test, model_name, config, duration,scaler.__class__.__name__
                )
                results.append(result)

                # Sauvegarde du modèle
                save_best_model(search.best_estimator_, model_name, scaler,Model_Path)
        results_df = pd.DataFrame(results)
        return self.create_composite_model(x_train,y_train,x_test,y_test,results_df,model_infos,config)


    def build_nn_pipeline(self):
        x_train, y_train, x_test, y_test,features,target, config = self.modeltrainer.process_data_and_train_model()
        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), QuantileTransformer()]
        results = []

        for model_name in self.models:  # `nn_models` est une liste ou un dictionnaire contenant les modèles de réseaux de neurones
            model, model_space = regression_models[model_name]  # Assurez-vous que `regression_models` contient les configurations des modèles de NN
            for scaler in scalers:
                print(f"Model : {model_name} -> Scaler : {scaler} ")
                pipeline = Pipeline([
                    ('scaler', scaler),
                    ('model', model())
                ])
                distributions = model_space

                clf = RandomizedSearchCV(pipeline, distributions, n_iter=10, cv=5, verbose=1, n_jobs=-1)
                start_time = time.time()
                search = clf.fit(x_train, y_train)
                duration = time.time() - start_time

                cv_results = cross_validate(search.best_estimator_, x_train, y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                y_pred = cross_val_predict(search.best_estimator_, x_train, y_train, cv=5)

                r2_per_dimension = [r2_score(y_train[:, i], y_pred[:, i]) for i in range(y_train.shape[1])]
                mae_per_dimension = np.mean(np.abs(y_train - y_pred), axis=0)
                spearman_per_dimension = [spearmanr(y_train[:, i], y_pred[:, i])[0] for i in range(y_train.shape[1])]
                MAE = np.mean(mae_per_dimension)
                R2 = np.mean(r2_per_dimension)
                SRC = np.mean(spearman_per_dimension)
                # Append results to the list
                result = {
                    'Model':  model_name + '_' + scaler.__class__.__name__,
                    'Best Parameters': search.best_params_,
                    'Mean CV Train Score': cv_results['train_score'].mean(),
                    'Mean CV Test Score': cv_results['test_score'].mean(),
                    'R2 Score': R2,
                    'MAE': MAE,
                    'SRC_Mean': SRC,
                    'X_Features': features,
                    'Y_Target': target,
                    'Start_date': config.get('date')['start_date'],
                    'End_date': config.get('date')['end_date'],
                    'Lookback': config.get('variables')['window_lookback(shift)'],
                    'Horizon': config.get('variables')['horizon'],
                    'Mode':config.get('mode'),
                    'Duration (seconds)': duration
                }
                for i in range(len(mae_per_dimension)):
                    result[f'MAE_T_{i}'] = mae_per_dimension[i]

                for i in range(len(r2_per_dimension)):
                    result[f'R2_T_{i}'] = r2_per_dimension[i]

                for i in range(len(spearman_per_dimension)):
                    result[f'SRC_T_{i}'] = spearman_per_dimension[i]
                    
                results.append(result)
                print(f'Scaler: {scaler.__class__.__name__}')
                print('Training scores:', search.cv_results_['mean_test_score'].mean())
                print('Best parameters:', search.best_params_)

        # Create a DataFrame and save to CSV or Excel
        results_df = pd.DataFrame(results)
        results_df.to_excel('model_results_nn.xlsx', index=False)

        print("Results saved to 'model_results_nn.xlsx'.")

    def create_composite_model(self,x_train, y_train, x_test, y_test,results_df,model_infos = None,config=None):
        # Charger les résultats des modèles
        column_name = model_infos["column_name"]
        # Sélection des meilleurs modèles par output
        best_models_per_output = {}
        for i in range(y_train.shape[1]):
            results_df[f"MAE-SRC_T_{i}"] = results_df[f'MAE_T_{i}'] - results_df[f'SRC_T_{i}']
            best_model_info = results_df.loc[results_df[f"MAE-SRC_T_{i}"].idxmin()]
            best_models_per_output[f'Output_{i}'] = {
                'Model': best_model_info['Model'],
                'R2': best_model_info[f'R2_T_{i}'],
                'MAE': best_model_info[f'MAE_T_{i}'],
                'SRC': best_model_info[f'SRC_T_{i}'],
                'MSE': best_model_info[f'MSE_T_{i}'],
                'RMSE': best_model_info[f'RMSE_T_{i}'],
                'MAPE': best_model_info[f'MAPE_T_{i}'],
            }

        best_models_df = pd.DataFrame(best_models_per_output).T
        best_models_df.to_csv(f'data/results/result_CrossVal_{column_name}.csv', index=True)

        # Création d'une liste pour stocker les modèles
        individual_models = []
        for i in range(y_train.shape[1]):
            model_filename = os.path.join(Model_Path, best_models_per_output[f'Output_{i}']['Model'] + '.pkl')
            with open(model_filename, 'rb') as file:
                model = pickle.load(file)
                individual_models.append(model)

        # Création et entraînement du modèle composite
        composite_model = CompositeModel(individual_models)
        composite_model.fit(x_train, y_train)

        if self.plot:
            y_pred = composite_model.predict(x_test)
            plot_prediction_results(y_test, y_pred, "composite_model", self.modeltrainer.variables_config["target_variable"],y_train)
        duration = 1
        result = evaluate_model(
                    composite_model, x_train, y_train,x_test, y_test,model_infos["column_name"], config, duration
                )
        model_composite_path = os.path.join(Contener_Path,model_infos["path"])

        result = pd.DataFrame([result])

        with open(model_composite_path, 'wb') as composite_file:
            pickle.dump(composite_model, composite_file)

        result.to_csv(f'data/results/result_Test_{column_name}.csv', index=True)

        print(f"Composite model saved to {model_composite_path}.")
        return result