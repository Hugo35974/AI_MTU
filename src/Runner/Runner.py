from src.Tools.tools import convert_dates, plot_prediction_results
from tabulate import tabulate
from src.Pretreatment.ModelTrainer import ModelTrainer
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import RandomizedSearchCV,  cross_validate,cross_val_predict
import time
from colorama import Fore, Style, init
init(autoreset=True)
import sys
from pathlib import Path
import pandas as pd
import pickle
import os
from joblib import Memory

from scipy.stats import spearmanr
Project_Path = Path(__file__).parents[2]
Main_Path = Path(__file__).parents[0]
sys.path.append(Project_Path)
Model_Path = os.path.join(Project_Path,'data','modelsave')

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.Model.ML import regression_models
from sklearn.ensemble import VotingRegressor,StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
class Run:
    def __init__(self, models,plot =True):
        self.modeltrainer = ModelTrainer()
        self.models = models
        self.plot = plot
    
    def _prepare_results(self, model_name, mae, y_true, y_pred,y_train):

        sd, ed, sdt, edt = convert_dates(self.modeltrainer)
        result = [model_name, sd, ed, (ed - sd).days, sdt, edt, (edt - sdt).days, mae]
        if self.plot:
            plot_prediction_results(y_true, y_pred, model_name, self.modeltrainer.variables_config["target_variable"],y_train)
        return result

    def _print_results(self, results, headers):
        headers_colored = [Fore.CYAN + header + Style.RESET_ALL for header in headers]
        table = tabulate(sorted(results, key=lambda x: x[7]), headers=headers_colored, tablefmt='pretty')
        print(table)

    def run(self):
        results = []
        for model_name in self.models:
            model = os.path.join(Model_Path,f'{model_name}.pkl')
            with open(model, 'rb') as model_file:
                loaded_model = pickle.load(model_file)
            for model_name in self.models:
                x_train, y_train, x_test, y_true, self.features,self.target,self.config= self.modeltrainer.process_data_and_train_model()
                y_pred = loaded_model.predict(x_test)   
                # Calculate the MAE
                mae = MAE(y_true, y_pred)

                results.append(self._prepare_results(model_name, mae, y_true, y_pred, y_train))
            self._print_results(results, ['Model', 'Start-Day-Train', 'End-Day-Train', 'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE'])

    def build_pipeline(self):

        x_train, y_train, x_test, y_test,features,target, config = self.modeltrainer.process_data_and_train_model()
        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), QuantileTransformer()]
        results = []
        cachedir = os.path.join(Model_Path,'cachedir')
        memory = Memory(location=cachedir, verbose=0)
        for model_name in self.models:
            model, model_space = regression_models[model_name]
            model_space_multi = {f'multioutput__estimator__{key}': value for key, value in model_space.items()}
            for scaler in scalers:
                print(f"Model : {model_name} -> Scaler : {scaler} ")
                
                if len(y_train.shape) == 1: 
                    feature_selector_space = {'feature_selector__k': list(range(0,x_train.shape[1]))}
                    y_train_ravel = np.ravel(y_train)
                    y_test_ravel = np.ravel(y_test)
                    pipeline = Pipeline([
                        ('scaler', scaler),
                        ('feature_selector', SelectKBest(f_regression)),
                        ('model', model()),
                    ])
                    distributions = {
                        **model_space,
                        **feature_selector_space,
                    }
                
                else:
                    y_train_ravel = y_train
                    y_test_ravel = y_test
                    feature_selector_space = {'multioutput__estimator__feature_selector__k': list(range(0,x_train.shape[1]))}
                    pipeline = Pipeline([
                        ('scaler', scaler),
                        ('multioutput', MultiOutputRegressor(Pipeline([
                            ('feature_selector', SelectKBest(f_regression)),
                            ('model', model())
                        ]),n_jobs=12))
                    ],memory=memory)

                    distributions = {
                        **model_space_multi,
                        **feature_selector_space,
                    }
                
                clf = RandomizedSearchCV(pipeline, distributions, n_iter=10, cv=5, verbose=1, n_jobs=-1)
                start_time = time.time()
                
                search = clf.fit(x_train, y_train_ravel)
                
                # Calculer la durée de l'entraînement
                duration = time.time() - start_time

                # Cross-validation results
                cv_results = cross_validate(search.best_estimator_, x_train, y_train_ravel, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                y_pred = cross_val_predict(search.best_estimator_, x_train, y_train_ravel, cv=5)
                spearman_per_dimension = [spearmanr(y_train_ravel[:, i], y_pred[:, i])[0] for i in range(y_train_ravel.shape[1])]

                r2_per_dimension = [r2_score(y_train_ravel[:, i], y_pred[:, i]) for i in range(y_train_ravel.shape[1])]
                mae_per_dimension = np.mean(np.abs(y_train_ravel - y_pred), axis=0)
                MAE = np.mean(mae_per_dimension)
                R2 = np.mean(r2_per_dimension)
                SRC = np.mean(spearman_per_dimension)
                # Append results to the list
                result = {
                    'Model':  model_name + '_' + scaler.__class__.__name__,
                    'Best Parameters': search.best_params_,
                    'Mean CV Train Score': cv_results['train_score'].mean(),
                    'Mean CV Test Score': cv_results['test_score'].mean(),
                    'R2_Mean': R2,
                    'MAE_Mean': MAE,
                    'SRC_Mean': SRC,
                    'X_Features': features,
                    'Y_Target': target,
                    'Start_date': config.get('date')['start_date'],
                    'End_date': config.get('date')['end_date'],
                    'Lookback': config.get('variables')['window_lookback(shift)'],
                    'Horizon': config.get('variables')['horizon'],
                    'Mode': config.get('mode'),
                    'Duration (seconds)': duration
                }

                # Add MAE for each dimension as separate columns
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

                best_model = search.best_estimator_
                file = os.path.join(Model_Path,f'{model_name}_{scaler.__class__.__name__}.pkl')

                with open(file, 'wb') as model_file:
                    pickle.dump(best_model, model_file)

        results_df = pd.DataFrame(results)
        results_df.to_csv('model_results.csv', index=False)
        results_df.to_excel('model_results.xlsx', index=False)

        print("Results saved to 'model_results.xlsx'.")
     
          
    
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

def create_composite_model(Model_Path, x_train, y_train):

    results_df = pd.read_csv('model_results.csv')

    # Sélection des meilleurs modèles par output
    best_models_per_output = {}
    for i in range(y_train.shape[1]):
        best_model_info = results_df.loc[results_df[f'R2_T_{i}'].idxmax()]
        best_models_per_output[f'Output_{i}'] = {
            'Model': best_model_info['Model'],
            'Best Parameters': best_model_info['Best Parameters'],
            'R2': best_model_info[f'R2_T_{i}'],
            'MAE': best_model_info[f'MAE_T_{i}'],
            'SRC': best_model_info[f'SRC_T_{i}']
        }

    # Sauvegarde des meilleurs modèles par output
    best_models_df = pd.DataFrame(best_models_per_output).T
    best_models_df.to_csv('best_models_per_output.csv', index=False)
    best_models_df.to_excel('best_models_per_output.xlsx', index=False)

    print("Best models per output saved to 'best_models_per_output.xlsx'.")

    # Création d'un modèle composite à partir des meilleurs modèles par output
    composite_model = MultiOutputRegressor(estimators=[
        (f'output_{i}', pickle.load(open(os.path.join(Model_Path, best_models_per_output[f'Output_{i}']['Model'] + '.pkl'), 'rb')))
        for i in range(y_train.shape[1])
    ])

    # Entraînement du modèle composite
    composite_model.fit(x_train, y_train)

    # Sauvegarde du modèle composite
    composite_model_file = os.path.join(Model_Path, 'composite_model.pkl')
    with open(composite_model_file, 'wb') as model_file:
        pickle.dump(composite_model, model_file)

    print("Composite model saved to 'composite_model.pkl'.")