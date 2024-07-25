from src.Tools.tools import convert_dates, plot_prediction_results, selectkbest
from tabulate import tabulate
from src.Pretreatment.ModelTrainer import ModelTrainer
import torch
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV,  cross_validate
import shap
from tpot import TPOTRegressor
import time
from colorama import Fore, Style, init
init(autoreset=True)
import sys
from pathlib import Path
import pandas as pd

Project_Path = Path(__file__).parents[2]
sys.path.append(Project_Path)

import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.Model.ML import regression_models

class Run:
    def __init__(self, models,plot =True):
        self.modeltrainer = ModelTrainer()
        self.models = models
        self.selection_functions = ["f_regression"]
        self.plot = plot
    
    def _process_and_evaluate(self, model_name, x_train, y_train, x_test, y_true, k=21, function_name="f_regression"):#ou k = 20 pour le décalage des t+1
        if model_name == "TPOT":
            # Définir le modèle TPOT
            x_train, x_test = selectkbest(x_train, y_train, x_test, k, function_name)
            model = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42,n_jobs=13)
            start_time = time.time()
            model.fit(x_train, y_train)
            duration = time.time() - start_time
        else :
            # x_train, x_test = selectkbest(x_train, y_train, x_test, k, function_name)
            model_func = self.models[model_name]
            model, duration = model_func(x_train, y_train, self.modeltrainer.model_config[model_name.lower()])
            # self.shap_lib(model,x_train,x_test)
        y_pred = model.predict(x_test)
        mae = round(MAE(y_true, y_pred), 2)
        
        return mae, duration, model, y_pred

    def _prepare_results(self, model_name, mae, duration, y_true, y_pred,y_train, best_k=None):

        sd, ed, sdt, edt = convert_dates(self.modeltrainer)
        result = [model_name, sd, ed, (ed - sd).days, sdt, edt, (edt - sdt).days, mae, duration]
        if best_k is not None:
            result.append(best_k)
        if self.plot:
            plot_prediction_results(y_true, y_pred, model_name, self.modeltrainer.variables_config["target_variable"],y_train)
        return result

    def _print_results(self, results, headers):
        headers_colored = [Fore.CYAN + header + Style.RESET_ALL for header in headers]
        table = tabulate(sorted(results, key=lambda x: x[7]), headers=headers_colored, tablefmt='pretty')
        print(table)

    def run_train_ml(self):
        results = []
        for model_name in self.models:
            x_train, y_train, x_test, y_true = self.modeltrainer.process_data_and_train_model()
            best_mae = float('inf')
            best_k = None
            best_duration = None
            for function_name in self.selection_functions:
                for k in range(12, 25):
                    print(Fore.RED + f"---------------------------------------- \nSearch the best k for {model_name} : {k}" + Style.RESET_ALL)
                    mae, duration, model, y_pred = self._process_and_evaluate(model_name, x_train, y_train, x_test, y_true, k, function_name)
                    if mae < best_mae:
                        best_mae = mae
                        best_k = k
                        best_duration = duration
                        print(Fore.GREEN + f"Find best k for {model_name} : {k}" + Style.RESET_ALL)
            results.append(self._prepare_results(model_name, best_mae, best_duration, y_true, y_pred,y_train, best_k))
        
        self._print_results(results, ['Model', 'Start-Day-Train', 'End-Day-Train', 'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE', "Duration (sec)", "Best K"])

    def run_train_dl(self):
        results = []
        param_dist = {
            "hidden_size": sp_randint(10, 100),
            "num_layers": sp_randint(1, 4),
            "output_size": [1], 
            "num_epochs": sp_randint(1000, 10000),
            "learning_rate": sp_uniform(0.0001, 0.01)
        }

        for model_name in self.models:
            x_train, y_train, x_test, y_true = self.modeltrainer.process_data_and_train_model()
            model_func = self.models[model_name]
            model_params = self.modeltrainer.model_config.get(model_name.lower(), {})
            model = model_func(**model_params)

            search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=20, cv=3, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
            search.fit(x_train, y_train)

            best_model = search.best_estimator_
            best_params = search.best_params_
            y_pred = best_model.predict(x_test)
            mae = round(MAE(y_true, y_pred), 2)
            duration = best_model.duration

            results.append(self._prepare_results(model_name, mae, duration, y_true, y_pred, y_train))
            print(best_params)

        self._print_results(results, ['Model', 'Start-Day-Train', 'End-Day-Train', 'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE', "Duration (sec)"])

    def run_dl(self):
        results = []
        for model_name in self.models:
            x_train, y_train, x_test, y_true = self.modeltrainer.process_data_and_train_model()
            model_func = self.models[model_name]
            model_params = self.modeltrainer.model_config.get(model_name.lower(), {})
            model = model_func(**model_params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mae = round(MAE(y_true, y_pred), 2)
            results.append(self._prepare_results(model_name, mae, model.duration, y_true, y_pred))
        self._print_results(results, ['Model', 'Start-Day-Train', 'End-Day-Train', 'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE', "Duration (sec)"])

    def run_ml(self):
        results = []
        for model_name in self.models:
            x_train, y_train, x_test, y_true = self.modeltrainer.process_data_and_train_model()
            mae, duration, model, y_pred = self._process_and_evaluate(model_name, x_train, y_train, x_test, y_true)                                                                                                
            results.append(self._prepare_results(model_name, mae, duration, y_true, y_pred, y_train))
        self._print_results(results, ['Model', 'Start-Day-Train', 'End-Day-Train', 'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE', "Duration (sec)"])

    def shap_lib(self, model, x_train_k, x_test_k):
        explainer = shap.Explainer(model, x_train_k)
        shap_values = explainer(x_test_k)
        shap.summary_plot(shap_values, x_test_k, feature_names=self.modeltrainer.features, plot_type="bar")

    def build_pipeline(self):

        x_train, y_train, x_test, y_test,features,target, config = self.modeltrainer.process_data_and_train_model()
        # scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), QuantileTransformer()]
        scalers = [StandardScaler()]
        results = []
        for model_name in self.models:
            model, model_space = regression_models[model_name]
            model_space_multi = {f'multioutput__estimator__{key}': value for key, value in model_space.items()}
            for scaler in scalers:
                print(f"Model : {model_name} -> Scaler : {scaler} ")
                
                if len(y_train.shape) == 1: 
                    feature_selector_space = {'feature_selector__k': list(range(10, 30))}
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
                    feature_selector_space = {'multioutput__estimator__feature_selector__k': list(range(1, 30))}
                    pipeline = Pipeline([
                        ('scaler', scaler),
                        ('multioutput', MultiOutputRegressor(Pipeline([
                            ('feature_selector', SelectKBest(f_regression)),
                            ('model', model())
                        ])))
                    ])

                    distributions = {
                        **model_space_multi,
                        **feature_selector_space,
                    }
                
                clf = RandomizedSearchCV(pipeline, distributions, n_iter=10, cv=5, verbose=1, n_jobs=-1)
                start_time = time.time()
                
                search = clf.fit(x_train, y_train_ravel)
                
                # Calculer la durée de l'entraînement
                duration = time.time() - start_time
                predictions = clf.predict(x_test)

               
                # Calculating metrics
                r2 = r2_score(y_test_ravel, predictions)
                mae = mean_absolute_error(y_test_ravel, predictions)
                mse = mean_squared_error(y_test_ravel, predictions)

                # Cross-validation results
                cv_results = cross_validate(search.best_estimator_, x_train, y_train_ravel, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

                # Append results to the list
                results.append({
                    'Model': model_name,
                    'Scaler': scaler.__class__.__name__,
                    'Best Parameters': search.best_params_,
                    'Mean CV Train Score': cv_results['train_score'].mean(),
                    'Mean CV Test Score': cv_results['test_score'].mean(),
                    'R2 Score': r2,
                    'MAE': mae,
                    'MSE': mse,
                    'X_Features': features,
                    'Y_Target': target,
                    'Start_date': config.get('date')['start_date'],
                    'End_date': config.get('date')['end_date'],
                    'Predict_start_date': config.get('date')['predict_start_date'],
                    'Predict_end_date': config.get('date')['predict_end_date'],
                    'Lookback': config.get('variables')['window_lookback(shift)'],
                    'Horizon': config.get('variables')['horizon'],
                    'Mode':config.get('mode'),
                    'Duration (seconds)': duration
                })

                print(f'Scaler: {scaler.__class__.__name__}')
                print('Training scores:', search.cv_results_['mean_test_score'].mean())
                print('Best parameters:', search.best_params_)

        
        # Create a DataFrame and save to CSV or Excel
        results_df = pd.DataFrame(results)
        results_df.to_csv('model_results.csv', index=False)
        results_df.to_excel('model_results.xlsx', index=False)

        print("Results saved to 'model_results.xlsx'.")
                    
            #plot_prediction_results(y_test_ravel, predictions, model_name, self.modeltrainer.variables_config["target_variable"], y_train)
    
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
                # Predictions
                predictions = clf.predict(x_test)

                # Calculating metrics
                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)

                # Cross-validation results
                cv_results = cross_validate(search.best_estimator_, x_train, y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

                # Append results to the list
                results.append({
                    'Model': model_name,
                    'Scaler': scaler.__class__.__name__,
                    'Best Parameters': search.best_params_,
                    'Mean CV Train Score': cv_results['train_score'].mean(),
                    'Mean CV Test Score': cv_results['test_score'].mean(),
                    'R2 Score': r2,
                    'MAE': mae,
                    'MSE': mse,
                    'X_Features': features,
                    'Y_Target': target,
                    'Start_date': config.get('date')['start_date'],
                    'End_date': config.get('date')['end_date'],
                    'Predict_start_date': config.get('date')['predict_start_date'],
                    'Predict_end_date': config.get('date')['predict_end_date'],
                    'Lookback': config.get('variables')['window_lookback(shift)'],
                    'Horizon': config.get('variables')['horizon'],
                    'Mode':config.get('mode'),
                    'Duration (seconds)': duration
                })

                print(f'Scaler: {scaler.__class__.__name__}')
                print('Training scores:', search.cv_results_['mean_test_score'].mean())
                print('Best parameters:', search.best_params_)

        # Create a DataFrame and save to CSV or Excel
        results_df = pd.DataFrame(results)
        results_df.to_excel('model_results_nn.xlsx', index=False)

        print("Results saved to 'model_results_nn.xlsx'.")

    def run_pipe_ml(self):
        self.build_pipeline()
    def run_pipe_dl(self):
        self.build_nn_pipeline()