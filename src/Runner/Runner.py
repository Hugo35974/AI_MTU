from src.Tools.tools import convert_dates, plot_prediction_results, selectkbest
from tabulate import tabulate
from src.Pretreatment.ModelTrainer import ModelTrainer
import torch
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV
import shap
from tpot import TPOTRegressor
from sklearn.metrics import mean_absolute_error as MAE
import time
from colorama import Fore, Style, init
init(autoreset=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Run:
    def __init__(self, models,plot =True):
        self.modeltrainer = ModelTrainer()
        self.models = models
        self.selection_functions = ["f_regression"]
        self.plot = plot
    
    def _process_and_evaluate(self, model_name, x_train, y_train, x_test, y_true, k=21, function_name="f_regression"):
        if model_name == "TPOT":
            # Définir le modèle TPOT
            x_train, x_test = selectkbest(x_train, y_train, x_test, k, function_name)
            model = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42,n_jobs=13)
            start_time = time.time()
            model.fit(x_train, y_train)
            duration = time.time() - start_time
        else :
            x_train, x_test = selectkbest(x_train, y_train, x_test, k, function_name)
            model_func = self.models[model_name]
            model, duration = model_func(x_train, y_train, self.modeltrainer.model_config[model_name.lower()])
            # self.shap_lib(model,x_train,x_test)
        y_pred = model.predict(x_test)
        mae = round(MAE(y_true, y_pred), 2)
        
        return mae, duration, model, y_pred

    def _prepare_results(self, model_name, mae, duration, y_true, y_pred, best_k=None):

        sd, ed, sdt, edt = convert_dates(self.modeltrainer)
        result = [model_name, sd, ed, (ed - sd).days, sdt, edt, (edt - sdt).days, mae, duration]
        if best_k is not None:
            result.append(best_k)
        if self.plot:
            plot_prediction_results(y_true, y_pred, model_name, self.modeltrainer.variables_config["target_variable"],
                                sdt, edt, mae)
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
            results.append(self._prepare_results(model_name, best_mae, best_duration, y_true, y_pred, best_k))
        
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

            results.append(self._prepare_results(model_name, mae, duration, y_true, y_pred))
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
            results.append(self._prepare_results(model_name, mae, duration, y_true, y_pred))
        self._print_results(results, ['Model', 'Start-Day-Train', 'End-Day-Train', 'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE', "Duration (sec)"])

    def shap_lib(self, model, x_train_k, x_test_k):
        explainer = shap.Explainer(model, x_train_k)
        shap_values = explainer(x_test_k)
        shap.summary_plot(shap_values, x_test_k, feature_names=self.modeltrainer.features, plot_type="bar")
