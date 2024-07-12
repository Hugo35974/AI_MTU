from src.Tools.tools import convert_dates, plot_prediction_results,selectkbest
from tabulate import tabulate
from src.Pretreatment.ModelTrainer import ModelTrainer
import torch
from sklearn.metrics import mean_absolute_error as MAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Run:
    def __init__(self, models):
        self.modeltrainer = ModelTrainer()
        self.models = models
        self.selection_functions = ["f_regression","pearson_corr"]

    def run_train(self):
        results = []
        for model_name in self.models:
            x_train, y_train, x_test, y_true = self.modeltrainer.process_data_and_train_model()

            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' non supported")
            
            model_func = self.models[model_name]
            best_k = None
            best_mae = float('inf')
            best_function = None
            
            for function_name in self.selection_functions:
                for k in range(12, 25):
                    print(f"Search best param for {model_name} with k = {k} using {function_name}")
                    x_train_k, x_test_k = selectkbest(x_train, y_train, x_test, k, function_name)

                    model, duration = model_func(x_train_k, y_train, self.modeltrainer.model_config[model_name.lower()])
                    y_pred = model.predict(x_test_k)
                    mae = round(MAE(y_true, y_pred), 2)
                                    
                    if mae < best_mae:
                        best_mae = mae
                        best_k = k
                        best_function = function_name
                        print(f"Find best param for {model_name} with k = {k} using {function_name} for MAE : {best_mae}")
            
            sd, ed, sdt, edt = convert_dates(self.modeltrainer.date_config)
            results.append([model_name, sd, ed, (ed - sd).days, sdt, edt, (edt - sdt).days, \
                            best_mae, best_k, best_function, duration])
        
        print(tabulate(sorted(results, key=lambda x: x[7]), headers=['Model', 'Start-Day-Train', 'End-Day-Train',
                'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE', "K-Best",\
                    "Function-K-Best", "Duration (sec)"], tablefmt='pretty'))
        
        print(f"Best k: {best_k}, Best MAE: {best_mae}, Best Function: {best_function}")

    def run(self):
        results = []
        for model_name in self.models:
            x_train, y_train, x_test, y_true = self.modeltrainer.process_data_and_train_model()

            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' non supported")
            
            model_func = self.models[model_name]
            
            x_train_k, x_test_k = selectkbest(x_train, y_train, x_test, 21, "f_regression")
            model, duration = model_func(x_train_k, y_train, self.modeltrainer.model_config[model_name.lower()])
            y_pred = model.predict(x_test_k)
            mae = round(MAE(y_true, y_pred), 2)
                                    
            sd, ed, sdt, edt = convert_dates(self.modeltrainer.date_config)
            results.append([model_name, sd, ed, (ed - sd).days, sdt, edt, (edt - sdt).days, \
                            mae, duration])
            
            plot_prediction_results(y_true, y_pred, model_name, self.modeltrainer.variables_config["target_variable"],
                                    self.modeltrainer.date_config["predict_start_date"], self.modeltrainer.date_config["predict_end_date"],mae)

        print(tabulate(sorted(results, key=lambda x: x[7]), headers=['Model', 'Start-Day-Train', 'End-Day-Train',
                'Train-Period', 'Start-Day-Test', 'End-Day-Test', 'Test-Period (days)', 'MAE', "Duration (sec)"], tablefmt='pretty'))