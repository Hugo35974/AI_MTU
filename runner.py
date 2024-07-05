
from Model.XGB import XGBM_model
from Model.LGB import LGBM_model
from Model.RFR import RFRM_model
from Tools.tools import eval_metrics,convert_dates
from tabulate import tabulate
from Model.ModelTrainer import ModelTrainer

def run(modeltrainer, models):
        results = []
        for model_name in models:
                x_train, y_train, x_test, y_true = modeltrainer.process_data_and_train_model()

                if model_name not in models:
                        raise ValueError(f"Model '{model_name}' non supported")

                model_func = models[model_name]
                model, duration = model_func(x_train, y_train, modeltrainer.model_config[model_name.lower()])

                y_pred = model.predict(x_test)
                sd, ed, sdt, edt = convert_dates(modeltrainer.date_config)
                results.append([model_name, sd, ed,(ed - sd).days,sdt, edt,(edt - sdt).days,\
                                round(eval_metrics(y_true, y_pred),2), duration])

        print(tabulate(sorted(results, key=lambda x: x[7]), headers=['Model','Start-Day-Train','End-Day-Train',\
                'Train-Period','Start-Day-Test','End-Day-Test','Test-Period (days)','MAE',"Duration (days)"],tablefmt='pretty'))
                
if __name__ == "__main__":
    modeltrainer = ModelTrainer()
    ai_functions = {
        'XGB': XGBM_model,
        'LGB': LGBM_model,
        'RFR': RFRM_model
    }
    run(modeltrainer,ai_functions)