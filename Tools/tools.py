from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd 
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

Project_Path = Path(__file__).parents[1]

def eval_metrics(y_true,y_pred):
    MAE = mean_absolute_error(y_true,y_pred)
    return MAE

def convert_dates(date_config):
    start_date = pd.to_datetime(date_config["start_date"]).date()
    end_date = pd.to_datetime(date_config["end_date"]).date()
    predict_start_date = pd.to_datetime(date_config["predict_start_date"]).date()
    predict_end_date = pd.to_datetime(date_config["predict_end_date"]).date()
    return start_date, end_date, predict_start_date, predict_end_date

def shifting(list_toshift : list,df :pd.DataFrame, time_to_shift):
    a,b = time_to_shift
    for var in list_toshift:
        for i in range(a,b):  
            df[f'{var}_lag_{i}h'] = df[var].shift(i)
    return df.dropna()

def selectkbest(x_train,y_train,x_test,k_best):  
    selector = SelectKBest(score_func=f_regression, k=k_best)
    x_train = selector.fit_transform(x_train, y_train)
    x_test = selector.transform(x_test)
    return x_train,x_test,selector

def show_result(selector,x_train):

    feature_scores = selector.scores_
    feature_pvalues = selector.pvalues_
    feature_names = x_train.columns
    scores_df = pd.DataFrame({'Feature': feature_names, 'Score': feature_scores, 'p-Value': feature_pvalues})
    scores_df = scores_df.sort_values(by='Score', ascending=False)
    print(scores_df)
    return scores_df

def export_result(scores_df,metrics):

    metrics_df = pd.DataFrame([metrics])
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f'{Project_Path}results/results_{timestamp}.csv'
    os.makedirs('results', exist_ok=True)
    full_results_df = pd.concat([scores_df, metrics_df], axis=1)
    full_results_df.to_csv(filename, index=False)

    print(f"Export results to CSV in : {filename}")


def plot_prediction_results(y_true, y_pred,model_name,target,start_date, end_date):
    prediction_results = pd.DataFrame(index=y_true.index)
    prediction_results[f'{target} réel'] = y_true.values
    prediction_results[f'{target} prédit'] = y_pred

    if not isinstance(prediction_results.index, pd.DatetimeIndex):
        prediction_results.index = pd.to_datetime(prediction_results.index)

    prediction_results = prediction_results.loc[start_date:end_date]

    prediction_results.plot()
    plt.legend()
    plt.ylabel('euro(€)')
    plt.xlabel('Days')
    plt.title(f'Prédiction of {target} with {model_name} model')
    plt.show()