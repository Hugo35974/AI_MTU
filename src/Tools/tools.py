from sklearn.metrics import mean_absolute_error
import pandas as pd 
import os
import seaborn as sns
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression

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
def feature_importance(X, y):

    model = RandomForestRegressor()
    model.fit(X, y)

    return model.feature_importances_


def pearson_corr(X, y):
    corr_matrix = np.corrcoef(X, y, rowvar=False)
    corr_with_target = np.abs(corr_matrix[:-1, -1])  
    return corr_with_target

# def shifting(list_toshift: list, df: pd.DataFrame, time_to_shift):
#     a, b = time_to_shift
#     new_columns = {}
    
#     for var in list_toshift:
#         for i in range(a, b):
#             new_column_name = f'{var}_lag_{i}h'
#             new_columns[new_column_name] = df[var].shift(i)
    
#     df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
#     df = df.dropna()
    
#     return df

def shifting(list_toshift: list, df: pd.DataFrame, time_to_shift):
    start, step = time_to_shift
    new_columns = {}
    
    for var in list_toshift:
        for i in range(1, step + 1):
            shift_hours = start * i
            new_column_name = f'{var}_lag_{shift_hours}h'
            new_columns[new_column_name] = df[var].shift(shift_hours)
    
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    df = df.dropna()
    
    return df

selection_functions = {
    "f_regression": f_regression,
    "pearson_corr": lambda X, y: (np.abs(np.corrcoef(X.T, y)[-1, :-1]), None),
}


def selectkbest(x_train, y_train, x_test, k, function_name):
    selector = SelectKBest(score_func=selection_functions[function_name], k=k)
    x_train_k = selector.fit_transform(x_train, y_train)
    x_test_k = selector.transform(x_test)
    return x_train_k, x_test_k

def show_result(selector,feature_names):

    feature_scores = selector.scores_
    feature_pvalues = selector.pvalues_
    scores_df = pd.DataFrame({'Feature': feature_names, 'Score': feature_scores, 'p-Value': feature_pvalues})
    scores_df = scores_df.sort_values(by='Score', ascending=False)
    print(scores_df)
    return scores_df

def export_result(scores_df,metrics):

    metrics_df = pd.DataFrame([metrics])
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f'{Project_Path}/results/results_{timestamp}.csv'
    os.makedirs('results', exist_ok=True)
    full_results_df = pd.concat([scores_df, metrics_df], axis=1)
    full_results_df.to_csv(filename, index=False)

    print(f"Export results to CSV in : {filename}")

def plot_prediction_results(y_true, y_pred, model_name, target, start_date, end_date, mae):
    # Convert the data into a DataFrame
    prediction_results = pd.DataFrame(index=y_true.index)
    prediction_results[f'{target} Actual'] = y_true.values
    prediction_results[f'{target} Predicted'] = y_pred

    # Ensure the index is a DatetimeIndex
    if not isinstance(prediction_results.index, pd.DatetimeIndex):
        prediction_results.index = pd.to_datetime(prediction_results.index)

    prediction_results = prediction_results.loc[start_date:end_date]
    prediction_results['Absolute Error'] = np.abs(prediction_results[f'{target} Actual'] - prediction_results[f'{target} Predicted'])

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), sharex=True)

    sns.lineplot(data=prediction_results, x=prediction_results.index, y=f'{target} Actual', ax=axes[0], linewidth=2.5, label=f'{target} Actual')
    sns.lineplot(data=prediction_results, x=prediction_results.index, y=f'{target} Predicted', ax=axes[0], linewidth=2.5, label=f'{target} Predicted',color="#FF941B")
    axes[0].set_title(f'Prediction of {target} with {model_name} model\nMAE: {mae}', fontsize=14, weight='bold')
    axes[0].set_ylabel('Euro (€)', fontsize=12)
    axes[0].legend(title='Legend')
    axes[0].grid(True, which='major', linestyle='-', linewidth=0.5)

    sns.lineplot(data=prediction_results, x=prediction_results.index, y='Absolute Error', ax=axes[1], linewidth=2.5, color='red', label='Absolute Error')
    axes[1].set_title('Absolute Error Over Time', fontsize=14, weight='bold')
    axes[1].set_ylabel('Euro (€)', fontsize=12)
    axes[1].set_xlabel('Days', fontsize=12)
    axes[1].legend(title='Legend')
    axes[1].grid(True, which='major', linestyle='-', linewidth=0.5)

    plt.xticks(rotation=45)

    for ax in axes:
        ax.set_facecolor('#FFFFFF')

    plt.tight_layout()
    plt.show()

# def objective(params):
#         model, duration = RNN_model(x_train, y_train, params)
#         with torch.no_grad():
#                 predictions = model(torch.tensor(x_valid, dtype=torch.float32).unsqueeze(1).to(device))
#                 loss = nn.MSELoss()(predictions, torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1).to(device))
#         return loss.item()   
#         x_train, y_train,x_valid,y_valid = modeltrainer.process_data_and_train_model()



# param_dist = {
# "hidden_size": sp_randint(10, 100),
# "num_layers": sp_randint(1, 3),
# "output_size": [1],  # En général, c'est fixé selon votre problème
# "num_epochs": sp_randint(100, 500),
# "learning_rate": sp_uniform(0.0001, 0.01)
# }

# n_iter_search = 20
# random_search = ParameterSampler(param_dist, n_iter=n_iter_search, random_state=42)

# best_loss = float('inf')
# best_params = None
# for params in random_search:
#         loss = objective(params)
#         if loss < best_loss:
#                 best_loss = loss
#                 best_params = params

# print("Best parameters found: ", best_params)
# print("Best validation loss: ", best_loss)