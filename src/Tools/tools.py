import pandas as pd 
import os
import seaborn as sns
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from scipy.stats import spearmanr
from sklearn.model_selection import RandomizedSearchCV, cross_validate, cross_val_predict
import time
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor

Project_Path = Path(__file__).parents[1]

def mase(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def eval_metrics(y_true,y_pred):
    MAE = mean_absolute_error(y_true,y_pred)
    return MAE

def convert_dates(date_config):
    start_date = pd.to_datetime(date_config.date_s).date()
    end_date = pd.to_datetime(date_config.date_end).date()
    predict_start_date = pd.to_datetime(date_config.predict_s).date()
    predict_end_date = pd.to_datetime(date_config.predict_e).date()
    return start_date, end_date, predict_start_date, predict_end_date

def feature_importance(x, y):

    model = RandomForestRegressor()
    model.fit(x, y)

    return model.feature_importances_

def pearson_corr(x, y):
    corr_matrix = np.corrcoef(x, y, rowvar=False)
    corr_with_target = np.abs(corr_matrix[:-1, -1])  
    return corr_with_target

def shifting(list_toshift: list, df: pd.DataFrame, time_to_shift):
    a, b = time_to_shift
    new_columns = {}
    
    for var in list_toshift:
        for i in range(a, b):
            new_column_name = f'{var}_lag_{i}h'
            new_columns[new_column_name] = df[var].shift(i)
    
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    df = df.dropna()
    
    return df

def shifting_by_day(list_toshift: list, df: pd.DataFrame, time_to_shift):
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

def selectkbest(x_train, y_train, x_test, k, function_name):
    selection_functions = {
        "f_regression": f_regression,
        "pearson_corr": lambda X, y: (np.abs(np.corrcoef(X.T, y)[-1, :-1]), None),
    }
    
    def process_target(y_col):
        selector = SelectKBest(score_func=selection_functions[function_name], k=k)
        x_train_k = selector.fit_transform(x_train, y_col)
        x_test_k = selector.transform(x_test)
        return x_train_k, x_test_k

    if y_train.ndim == 1:
        return process_target(y_train)
    else:
        results = Parallel(n_jobs=-1)(delayed(process_target)(y_train[:, i]) for i in range(y_train.shape[1]))
        x_train_k = np.hstack([r[0] for r in results])
        x_test_k = np.hstack([r[1] for r in results])
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

def multi_step(data, target, horizon):
    """
    Generate multi-step prediction columns.
    """
    namelist = []
    s,t = horizon
    for time_step in range(s,t):
        name = 't' + '+' + str(time_step)
        data[name] = data[target].shift(-time_step)
        namelist.append(name)
    return data, namelist

def remove_rows_hour_col(data,hour):
    """
    Remove rows where the hour is 23 and drop the 'hour' column.
    """
    data = data[data['hour'] == hour]
    data = data.drop(['hour'], axis=1)
    data = data.reset_index(drop=True)
    return data

def plot_prediction_results(y_true, y_pred, model_name, target, y_train):
    # Flatten the arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Create a DataFrame
    prediction_results = pd.DataFrame({
        'Index': range(len(y_true_flat)),
        f'{target} Actual': y_true_flat,
        f'{target} Predicted': y_pred_flat
    })

    # Compute additional scores
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the prediction results
    sns.lineplot(data=prediction_results, x='Index', y=f'{target} Actual', ax=ax, linewidth=2.5, label=f'{target} Actual')
    sns.lineplot(data=prediction_results, x='Index', y=f'{target} Predicted', ax=ax, linewidth=2.5, label=f'{target} Predicted', color="#FF941B")
    ax.set_title(f'Prediction of {target} with {model_name} model\nMAE: {mae:.2f}', fontsize=14, weight='bold')
    ax.set_ylabel('Euro (€)', fontsize=12)
    ax.legend(title='Legend')
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FFFFFF')

    plt.tight_layout()

    # Create a table of the scores
    scores_data = {
        'Metric': ['MAE', 'RMSE', 'R²'],
        'Score': [mae, rmse, r2]
    }
    scores_df = pd.DataFrame(scores_data)

    # Create a figure and axis for the table
    fig, ax_table = plt.subplots(figsize=(6, 2))  # Adjust size as needed
    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=scores_df.values, colLabels=scores_df.columns, cellLoc='center', loc='center')

    plt.show()

    # def shap_lib(self, model, x_train_k, x_test_k):
    #     explainer = shap.Explainer(model, x_train_k)
    #     shap_values = explainer(x_test_k)
    #     shap.summary_plot(shap_values, x_test_k, feature_names=self.modeltrainer.features, plot_type="bar")

def setup_pipeline_single_output(scaler, model, model_space, x_train):

    pipeline = Pipeline([
        ('scaler', scaler),
        ('feature_selector', SelectKBest(f_regression)),
        ('model', model()),
    ],n_jobs=-1)
    distributions = {**model_space, 'feature_selector__k': list(range(0, x_train.shape[1]))}
    return pipeline, distributions

def setup_pipeline_multi_output(scaler, model, model_space_multi, x_train):

    pipeline = Pipeline([
        ('scaler', scaler),
        ('multioutput', MultiOutputRegressor(Pipeline([
            ('feature_selector', SelectKBest(f_regression)),
            ('model', model())
        ]), n_jobs=-1))
    ])
    distributions = {**model_space_multi, 'multioutput__estimator__feature_selector__k': list(range(0, x_train.shape[1]))}
    return pipeline, distributions

def train_and_search(pipeline, distributions, x_train, y_train_ravel):
    clf = RandomizedSearchCV(pipeline, distributions, n_iter=10, cv=5, verbose=1, n_jobs=-1)
    start_time = time.time()
    search = clf.fit(x_train, y_train_ravel)
    duration = time.time() - start_time
    return search, duration

def evaluate_model(search, x_train, y_train_ravel, y_test_ravel, model_name, scaler, features, target, config, duration):
    cv_results = cross_validate(search.best_estimator_, x_train, y_train_ravel, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    y_pred = cross_val_predict(search.best_estimator_, x_train, y_train_ravel, cv=5)

    spearman_per_dimension = [spearmanr(y_train_ravel[:, i], y_pred[:, i])[0] for i in range(y_train_ravel.shape[1])]
    r2_per_dimension = [r2_score(y_train_ravel[:, i], y_pred[:, i]) for i in range(y_train_ravel.shape[1])]
    mae_per_dimension = np.mean(np.abs(y_train_ravel - y_pred), axis=0)

    MAE = np.mean(mae_per_dimension)
    R2 = np.mean(r2_per_dimension)
    SRC = np.mean(spearman_per_dimension)

    result = {
        'Model': model_name + '_' + scaler.__class__.__name__,
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

    # Add MAE, R2, SRC for each dimension as separate columns
    for i in range(len(mae_per_dimension)):
        result[f'MAE_T_{i}'] = mae_per_dimension[i]
    for i in range(len(r2_per_dimension)):
        result[f'R2_T_{i}'] = r2_per_dimension[i]
    for i in range(len(spearman_per_dimension)):
        result[f'SRC_T_{i}'] = spearman_per_dimension[i]

    return result

def save_best_model(best_model, model_name, scaler,modelpath):
    file = os.path.join(modelpath, f'{model_name}_{scaler.__class__.__name__}.pkl')
    with open(file, 'wb') as model_file:
        pickle.dump(best_model, model_file)
    print(f"Model saved: {file}")