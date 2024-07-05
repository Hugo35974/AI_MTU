from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

# def XGBM_training(x, y):
#     hyperparameters = {'learning_rate': [0.025, 0.0125],
#                        'n_estimators': range(50, 500, 50),
#                        'max_depth': range(3, 14, 2),
#                        'subsample': np.arange(0.1, 1.1, 0.3),
#                        'colsample_bytree': np.arange(0.1, 1.1, 0.3)}

#     model = XGBRegressor(n_jobs=1)

#     rsearch = RandomizedSearchCV(
#         estimator=model, param_distributions=hyperparameters,
#         scoring='neg_mean_absolute_error',
#         n_jobs=10, cv=5, return_train_score=True,verbose=2)

#     rsearch.fit(x, y)
#     print(rsearch.cv_results_)
#     print(rsearch.best_params_)
#     print(rsearch.best_score_)
#     return rsearch.best_estimator_,rsearch


def XGBM_model(x,y,params):

    learning_rate = params["learning_rate"]
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    subsample = params["subsample"]
    colsample_bytree = params["colsample_bytree"]

    start_time = datetime.now()

    XGBM = XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth= max_depth,
                                            subsample= subsample, colsample_bytree = colsample_bytree, random_state=100,n_jobs=10)

    XGBM.fit(x,y)

    end_time = datetime.now()
    duration = (end_time - start_time)

    return XGBM,duration