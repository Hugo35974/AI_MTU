from lightgbm import LGBMRegressor
import numpy as np
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

def LGBM_training(x, y):
    # Hyperparameters grid including your specific parameters
    hyperparameters = {
        'learning_rate': [0.025, 0.0125],
        'n_estimators': range(50, 500, 50),
        'max_depth': range(3, 14, 2),
        'subsample': np.arange(0.1, 1.1, 0.3),
        'colsample_bytree': np.arange(0.1, 1.1, 0.3),
        'num_leaves': [10],  # Fixed num_leaves parameter
        'min_child_samples': [60],  # Fixed min_child_samples parameter
    }

    # Initialize LightGBM Regressor
    model = LGBMRegressor(n_jobs=10)

    # Randomized search cross-validation
    rsearch = RandomizedSearchCV(
        estimator=model,
        param_distributions=hyperparameters,
        scoring='neg_mean_absolute_error',
        n_jobs=10,
        cv=5,
        return_train_score=True,
        verbose=-1
    )

    # Perform randomized search
    rsearch.fit(x, y)


    return rsearch.best_params_,rsearch.best_score_

def LGBM_model(x,y,params):

    num_leaves = params["num_leaves"]
    min_child_samples = params["min_child_samples"]
    learning_rate = params["learning_rate"]
    n_estimators = params["n_estimators"]
    max_depth= params["max_depth"]
    subsample= params["subsample"]
    colsample_bytree = params["colsample_bytree"]

    start_time = datetime.now()

    LGB = LGBMRegressor(num_leaves=num_leaves,min_child_samples = min_child_samples,learning_rate=learning_rate,n_estimators=n_estimators,
                                               max_depth= max_depth,subsample= subsample,colsample_bytree = colsample_bytree,random_state=100,
                                               n_jobs = 10,verbose=-1)

    LGB.fit(x,y)

    end_time = datetime.now()
    duration = (end_time - start_time)
    
    return LGB, duration

