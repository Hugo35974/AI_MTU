from lightgbm import LGBMRegressor
import numpy as np
from datetime import datetime

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