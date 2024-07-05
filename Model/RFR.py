from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

def RFRM_model(x, y, params):
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    min_samples_split = params["min_samples_split"]
    min_samples_leaf = params["min_samples_leaf"]
    max_features = params["max_features"]
    bootstrap = params["bootstrap"]

    start_time = datetime.now()

    # Création du modèle RandomForestRegressor
    RFR = RandomForestRegressor(n_estimators=n_estimators, 
                               max_depth=max_depth, 
                               min_samples_split=min_samples_split, 
                               min_samples_leaf=min_samples_leaf, 
                               max_features=max_features, 
                               bootstrap=bootstrap,
                               random_state=100,
                               n_jobs=10)

    # Entraînement du modèle
    RFR.fit(x, y)

    end_time = datetime.now()
    duration = (end_time - start_time)

    return RFR, duration