from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV

def RandomForest_training(x, y):
    # Hyperparameters grid for Random Forest with ranges
    hyperparameters = {
        'n_estimators': range(100, 1001, 100),  # de 100 à 1000 par pas de 100
        'max_depth': range(10, 101, 10),         # de 10 à 100 par pas de 10
        'min_samples_split': range(2, 11),       # de 2 à 10
        'min_samples_leaf': range(1, 11),        # de 1 à 10
        'max_features': ['auto', 'sqrt', 'log2'], # choix de valeurs discrètes
        'bootstrap': [True, False],              # choix de valeurs discrètes
    }

    # Initialize Random Forest Regressor
    model = RandomForestRegressor()

    # Randomized search cross-validation
    rsearch = RandomizedSearchCV(
        estimator=model,
        param_distributions=hyperparameters,
        scoring='neg_mean_absolute_error',
        n_jobs=10,
        cv=5,
        return_train_score=True,
        verbose=2
    )

    # Perform randomized search
    rsearch.fit(x, y)

    # Print results
    print(rsearch.cv_results_)
    print(rsearch.best_params_)
    
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