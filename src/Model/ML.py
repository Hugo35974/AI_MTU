from xgboost import XGBRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
    PassiveAggressiveRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR, SVR
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import sys
from pathlib import Path
Project_Path = Path(__file__).parents[0]
sys.path.append(Project_Path)

from .NN import LSTM_Model, GRU_Model, RNN_Model

regression_models = {
    BayesianRidge.__name__: (BayesianRidge, {
        'model__tol': [1e-2, 1e-3, 1e-4],
        'model__alpha_1': [1e-5, 1e-6, 1e-7],
        'model__alpha_2': [1e-5, 1e-6, 1e-7],
        'model__lambda_1': [1e-5, 1e-6, 1e-7],
        'model__lambda_2': [1e-5, 1e-6, 1e-7],
    }),
    DummyRegressor.__name__: (DummyRegressor, {}),
    GaussianProcessRegressor.__name__: (GaussianProcessRegressor, {
        'model__alpha': [1e-8, 1e-9, 1e-10, 1e-11, 1e-12],
        'model__n_restarts_optimizer': [0, 1, 2, 3],
        'model__normalize_y': [True, False],
    }),
    KernelRidge.__name__: (KernelRidge, {
        'model__alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
        'model__kernel': ['linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'cosine'],
        'model__degree': [2, 3, 4, 5, 6],
        'model__coef0': [0.0, 0.5, 1.0],
    }),
    KNeighborsRegressor.__name__: (KNeighborsRegressor, {
        'model__n_neighbors': list(range(1, 50)),
        'model__weights': ['uniform', 'distance'],
        'model__p': [2, 3, 4],
    }),
    NuSVR.__name__: (NuSVR, {
        'model__nu': [0.2, 0.4, 0.6, 0.8],
        'model__C': [0.001, 0.01, 0.1, 1.0],
        'model__kernel': ['rbf', 'sigmoid'],
        'model__gamma': ['scale', 'auto'],
        'model__coef0': [0.0, 0.5, 1.0],
        'model__shrinking': [True, False],
        'model__tol': [1e-3, 1e-4],
        'model__cache_size': [500],
    }),
    PassiveAggressiveRegressor.__name__: (PassiveAggressiveRegressor, {
        'model__C': [0.001, 0.01, 0.1],
        'model__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        'model__epsilon': [0.001, 0.01, 0.1, 1.0],
        'model__early_stopping': [True, False],
    }),
    SGDRegressor.__name__: (SGDRegressor, {
        'model__loss': ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'model__penalty': ['l2', 'l1', 'elasticnet'],
        'model__alpha': [1e-3, 1e-4, 1e-5],
        'model__l1_ratio': [0.0, 0.15, 0.5, 1.0],
        'model__max_iter': [100, 1000, 10000],
        'model__tol': [1e-2, 1e-3, 1e-4],
        'model__epsilon': [0.001, 0.01, 0.1, 1.0],
        'model__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'model__early_stopping': [True, False],
    }),
    XGBRegressor.__name__: (XGBRegressor, {
        'model__n_estimators': [50, 100, 150, 200],
        'model__learning_rate': [0.01, 0.025, 0.1, 0.2],
        'model__max_depth': [3, 5, 7, 9],
        'model__subsample': [0.1, 0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
    }),
    RandomForestRegressor.__name__: (RandomForestRegressor, {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
    }),
    GradientBoostingRegressor.__name__: (GradientBoostingRegressor, {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
    }),
    LinearRegression.__name__: (LinearRegression, {
        # Linear Regression does not have hyperparameters for tuning
    }),
    Ridge.__name__: (Ridge, {
        'model__alpha': [0.1, 1.0, 10.0],
    }),
    Lasso.__name__: (Lasso, {
        'model__alpha': [0.1, 1.0, 10.0],
    }),
    ElasticNet.__name__: (ElasticNet, {
        'model__alpha': [0.1, 1.0, 10.0],
        'model__l1_ratio': [0.1, 0.5, 0.9],
    }),
    SVR.__name__: (SVR, {
        'model__C': [0.1, 1.0, 10.0],
        'model__epsilon': [0.01, 0.1, 0.2],
        'model__kernel': ['linear', 'poly', 'rbf'],
    }),
    DecisionTreeRegressor.__name__: (DecisionTreeRegressor, {
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
    }),
    MLPRegressor.__name__: (MLPRegressor, {
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__solver': ['adam', 'sgd'],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__learning_rate': ['constant', 'adaptive'],
        'model__max_iter':[1000]
    }),
    LSTM_Model.__name__: (LSTM_Model, {
        'model__hidden_size': [50, 100],
        'model__num_layers': [1, 2],
        'model__num_epochs': [100, 200],
        'model__learning_rate': [0.001, 0.01]
    }),
    GRU_Model.__name__: (GRU_Model, {
        'model__hidden_size': [50, 100],
        'model__num_layers': [1, 2],
        'model__num_epochs': [100, 200,1000],
        'model__learning_rate': [0.001, 0.01]
    }),
    RNN_Model.__name__: (RNN_Model, {
        'model__hidden_size': [50, 100],
        'model__num_layers': [1, 2],
        'model__num_epochs': [100, 200],
        'model__learning_rate': [0.001, 0.01]
    })
}
