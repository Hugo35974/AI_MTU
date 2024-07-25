from src.Runner.Runner import Run

ml_functions = [
    "BayesianRidge",
    "DummyRegressor",
    "GaussianProcessRegressor",
    "KernelRidge",
    "KNeighborsRegressor",
    "NuSVR",
    "PassiveAggressiveRegressor",
    "SGDRegressor",
    "XGBRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SVR",
    "DecisionTreeRegressor",
    "MLPRegressor"
]

# runner_ml = Run(ml_functions, plot = True)
# runner_ml.run_pipe_ml()

dl_functions = [
    "LSTM_Model",
    # "GRU_Model",
    # "RNN_Model"
    ]

runner_dl = Run(dl_functions, plot = True)
runner_dl.run_pipe_dl()

