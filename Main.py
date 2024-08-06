from src.Runner.Runner import Run

ml_pipeline = [
    "BayesianRidge",
    # "DummyRegressor",
    # "GaussianProcessRegressor",
    # "KernelRidge",
    # "KNeighborsRegressor",
    # "NuSVR",
    # "PassiveAggressiveRegressor",
    # "SGDRegressor",
    # "XGBRegressor",
    # "RandomForestRegressor",
    # "GradientBoostingRegressor",
    # "LinearRegression",
    # "Ridge",
    # "Lasso",
    # "ElasticNet",
    # "SVR",
    # "DecisionTreeRegressor",
    # "MLPRegressor"
]

runner_ml = Run(ml_pipeline, plot = True)
runner_ml.build_pipeline()

# dl_functions = [
#     "LSTM_Model",
#     "GRU_Model",
#     "RNN_Model"
#     ]

# runner_dl = Run(dl_functions, plot = True)
# runner_dl.build_nn_pipeline()

# ml_run_model = [
#     # "KNeighborsRegressor_MinMaxScaler",
#     "Ridge_MinMaxScaler",
    # ]

# runner_ml = Run(ml_run_model, plot = True)
# runner_ml.run()
# model = ["composite_model"]
# runner_dl = Run(model, plot = True)
# runner_dl.run()
