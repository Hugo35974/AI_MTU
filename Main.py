from src.Model.XGB import XGBM_model, XGBM_training
from src.Model.LGB import LGBM_model, LGBM_training
from src.Model.RFR import RFRM_model, RandomForest_training
from src.Model.NN import LSTM_Model,GRU_Model,RNN_Model
from src.Runner.Runner import Run

ml_functions = {
    'LGB': LGBM_model,
    'XGB': XGBM_model,
    'RFR': RFRM_model,
}

runner_ml = Run(ml_functions, plot = True)
runner_ml.run_ml()

# dl_functions = {
#     "LSTM" : LSTM_Model,
#     "GRU" : GRU_Model,
#     "RNN" : RNN_Model
# }
# runner_dl = Run(dl_functions, plot = False)
# runner_dl.run_dl()

