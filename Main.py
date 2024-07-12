from src.Model.XGB import XGBM_model, XGBM_training
from src.Model.LGB import LGBM_model, LGBM_training
from src.Model.RFR import RFRM_model, RandomForest_training
from src.Model.RNN import RNN_model
from src.Runner.Runner import Run

ai_functions = {
    'LGB': LGBM_model,
    'XGB': XGBM_model,
    'RFR': RFRM_model,
    "RNN" : RNN_model
}
runner = Run(ai_functions)
runner.run()
