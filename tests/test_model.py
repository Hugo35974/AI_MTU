import os
from pathlib import Path

Test_Path = Path(__file__).parents[0]

from src.Runner.Runner import Run


def test_model():

    ML = ["Ridge_MinMaxScaler"]

    model = os.path.join(Test_Path,"data_test")

    runner_ml = Run(ML)
    runner_ml.model_path = model
    result = runner_ml.run()

    assert round(result[0][7],2) == 22.87

