import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from tabulate import tabulate

from src.Model.CompositeModel import CompositeModel
from src.Pretreatment.ModelTrainer import ModelTrainer
from src.Tools.tools import (
    convert_dates,
    evaluate_model,
    plot_prediction_results,
    save_best_model,
    setup_pipeline_multi_output,
    setup_pipeline_single_output,
    train_and_search,
)
from src.Model.ML import regression_models

# Initialize colorama for colored terminal output
init(autoreset=True)

# Define project paths
PROJECT_PATH = Path(__file__).parents[2]
MODEL_PATH = os.path.join(PROJECT_PATH, "data", "modelsave")
CONTENER_PATH = os.path.join(PROJECT_PATH, "contener")


class Run:
    """
    The Run class is responsible for executing model training and evaluation
    processes. It supports multiple models and scalers.
    """

    def __init__(self, models=None, plot=False):
        """
        Initialize the Run class with optional models and plotting flag.

        :param models: List of model names to use. If None, default models are used.
        :param plot: Boolean flag to indicate whether to plot results.
        """
        self.model_trainer = ModelTrainer()
        self.models = models if models else self.model_trainer.models
        self.plot = plot
        self.model_path = MODEL_PATH

    def _prepare_results(self, model_name, mae):
        """
        Prepare results for a given model.

        :param model_name: Name of the model.
        :param mae: Mean Absolute Error.
        :return: List of results.
        """
        start_date, end_date, start_test, end_test = convert_dates(self.model_trainer)
        result = [
            model_name,
            start_date,
            end_date,
            (end_date - start_date).days,
            start_test,
            end_test,
            (end_test - start_test).days,
            mae,
        ]
        return result

    def _print_results(self, results, headers):
        """
        Print results in a tabulated format with colored headers.

        :param results: List of results to display.
        :param headers: List of header names.
        """
        headers_colored = [Fore.CYAN + header + Style.RESET_ALL for header in headers]
        table = tabulate(
            sorted(results, key=lambda x: x[7]),
            headers=headers_colored,
            tablefmt="pretty",
        )
        print(table)

    def run(self):
        """
        Run the model evaluation and print results.
        """
        results = []
        for model_name in self.models:
            model_file = os.path.join(self.model_path, f"{model_name}.pkl")
            with open(model_file, "rb") as file:
                loaded_model = pickle.load(file)
                (
                    _,
                    _,
                    x_test,
                    y_true,
                    _,
                    _,
                    _,
                ) = self.model_trainer.process_data_and_train_model()
                y_pred = loaded_model.predict(x_test)
                mae = MAE(y_true, y_pred)
                results.append(self._prepare_results(model_name, mae))

        self._print_results(
            results,
            [
                "Model",
                "Start-Day-Train",
                "End-Day-Train",
                "Train-Period",
                "Start-Day-Test",
                "End-Day-Test",
                "Test-Period (days)",
                "MAE",
            ],
        )
        return results

    def configure_pipeline(
        self,
        model_name,
        model,
        model_space,
        model_space_multi,
        scaler,
        x_train,
        y_train,
    ):
        """
        Configure the pipeline and hyperparameter distributions for the given model.

        :param model_name: Name of the model.
        :param model: Model class.
        :param model_space: Hyperparameter space for single output.
        :param model_space_multi: Hyperparameter space for multi-output.
        :param scaler: Scaler to use.
        :param x_train: Training features.
        :param y_train: Training labels.
        :return: Tuple of pipeline and distributions.
        """
        if model_name in ["LSTM_Model", "GRU_Model", "RNN_Model"]:
            pipeline = Pipeline([("scaler", scaler), ("model", model())])
            distributions = model_space
        else:
            if y_train.ndim == 1:
                pipeline, distributions = setup_pipeline_single_output(
                    scaler, model, model_space, x_train
                )
                y_train = np.ravel(y_train)
            else:
                pipeline, distributions = setup_pipeline_multi_output(
                    scaler, model, model_space_multi, x_train
                )

        return pipeline, distributions

    def build_pipeline(self, df=None, model_infos=None):
        """
        Build and evaluate pipelines for each model and scaler combination.

        :param df: DataFrame to use. Default is None.
        :param model_infos: Information about models. Default is None.
        :return: Results DataFrame.
        """
        (
            x_train,
            y_train,
            x_test,
            y_test,
            _,
            _,
            config,
        ) = self.model_trainer.process_data_and_train_model(df, model_infos)
        scalers = [
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
            QuantileTransformer(),
        ]
        results = []

        for model_name in self.models:
            model, model_space = regression_models[model_name]
            model_space_multi = {
                f"multioutput__estimator__{key}": value
                for key, value in model_space.items()
            }

            for scaler in scalers:
                print(f"Model: {model_name} -> Scaler: {scaler}")

                pipeline, distributions = self.configure_pipeline(
                    model_name,
                    model,
                    model_space,
                    model_space_multi,
                    scaler,
                    x_train,
                    y_train,
                )

                search, duration = train_and_search(
                    pipeline, distributions, x_train, y_train
                )
                result = evaluate_model(
                    search.best_estimator_,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    model_name,
                    config,
                    duration,
                    scaler.__class__.__name__,
                )
                results.append(result)

                save_best_model(search.best_estimator_, model_name, scaler, MODEL_PATH)

        results_df = pd.DataFrame(results)
        return self.create_composite_model(
            x_train, y_train, x_test, y_test, results_df, model_infos, config
        )

    def create_composite_model(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        results_df,
        model_infos=None,
        config=None,
    ):
        """
        Create and evaluate a composite model using the best individual models.

        :param x_train: Training features.
        :param y_train: Training labels.
        :param x_test: Testing features.
        :param y_test: Testing labels.
        :param results_df: DataFrame with model evaluation results.
        :param model_infos: Information about models.
        :param config: Configuration settings.
        :return: Evaluation result of the composite model.
        """
        column_name = model_infos["column_name"]
        best_models_per_output = {}

        for i in range(y_train.shape[1]):
            results_df[f"MAE-SRC_T_{i}"] = (
                results_df[f"MAE_T_{i}"] - results_df[f"SRC_T_{i}"]
            )
            best_model_info = results_df.loc[results_df[f"MAE-SRC_T_{i}"].idxmin()]
            best_models_per_output[f"Output_{i}"] = {
                "Model": best_model_info["Model"],
                "R2": best_model_info[f"R2_T_{i}"],
                "MAE": best_model_info[f"MAE_T_{i}"],
                "SRC": best_model_info[f"SRC_T_{i}"],
                "MSE": best_model_info[f"MSE_T_{i}"],
                "RMSE": best_model_info[f"RMSE_T_{i}"],
                "MAPE": best_model_info[f"MAPE_T_{i}"],
            }

        best_models_df = pd.DataFrame(best_models_per_output).T
        best_models_df.to_csv(
            f"data/results/result_CrossVal_{column_name}.csv", index=True
        )

        # Load individual models
        individual_models = []
        for i in range(y_train.shape[1]):
            model_filename = os.path.join(
                MODEL_PATH, best_models_per_output[f"Output_{i}"]["Model"] + ".pkl"
            )
            with open(model_filename, "rb") as file:
                model = pickle.load(file)
                individual_models.append(model)

        # Create and train the composite model
        composite_model = CompositeModel(individual_models)
        composite_model.fit(x_train, y_train)

        if self.plot:
            y_pred = composite_model.predict(x_test)
            plot_prediction_results(
                y_test,
                y_pred,
                "composite_model",
                self.model_trainer.variables_config["target_variable"],
                y_train,
            )

        duration = 1
        result = evaluate_model(
            composite_model,
            x_train,
            y_train,
            x_test,
            y_test,
            model_infos["column_name"],
            config,
            duration,
        )

        model_composite_path = os.path.join(CONTENER_PATH, model_infos["path"])

        result_df = pd.DataFrame([result])

        with open(model_composite_path, "wb") as composite_file:
            pickle.dump(composite_model, composite_file)

        result_df.to_csv(f"data/results/result_Test_{column_name}.csv", index=True)
        print(f"Composite model saved to {model_composite_path}.")
        return result_df
