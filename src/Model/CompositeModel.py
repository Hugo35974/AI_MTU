import numpy as np


class CompositeModel:
    def __init__(self, individual_models):
        self.individual_models = individual_models

    def fit(self, x_train, y_train):
        for i, model in enumerate(self.individual_models):
            model.fit(x_train, y_train[:, i].reshape(-1, 1))

    def predict(self, x_test):
        predictions = []
        for model in self.individual_models:
            predictions.append(model.predict(x_test))
        return np.column_stack(predictions)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        :param deep: If True, will return the parameters of the individual models.
        :return: Dictionary of parameters.
        """
        params = {'individual_models': self.individual_models}
        if deep:
            for i, model in enumerate(self.individual_models):
                # Add the parameters of each individual model
                model_params = model.get_params(deep=True)
                for key, value in model_params.items():
                    params[f'individual_models_{i}__{key}'] = value
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        :param params: Dictionary of parameters to set.
        :return: self
        """
        for key, value in params.items():
            if key == 'individual_models':
                self.individual_models = value
            else:
                # Parameters for individual models are in the form 'individual_models_<i>__<param>'
                key_split = key.split('__', 1)
                if len(key_split) == 2:
                    model_idx = int(key_split[0].split('_')[2])
                    param_name = key_split[1]
                    self.individual_models[model_idx].set_params(**{param_name: value})
        return self
