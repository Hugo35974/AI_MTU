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