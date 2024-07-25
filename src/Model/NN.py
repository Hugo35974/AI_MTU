from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, hidden_size)

    def forward(self, lstm_output, hidden):
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        output = torch.cat((hidden, attn_applied), 1)
        output = self.attn_combine(output)
        return output
    
class Model(nn.Module):
    def __init__(self, model_class, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.2)
        self.attention = Attention(hidden_size)
        self.lstm = model_class(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if isinstance(self.lstm, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            lstm_out, hn = self.lstm(x, h0)
        
        lstm_out = self.dropout(lstm_out)
        attn_out = self.attention(lstm_out, hn[-1])
        out = self.fc(attn_out)
        return out

class ModelRun(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_size=50, num_layers=1, num_epochs=100, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = None
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = None

    def _initialize_model(self, model_class, input_size):
        self.model = Model(model_class, input_size, self.hidden_size, self.num_layers, self.output_size).to(device)

    def fit(self, X, y):
        input_size = X.shape[1]
        # Adjust y shape for consistency
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
        self.output_size = y.shape[1]
        x_train_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        self._initialize_model(self.model_class, input_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        start_time = datetime.now()

        for epoch in range(self.num_epochs):
            self.model.train()
            outputs = self.model(x_train_tensor)
            outputs = outputs.view(-1, self.output_size)
            y_train_tensor = y_train_tensor.view(-1, self.output_size) # Adjust shape for consistency
            loss = criterion(outputs, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        
        self.duration = (datetime.now() - start_time).total_seconds()
        return self

    def predict(self, X):
        self.model.eval()
        x_test_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            predictions = self.model(x_test_tensor).cpu().numpy()
        
        # Adjust predictions shape for consistency
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)

    def get_params(self, deep=True):
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
class LSTM_Model(ModelRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = nn.LSTM

class GRU_Model(ModelRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = nn.GRU

class RNN_Model(ModelRun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = nn.RNN
