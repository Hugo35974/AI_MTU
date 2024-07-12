import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size).to(device)
        self.attn_combine = nn.Linear(self.hidden_size * 2, hidden_size).to(device)

    def forward(self, lstm_output, hidden):
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2).to(device)
        attn_weights = torch.softmax(attn_weights, dim=1).to(device)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1).to(device)
        output = torch.cat((hidden, attn_applied), 1).to(device)
        output = self.attn_combine(output)
        return output
    
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.2).to(device)
        self.attention = Attention(hidden_size)
        self.lstm  = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out) 
        attn_out = self.attention(lstm_out, hn[-1])
        out = self.fc(attn_out)
        return out

        
    def predict(self, x_pred):
        self.eval()
        if isinstance(x_pred, pd.DataFrame):
            x_pred = x_pred.to_numpy()


        x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            outputs = self(x_pred_tensor)

        return outputs.cpu().numpy()

def RNN_model(x, y, params):
    input_size = x.shape[1]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    output_size = params["output_size"]
    num_epochs = params["num_epochs"]
    learning_rate = params["learning_rate"]

    # Convertir en tensor et déplacer sur le périphérique approprié
    x_train_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device) # Add batch dimension and move to device
    y_train_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    print(x_train_tensor.shape, y_train_tensor.shape)
    model = RNNModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = datetime.now()

    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    end_time = datetime.now()
    duration = end_time - start_time

    return model, duration

