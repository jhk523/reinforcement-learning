# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:0")

flight_data = sns.load_dataset('flights')
print(flight_data)
# print(flight_data.shape)

# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 15
# fig_size[1] = 5
# plt.rcParams["figure.figsize"] = fig_size
#
# plt.title('Month vs Passenger')
# plt.ylabel('Total Passengers')
# plt.xlabel('Months')
# plt.grid(True)
# plt.autoscale(axis='x',tight=True)
# plt.plot(flight_data['passengers'])
# plt.show()

# Data Preprocessing
all_data = flight_data['passengers'].values.astype(float)

test_data_size = 12

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

print()
print("Train data:", len(train_data))
print("Test data:", len(test_data))

# Normalize data using min/max scaler with minimum and maximum values of -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

print(train_data_normalized[:5])
print(train_data_normalized[-5:])

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
print(train_data_normalized)


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


train_window = 12
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
print(train_inout_seq)

model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print()
print(model)

epochs = 150

model.train()

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq.to(device))

        single_loss = loss_function(y_pred, labels.to(device))
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# Make Predictions
fut_pred = 12

test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)

# Test Predictions
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        test_inputs.append(model(seq.to(device)).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
print(actual_predictions)

x = np.arange(132, 144, 1)
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.plot(x,actual_predictions)
plt.show()