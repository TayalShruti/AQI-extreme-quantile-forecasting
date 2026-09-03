#Hyperparameter Optimization

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import optuna


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]   
        y = self.outputs[idx]  
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


def create_sequences(features, target, lookback, forecast_horizon):
    X, Y = [], []
    n = len(features)
    for i in range(n - lookback - forecast_horizon + 1):
        X.append(features[i : i + lookback])
        Y.append(target[i + lookback : i + lookback + forecast_horizon])
    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(Y)


def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]  
        loss_q = torch.max((q - 1) * errors, q * errors).mean()
        losses.append(loss_q)
    return torch.mean(torch.stack(losses))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class QuantileTCN(nn.Module):
    def __init__(self, input_dim, num_channels, num_steps, quantiles, kernel_size=2, dropout=0.2):
        super(QuantileTCN, self).__init__()

        self.num_steps = num_steps
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        last_channels = num_channels[-1]

        self.layer_norm = nn.LayerNorm(last_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(last_channels, num_steps * self.num_quantiles)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        last = out[:, :, -1]
        last = self.layer_norm(last)
        last = self.dropout(last)
        out = self.fc(last)

        return out.view(-1, self.num_steps, self.num_quantiles)
        

df = pd.read_csv("Delhi.csv")

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y')
df = df.reset_index(drop=True)
df = df.set_index('Timestamp')

features_list = ['AQI', 'PM2.5 ', 'PM10 ', 'winddir', 'Solar Radiation', 'precipprob', 'Max temp', 'Min temp', 'precip', 'windspeed', 'NO2 ', 'SO2 ', 'Max 8-h CO', 'Max 8-h Ozone', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precipcover', 'windgust', 'sealevelpressure', 'cloudcover', 'visibility', 'solarenergy', 'uvindex', 'NH3']

target_col = "AQI"

forecast_horizon = 1
quantiles = globals().get("quantiles", [0.90, 0.95, 0.97])


def objective(trial):
    lookback = trial.suggest_categorical("lookback", [10, 25, 40, 55, 70])
    n_layers = trial.suggest_int("n_layers", 1, 3)   
    num_channels = [
        trial.suggest_categorical(f"n_channels_L{i+1}", [32, 64])
        for i in range(n_layers)
    ]
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "RMSprop"]).lower()
    epochs = 100  

    start_date = pd.Timestamp("2017-01-01")
    initial_train_end = pd.Timestamp("2023-06-30")
    test_start = pd.Timestamp("2023-07-01")
    test_end = pd.Timestamp("2023-12-31")

    month_starts = pd.date_range(test_start, test_end, freq="MS")  
    month_ends = pd.date_range(test_start, test_end, freq="ME")     

    daily_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_start = start_date
    train_end = initial_train_end

    for m_start, m_end in zip(month_starts, month_ends):
        print(f"\n=== Predicting {m_start.strftime('%B %Y')} ===")

        train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
        test_df = df[(df.index >= m_start - pd.Timedelta(days=lookback)) & (df.index <= m_end)].copy()

        train_features = train_df[features_list].values
        train_target = train_df[target_col].values.ravel()
        test_features = test_df[features_list].values
        test_target = test_df[target_col].values.ravel()

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        train_features_scaled = feature_scaler.fit_transform(train_features)
        test_features_scaled = feature_scaler.transform(test_features)
        train_target_scaled = target_scaler.fit_transform(train_target.reshape(-1, 1)).ravel()
        test_target_scaled = target_scaler.transform(test_target.reshape(-1, 1)).ravel()

        X_train, Y_train = create_sequences(train_features_scaled, train_target_scaled,
                                            lookback, forecast_horizon)
        X_test, Y_test = create_sequences(test_features_scaled, test_target_scaled,
                                          lookback, forecast_horizon)

        train_loader = DataLoader(TimeSeriesDataset(X_train, Y_train),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TimeSeriesDataset(X_test, Y_test),
                                 batch_size=batch_size, shuffle=False)

        model = QuantileTCN(input_dim=X_train.shape[2],
                            num_channels=num_channels,
                            num_steps=Y_train.shape[1],
                            quantiles=quantiles,
                            kernel_size=kernel_size,
                            dropout=dropout).to(device)

        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=lr)

        # train
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = quantile_loss(preds, yb, quantiles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # validate
        model.eval()
        with torch.no_grad():
            losses = []
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = quantile_loss(preds, yb, quantiles)
                losses.append(loss.item())
            if losses:
                daily_losses.append(np.mean(losses))

        days_in_month = (m_end - m_start).days + 1
        train_start += pd.Timedelta(days=days_in_month)
        train_end += pd.Timedelta(days=days_in_month)

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(np.mean(daily_losses))

# run optuna study
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)

print("Best Configuration:", study.best_trial.params)
print("Best Avg Quantile Loss (averaged over folds):", study.best_trial.value)

#####Running the final model on the testing period

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]   
        y = self.outputs[idx]  
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


def create_sequences(features, target, lookback, forecast_horizon):
    X, Y = [], []
    n = len(features)
    for i in range(n - lookback - forecast_horizon + 1):
        X.append(features[i : i + lookback])
        Y.append(target[i + lookback : i + lookback + forecast_horizon])
    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(Y)


def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]  # (batch, num_steps)
        loss_q = torch.max((q - 1) * errors, q * errors).mean()
        losses.append(loss_q)
    return torch.mean(torch.stack(losses))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class QuantileTCN(nn.Module):
    def __init__(self, input_dim, num_channels, num_steps, quantiles, kernel_size=2, dropout=0.2):
        super(QuantileTCN, self).__init__()

        self.num_steps = num_steps
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        last_channels = num_channels[-1]
        
        self.layer_norm = nn.LayerNorm(last_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(last_channels, num_steps * self.num_quantiles)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        last = out[:, :, -1]
        last = self.layer_norm(last)
        last = self.dropout(last)
        out = self.fc(last)

        return out.view(-1, self.num_steps, self.num_quantiles)


df = pd.read_csv("Delhi.csv")

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y')
df = df.reset_index(drop=True)
df = df.set_index('Timestamp')

features_list = ['AQI', 'PM2.5 ', 'PM10 ', 'winddir', 'Solar Radiation', 'precipprob', 'Max temp', 'Min temp', 'precip', 'windspeed', 'NO2 ', 'SO2 ', 'Max 8-h CO', 'Max 8-h Ozone', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precipcover', 'windgust', 'sealevelpressure', 'cloudcover', 'visibility', 'solarenergy', 'uvindex', 'NH3']
target_col = "AQI"

forecast_horizon = 1
quantiles = globals().get("quantiles", [0.90, 0.95, 0.97])

test_year = 2024
n_layers=2
kernel_size=3
lookback = 10
batch_size = 64
dropout = 0.10802420794458775
num_channels=[32, 64]
quantiles = [0.90, 0.95, 0.97]
num_epochs = 100
lr = 0.0007356152105162272

all_preds = []
all_actuals=[]

start_date = pd.Timestamp("2017-07-01")
initial_train_end = pd.Timestamp("2023-12-31")
test_start = pd.Timestamp("2024-01-01")
test_end = pd.Timestamp("2024-12-31")

month_starts = pd.date_range(test_start, test_end, freq="MS")  
month_ends = pd.date_range(test_start, test_end, freq="ME")     

daily_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rolling-window testing

train_start = start_date
train_end = initial_train_end

for m_start, m_end in zip(month_starts, month_ends):
    print(f"\n=== Predicting {m_start.strftime('%B %Y')} ===")

    train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    test_df = df[(df.index >= m_start - pd.Timedelta(days=lookback)) & (df.index <= m_end)].copy()

    train_features = train_df[features_list].values
    train_target = train_df[target_col].values.ravel()
    test_features = test_df[features_list].values
    test_target = test_df[target_col].values.ravel()

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features_scaled = feature_scaler.fit_transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)

    train_target_scaled = target_scaler.fit_transform(train_target.reshape(-1, 1)).ravel()
    test_target_scaled = target_scaler.transform(test_target.reshape(-1, 1)).ravel()

    X_train, Y_train = create_sequences(train_features_scaled, train_target_scaled,
                                        lookback, forecast_horizon)
    X_test, Y_test = create_sequences(test_features_scaled, test_target_scaled,
                                      lookback, forecast_horizon)

    train_loader = DataLoader(TimeSeriesDataset(X_train, Y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, Y_test),
                             batch_size=batch_size, shuffle=False)  

    # Model training
    input_size = X_train.shape[2]
    model = QuantileTCN(input_dim=X_train.shape[2],
                            num_channels=num_channels,
                            num_steps=Y_train.shape[1],
                            quantiles=quantiles,
                            kernel_size=kernel_size,
                            dropout=dropout).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = quantile_loss(outputs, targets, quantiles)
            loss.backward()
            optimizer.step()

    # Prediction
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  

            out = outputs.cpu().numpy()    
            B, H, Q = out.shape
            out_2d = out.reshape(-1, 1)    
            out_inv = target_scaler.inverse_transform(out_2d)
            preds = out_inv.reshape(B, H, Q)
            all_preds.append(preds)

            targets_np = targets.cpu().numpy()        

            actuals_orig = target_scaler.inverse_transform(
            targets_np.reshape(-1, 1)
               ).reshape(targets_np.shape)

            all_actuals.append(actuals_orig)

    days_in_month = (m_end - m_start).days + 1
    train_start += pd.Timedelta(days=days_in_month)
    train_end += pd.Timedelta(days=days_in_month)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

arr=np.concatenate(all_preds, axis=0)
actual_vals=np.concatenate(all_actuals, axis=0)

def quantile_loss_(actual, predicted, quantile):
    error = actual - predicted
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

qloss_90=quantile_loss_(actual_vals.squeeze(), arr[:,:,0].squeeze(), 0.90)
qloss_95=quantile_loss_(actual_vals.squeeze(), arr[:,:,1].squeeze(), 0.95)
qloss_97=quantile_loss_(actual_vals.squeeze(), arr[:,:,2].squeeze(), 0.97)