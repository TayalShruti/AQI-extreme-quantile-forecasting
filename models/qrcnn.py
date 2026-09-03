#Hyperparameter Optimization

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class QuantileCNN(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, output_steps,
                 num_quantiles, n_conv_layers, fc_hidden, dropout):
        super(QuantileCNN, self).__init__()

        self.num_quantiles = num_quantiles
        self.output_steps = output_steps
        
        self.conv_stack = nn.Sequential()
        in_channels = input_size
        curr_filters = num_filters

        for i in range(n_conv_layers):
            self.conv_stack.add_module(
                f"block_{i}", 
                ConvBlock(in_channels, curr_filters, kernel_size)
            )
            in_channels = curr_filters
            curr_filters *= 2

        self.gap = nn.AdaptiveAvgPool1d(1)

        final_channels = in_channels 
        
        self.fc_stack = nn.Sequential(
            nn.Linear(final_channels, fc_hidden),
            nn.LayerNorm(fc_hidden), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, output_steps * num_quantiles)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_stack(x)
        x = self.gap(x).view(x.size(0), -1) 
        x = self.fc_stack(x)

        return x.view(-1, self.output_steps, self.num_quantiles)
        

df = pd.read_csv("Delhi.csv")

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y')
df = df.reset_index(drop=True)
df = df.set_index('Timestamp')

features_list = ['AQI', 'PM2.5 ', 'PM10 ', 'winddir', 'Solar Radiation', 'precipprob', 'Max temp', 'Min temp', 'precip', 'windspeed', 'NO2 ', 'SO2 ', 'Max 8-h CO', 'Max 8-h Ozone', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precipcover', 'windgust', 'sealevelpressure', 'cloudcover', 'visibility', 'solarenergy', 'uvindex', 'NH3']

target_col = "AQI"

forecast_horizon = 1
quantiles = globals().get("quantiles", [0.90, 0.95, 0.97])


def objective(trial):

    lookback=trial.suggest_categorical("lookback", [10, 25, 40, 55, 70])
    epochs = 100  
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 3)
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64, 128])
    fc_hidden = trial.suggest_int("fc_hidden", 50, 300, step=50)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"]).lower()

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

        X_train, Y_train = create_sequences(train_features_scaled, train_target_scaled, lookback, forecast_horizon)
        X_test, Y_test = create_sequences(test_features_scaled, test_target_scaled, lookback, forecast_horizon)

        train_loader = DataLoader(TimeSeriesDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TimeSeriesDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

        model = QuantileCNN(
            input_size=X_train.shape[2],
            num_filters=num_filters,
            kernel_size=kernel_size,
            output_steps=Y_train.shape[1],
            num_quantiles=len(quantiles),
            n_conv_layers=n_conv_layers,
            fc_hidden=fc_hidden,
            dropout=dropout
        ).to(device)

        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=lr)

        # train
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

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
        errors = target - preds[:, :, i]  
        loss_q = torch.max((q - 1) * errors, q * errors).mean()
        losses.append(loss_q)
    return torch.mean(torch.stack(losses))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class QuantileCNN(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, output_steps,
                 num_quantiles, n_conv_layers, fc_hidden, dropout):
        super(QuantileCNN, self).__init__()

        self.num_quantiles = num_quantiles
        self.output_steps = output_steps
        
        self.conv_stack = nn.Sequential()
        in_channels = input_size
        curr_filters = num_filters

        for i in range(n_conv_layers):
            self.conv_stack.add_module(
                f"block_{i}", 
                ConvBlock(in_channels, curr_filters, kernel_size)
            )
            in_channels = curr_filters
            curr_filters *= 2

        self.gap = nn.AdaptiveAvgPool1d(1)

        final_channels = in_channels 
        
        self.fc_stack = nn.Sequential(
            nn.Linear(final_channels, fc_hidden),
            nn.LayerNorm(fc_hidden), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, output_steps * num_quantiles)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_stack(x)
        x = self.gap(x).view(x.size(0), -1) 
        x = self.fc_stack(x)
        return x.view(-1, self.output_steps, self.num_quantiles)
        

df = pd.read_csv("Delhi.csv")

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y')
df = df.reset_index(drop=True)
df = df.set_index('Timestamp')

features_list = ['AQI', 'PM2.5 ', 'PM10 ', 'winddir', 'Solar Radiation', 'precipprob', 'Max temp', 'Min temp', 'precip', 'windspeed', 'NO2 ', 'SO2 ', 'Max 8-h CO', 'Max 8-h Ozone', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precipcover', 'windgust', 'sealevelpressure', 'cloudcover', 'visibility', 'solarenergy', 'uvindex', 'NH3']

target_col = "AQI"

forecast_horizon = 1
quantiles = globals().get("quantiles", [0.90, 0.95, 0.97])

test_year = 2024
num_filters=64
kernel_size=4
n_conv_layers=2
fc_hidden=50
lookback = 55
batch_size = 64
dropout = 0.45192835396171444
quantiles = [0.90, 0.95, 0.97]
num_epochs = 100  
lr = 0.0007922979263807968

all_preds = []
all_actuals=[]

start_date = pd.Timestamp("2017-07-01")
initial_train_end = pd.Timestamp("2023-12-31")
test_start = pd.Timestamp("2024-01-01")
test_end = pd.Timestamp("2024-12-31")

month_starts = pd.date_range(test_start, test_end, freq="MS")  
month_ends = pd.date_range(test_start, test_end, freq="M")     

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
    model = QuantileCNN(
    input_size=input_size,
    num_filters=num_filters,
    kernel_size=kernel_size,
    output_steps=forecast_horizon,
    num_quantiles=len(quantiles),
    n_conv_layers=n_conv_layers,
    fc_hidden=fc_hidden,
    dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

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