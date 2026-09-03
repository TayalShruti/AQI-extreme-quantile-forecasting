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


class QuantileMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_steps, quantiles, dropout=0.0):
        super().__init__()
        self.num_steps = num_steps
        self.num_quantiles = len(quantiles)

        layers = []
        in_dim = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()

        self.post_norm = nn.LayerNorm(in_dim) if hidden_sizes else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_steps * self.num_quantiles)

    def forward(self, x):
        h = self.feature_extractor(x)
        h = self.post_norm(h)
        h = self.dropout(h)
        out = self.fc(h)
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

    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = [
        trial.suggest_int(f"n_neurons_L{i+1}", 50, 300, step=50)
        for i in range(n_layers)
    ]
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lookback = trial.suggest_categorical("lookback", [10, 25, 40, 55, 70])
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
        train_target   = train_df[target_col].values.ravel()

        test_features = test_df[features_list].values
        test_target   = test_df[target_col].values.ravel()

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        train_features_scaled = feature_scaler.fit_transform(train_features)
        test_features_scaled  = feature_scaler.transform(test_features)

        train_target_scaled = target_scaler.fit_transform(train_target.reshape(-1, 1)).ravel()
        test_target_scaled  = target_scaler.transform(test_target.reshape(-1, 1)).ravel()

        X_train, Y_train = create_sequences(train_features_scaled, train_target_scaled,
                                            lookback, 1)

        X_test, Y_test = create_sequences(test_features_scaled, test_target_scaled,
                                          lookback, 1)

        train_loader = DataLoader(TimeSeriesDataset(X_train, Y_train),
                                  batch_size=batch_size, shuffle=True)

        test_loader  = DataLoader(TimeSeriesDataset(X_test, Y_test),
                                  batch_size=batch_size, shuffle=False)

        model = QuantileMLP(
            input_size=X_train.shape[1] * X_train.shape[2],
            hidden_sizes=hidden_sizes,
            num_steps=Y_train.shape[1],
            quantiles=quantiles,
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
                xb = xb.view(xb.size(0), -1).to(device)
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
                xb = xb.view(xb.size(0), -1).to(device)
                yb = yb.to(device)
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


class QuantileMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_steps, quantiles, dropout=0.0):
        super().__init__()
        self.num_steps = num_steps
        self.num_quantiles = len(quantiles)

        layers = []
        in_dim = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()

        self.post_norm = nn.LayerNorm(in_dim) if hidden_sizes else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_steps * self.num_quantiles)

    def forward(self, x):
        h = self.feature_extractor(x)
        h = self.post_norm(h)
        h = self.dropout(h)
        out = self.fc(h)
        return out.view(-1, self.num_steps, self.num_quantiles)
        

df = pd.read_csv("Delhi.csv")

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y')
df = df.reset_index(drop=True)
df = df.set_index('Timestamp')

features_list =  ['AQI', 'PM2.5 ', 'PM10 ', 'winddir', 'Solar Radiation', 'precipprob', 'Max temp', 'Min temp', 'precip', 'windspeed', 'NO2 ', 'SO2 ', 'Max 8-h CO', 'Max 8-h Ozone', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precipcover', 'windgust', 'sealevelpressure', 'cloudcover', 'visibility', 'solarenergy', 'uvindex', 'NH3']
target_col = "AQI"

forecast_horizon = 1
quantiles = globals().get("quantiles", [0.90, 0.95, 0.97])


test_year = 2024
lookback = 10
batch_size = 32
dropout = 0.32555361359025964
hidden_sizes = [50, 150, 200]
quantiles = [0.90, 0.95, 0.97]
num_epochs = 100  
lr = 4.648723255851337e-05

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
    test_df  = df[(df.index >= m_start - pd.Timedelta(days=lookback)) & (df.index <= m_end)].copy()

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
    
    # Model Training
    input_size = X_train.shape[1] * X_train.shape[2] # lookback * n_features 
    model = QuantileMLP(input_size, hidden_sizes, forecast_horizon, quantiles, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.view(inputs.size(0), -1).to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = quantile_loss(outputs, targets, quantiles)
            loss.backward()
            optimizer.step()

    # Prediction
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
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