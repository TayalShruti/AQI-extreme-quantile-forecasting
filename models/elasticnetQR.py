import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.preprocessing import MinMaxScaler

def solve_enet_qr(X, y, tau, lambda_reg, alpha):

    n_samples, n_features = X.shape

    beta = cp.Variable(n_features)

    errors = y - X @ beta

    pinball_loss = cp.sum(
        cp.maximum(tau * errors, (tau - 1) * errors)
    ) / n_samples

    l1_penalty = alpha * cp.norm1(beta[1:])        
    l2_penalty = (1 - alpha) * cp.sum_squares(beta[1:])

    objective = cp.Minimize(
        pinball_loss + lambda_reg * (l1_penalty + l2_penalty)
    )

    problem = cp.Problem(objective)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    if beta.value is None:
        return np.zeros(n_features)

    return beta.value

df = pd.read_csv("Delhi.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y')
df = df.sort_values('Timestamp').set_index('Timestamp')

features_list = [
    'AQI', 'PM2.5 ', 'PM10 ', 'NO2 ', 'SO2 ',
    'Max temp', 'Min temp', 'precip',
    'windspeed', 'winddir', 'Max 8-h CO', 'Max 8-h Ozone',
    'Solar Radiation', 'temp', 'feelslikemax',
    'feelslikemin', 'feelslike', 'dew', 'humidity',
    'precipprob', 'precipcover', 'windgust',
    'sealevelpressure', 'cloudcover', 'visibility',
    'solarenergy', 'uvindex', 'NH3'
]

target_col = 'AQI'
lookback = 10
forecast_horizon = 1
quantiles = [0.90, 0.95, 0.97]
lambda_reg = 0.00053
alpha = 0.9

def create_sequences_flat(features, target, lookback, horizon):
    X, y = [], []
    for i in range(len(features) - lookback - horizon + 1):
        X.append(features[i:i + lookback].flatten())
        y.append(target[i + lookback])
    return np.array(X), np.array(y)

start_date = pd.Timestamp("2017-07-01")
initial_train_end = pd.Timestamp("2023-12-31")
test_start = pd.Timestamp("2024-01-01")
test_end = pd.Timestamp("2024-12-31")

month_starts = pd.date_range(test_start, test_end, freq="MS")
month_ends = pd.date_range(test_start, test_end, freq="ME")

train_start = start_date
train_end = initial_train_end

all_preds = []
all_actuals = []

for m_start, m_end in zip(month_starts, month_ends):

    print(f"\n=== Predicting {m_start.strftime('%B %Y')} ===")

    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    test_df = df[(df.index >= m_start - pd.Timedelta(days=lookback)) &
                 (df.index <= m_end)]

    f_scaler = MinMaxScaler()
    t_scaler = MinMaxScaler()

    X_train_raw = f_scaler.fit_transform(train_df[features_list])
    X_test_raw = f_scaler.transform(test_df[features_list])

    y_train_raw = t_scaler.fit_transform(train_df[[target_col]]).ravel()
    y_test_raw = t_scaler.transform(test_df[[target_col]]).ravel()

    X_train, y_train = create_sequences_flat(
        X_train_raw, y_train_raw, lookback, forecast_horizon
    )
    X_test, y_test = create_sequences_flat(
        X_test_raw, y_test_raw, lookback, forecast_horizon
    )

    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    month_preds = np.zeros((len(y_test), len(quantiles)))

    for qi, tau in enumerate(quantiles):
        beta_hat = solve_enet_qr(
            X_train, y_train, tau, lambda_reg, alpha
        )
        month_preds[:, qi] = X_test @ beta_hat

    preds_orig = t_scaler.inverse_transform(month_preds)
    actuals_orig = t_scaler.inverse_transform(y_test.reshape(-1, 1))

    all_preds.append(preds_orig)
    all_actuals.append(actuals_orig)

    days = (m_end - m_start).days + 1
    train_start += pd.Timedelta(days=days)
    train_end += pd.Timedelta(days=days)

all_preds = np.vstack(all_preds)
all_actuals = np.vstack(all_actuals)

import numpy as np
def quantile_loss_(actual, predicted, quantile):
    error = actual - predicted
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

qloss_90=quantile_loss_(all_actuals.squeeze(), all_preds[:,0].squeeze(), 0.90)
qloss_95=quantile_loss_(all_actuals.squeeze(), all_preds[:,1].squeeze(), 0.95)
qloss_97=quantile_loss_(all_actuals.squeeze(), all_preds[:,2].squeeze(), 0.97)