import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from asgl import Regressor

df = pd.read_csv("Delhi.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y')
df = df.sort_values('Timestamp').set_index('Timestamp')

features_list = ['AQI', 'PM2.5 ', 'PM10 ', 'NO2 ', 'SO2 ', 'Max temp', 'Min temp', 'precip',
                 'windspeed', 'winddir', 'Max 8-h CO', 'Max 8-h Ozone', 'Solar Radiation',
                 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity',
                 'precipprob', 'precipcover', 'windgust', 'sealevelpressure', 'cloudcover',
                 'visibility', 'solarenergy', 'uvindex', 'NH3']

target_col = 'AQI'
forecast_horizon = 1
quantiles = [0.90, 0.95, 0.97]

def create_sequences_flat(features, target, lookback, forecast_horizon):
    X, Y = [], []
    n = len(features)
    for i in range(n - lookback - forecast_horizon + 1):
        X.append(features[i : i + lookback].flatten())
        Y.append(target[i + lookback])
    return np.array(X), np.array(Y)

best_lambda = 0.00053

lookback = 10

all_preds = []
all_actuals = []

start_date = pd.Timestamp("2017-07-01")
initial_train_end = pd.Timestamp("2023-12-31")
test_start = pd.Timestamp("2024-01-01")
test_end = pd.Timestamp("2024-12-31")

month_starts = pd.date_range(test_start, test_end, freq="MS")
month_ends = pd.date_range(test_start, test_end, freq="ME")

train_start = start_date
train_end = initial_train_end

for m_start, m_end in zip(month_starts, month_ends):

    print(f"\n=== Predicting {m_start.strftime('%B %Y')} ===")

    train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    test_df = df[(df.index >= m_start - pd.Timedelta(days=lookback)) &
                 (df.index <= m_end)].copy()

    f_scaler, t_scaler = MinMaxScaler(), MinMaxScaler()

    train_feat = f_scaler.fit_transform(train_df[features_list])
    test_feat = f_scaler.transform(test_df[features_list])

    train_targ = t_scaler.fit_transform(train_df[[target_col]]).ravel()
    test_targ = t_scaler.transform(test_df[[target_col]]).ravel()

    X_train, Y_train = create_sequences_flat(
        train_feat, train_targ, lookback, forecast_horizon
    )
    X_test, Y_test = create_sequences_flat(
        test_feat, test_targ, lookback, forecast_horizon
    )

    month_preds = np.zeros((len(Y_test), len(quantiles)))

    for qi, q in enumerate(quantiles):
        model = Regressor(
            model="qr",
            penalization="lasso",
            quantile=q,
            lambda1=best_lambda
        )

        model.fit(X_train, Y_train)
        month_preds[:, qi] = model.predict(X_test)

    preds_orig = t_scaler.inverse_transform(month_preds)
    actuals_orig = t_scaler.inverse_transform(Y_test.reshape(-1, 1))

    all_preds.append(preds_orig)
    all_actuals.append(actuals_orig)

    days_in_month = (m_end - m_start).days + 1
    train_start += pd.Timedelta(days=days_in_month)
    train_end += pd.Timedelta(days=days_in_month)

all_preds = np.vstack(all_preds)        
all_actuals = np.vstack(all_actuals)   

import numpy as np
def quantile_loss_(actual, predicted, quantile):
    error = actual - predicted
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

qloss_90=quantile_loss_(all_actuals.squeeze(), all_preds[:,0].squeeze(), 0.90)
qloss_95=quantile_loss_(all_actuals.squeeze(), all_preds[:,1].squeeze(), 0.95)
qloss_97=quantile_loss_(all_actuals.squeeze(), all_preds[:,2].squeeze(), 0.97)