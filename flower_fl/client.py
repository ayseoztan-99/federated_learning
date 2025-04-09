import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from config import SERVER_ADDRESS, CLIENT_DATA_DIR, TH, TD, TW, TP, LOCAL_EPOCHS
from model import build_multi_lstm_model

client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"[Client {client_id}] Initializing...")

file_path = os.path.join(CLIENT_DATA_DIR, f"client_data_{client_id}.csv")
df = pd.read_csv(file_path)
df = df.sort_values(by=["location", "timestep"]).reset_index(drop=True)

scaler = MinMaxScaler()
df["flow"] = scaler.fit_transform(df["flow"].values.reshape(-1, 1))

timesteps_per_day = 288
train_days = 50
test_days = 12
train_size = train_days * timesteps_per_day

X_recent, X_daily, X_weekly, Y = [], [], [], []
locations = df["location"].unique()

for loc in locations:
    df_loc = df[df["location"] == loc].reset_index(drop=True)
    for i in range(train_size, len(df_loc) - TP):
        recent = df_loc["flow"].iloc[i - TH:i].values
        daily = df_loc["flow"].iloc[i - TD - timesteps_per_day:i - timesteps_per_day].values
        weekly = df_loc["flow"].iloc[i - TW - 7 * timesteps_per_day:i - 7 * timesteps_per_day].values
        target = df_loc["flow"].iloc[i:i + TP].values

        if len(recent) == TH and len(daily) == TD and len(weekly) == TW and len(target) == TP:
            X_recent.append(recent)
            X_daily.append(daily)
            X_weekly.append(weekly)
            Y.append(target)

X_recent = np.array(X_recent).reshape(-1, TH, 1)
X_daily = np.array(X_daily).reshape(-1, TD, 1)
X_weekly = np.array(X_weekly).reshape(-1, TW, 1)
Y = np.array(Y).reshape(-1, TP)

split = int(0.8 * len(Y))
x_recent_train, x_recent_test = X_recent[:split], X_recent[split:]
x_daily_train, x_daily_test = X_daily[:split], X_daily[split:]
x_weekly_train, x_weekly_test = X_weekly[:split], X_weekly[split:]
y_train, y_test = Y[:split], Y[split:]

y_train_dict = {
    "output_recent": y_train,
    "output_daily": y_train,
    "output_weekly": y_train,
    "final_output": y_train
}
y_test_dict = {
    "output_recent": y_test,
    "output_daily": y_test,
    "output_weekly": y_test,
    "final_output": y_test
}

model = build_multi_lstm_model(TH, TD, TW, TP)
print(f"[Client {client_id}] Model has been built and compiled.")

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(
            [x_recent_train, x_daily_train, x_weekly_train],
            y_train_dict,
            epochs=LOCAL_EPOCHS,
            verbose=0
        )
        return model.get_weights(), len(x_recent_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        preds = model.predict([x_recent_test, x_daily_test, x_weekly_test], verbose=0)

        pred_final = scaler.inverse_transform(preds[3])
        true_final = scaler.inverse_transform(y_test)

        loss = model.evaluate([x_recent_test, x_daily_test, x_weekly_test], y_test_dict, verbose=0)
        rmse = np.sqrt(mean_squared_error(true_final, pred_final))
        r2 = r2_score(true_final, pred_final)
        mae = np.mean(np.abs(true_final - pred_final))
        mape = np.mean(np.abs((true_final - pred_final) / (true_final + 1e-5))) * 100

        print(f"[Client {client_id}] Loss: {loss[0]:.4f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

        return loss[0], len(x_recent_test), {
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "mape": mape,
            "client_id": client_id
        }

if __name__ == "__main__":
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=FLClient().to_client()
    )
