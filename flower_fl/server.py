import flwr as fl
import pandas as pd
import numpy as np
import os
from config import (
    SERVER_ADDRESS, NUM_ROUNDS, LOCAL_EPOCHS,
    MIN_FIT_CLIENTS, MIN_EVAL_CLIENTS, MIN_AVAILABLE_CLIENTS,
    RESULTS_DIR
)

class AggregateMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics_records = []

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            print(f"[Server] Round {server_round} - No evaluation results received.")
            return None, {}

        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        for _, result in results:
            self.metrics_records.append({
                "round": server_round,
                "client_id": result.metrics.get("client_id", "unknown"),
                "loss": result.loss,
                "rmse": result.metrics["rmse"],
                "r2": result.metrics["r2"],
                "mae": result.metrics["mae"],
                "mape": result.metrics["mape"],
                "model_type": "client"
            })

        rmse_list = [r.metrics["rmse"] for _, r in results]
        r2_list = [r.metrics["r2"] for _, r in results]
        loss_list = [r.loss for _, r in results]
        mae_list = [r.metrics["mae"] for _, r in results]
        mape_list = [r.metrics["mape"] for _, r in results]

        avg_rmse = float(np.mean(rmse_list))
        avg_r2 = float(np.mean(r2_list))
        avg_loss = float(np.mean(loss_list))
        avg_mae = float(np.mean(mae_list))
        avg_mape = float(np.mean(mape_list))

        self.metrics_records.append({
            "round": server_round,
            "client_id": "global",
            "loss": avg_loss,
            "rmse": avg_rmse,
            "r2": avg_r2,
            "mae": avg_mae,
            "mape": avg_mape,
            "model_type": "global"
        })

        print(f"\n[Server] Round {server_round} Results:")
        print(f"  → Global RMSE: {avg_rmse:.2f}")
        print(f"  → Global R²:   {avg_r2:.4f}")
        print(f"  → Global MAE:  {avg_mae:.2f}")
        print(f"  → Global MAPE:{avg_mape:.2f}%")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        filename = f"epoch_{LOCAL_EPOCHS}_num_rounds_{NUM_ROUNDS}.csv"
        df = pd.DataFrame(self.metrics_records)
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)

        return aggregated_loss, {
            "rmse": avg_rmse,
            "r2": avg_r2,
            "mae": avg_mae,
            "mape": avg_mape
        }

def fit_config(server_round):
    return {"epoch": LOCAL_EPOCHS}

strategy = AggregateMetricsStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_evaluate_clients=MIN_EVAL_CLIENTS,
    min_available_clients=MIN_AVAILABLE_CLIENTS,
    on_fit_config_fn=fit_config,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy
    )
