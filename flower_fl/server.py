# server_lstm.py

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
            print(f"Round {server_round} - Failed to receive client results.")
            return None, {}

        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # Process client metrics
        for _, result in results:
            client_id = result.metrics.get("client_id", "unknown")
            rmse = result.metrics["rmse"]
            r2 = result.metrics["r2"]
            loss = result.loss

            self.metrics_records.append({
                "round": server_round,
                "client_id": client_id,
                "loss": loss,
                "rmse": rmse,
                "r2": r2,
                "model_type": "client"
            })

        # Calculate global averages
        rmse_list = [r.metrics["rmse"] for _, r in results]
        r2_list = [r.metrics["r2"] for _, r in results]
        loss_list = [r.loss for _, r in results]

        avg_rmse = np.mean(rmse_list)
        avg_r2 = np.mean(r2_list)
        avg_loss = np.mean(loss_list)

        self.metrics_records.append({
            "round": server_round,
            "client_id": "global",
            "loss": avg_loss,
            "rmse": avg_rmse,
            "r2": avg_r2,
            "model_type": "global"
        })

        print(f"\n Round {server_round} results:")
        print(f"Global RMSE: {avg_rmse:.4f}, Global R²: {avg_r2:.4f}")

        # Dynamically save CSV file
        from config import LOCAL_EPOCHS, NUM_ROUNDS, RESULTS_DIR
        import os
        import pandas as pd

        os.makedirs(RESULTS_DIR, exist_ok=True)
        filename = f"epoch_{LOCAL_EPOCHS}_num_rounds_{NUM_ROUNDS}.csv"
        df = pd.DataFrame(self.metrics_records)
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)

        return aggregated_loss, {"rmse": avg_rmse, "r2": avg_r2}


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
