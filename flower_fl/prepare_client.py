# prepare_clients.py

import pandas as pd
import os

df = pd.read_csv("traffic.csv")
locations = df["location"].unique()

num_clients = 10
locations_per_client = len(locations) // num_clients
os.makedirs("client_datasets", exist_ok=True)

for i in range(num_clients):
    client_locs = locations[i * locations_per_client : (i + 1) * locations_per_client]
    client_df = df[df["location"].isin(client_locs)]
    client_df.to_csv(f"client_datasets/client_data_{i+1}.csv", index=False)

print("✅ Tüm client verileri oluşturuldu.")
