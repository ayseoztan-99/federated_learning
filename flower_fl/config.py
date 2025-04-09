# config.py

SERVER_ADDRESS = "localhost:8080"
NUM_ROUNDS = 3
LOCAL_EPOCHS = 20
MIN_FIT_CLIENTS = 10
MIN_EVAL_CLIENTS = 10
MIN_AVAILABLE_CLIENTS = 10
RESULTS_DIR = "results"
CLIENT_DATA_DIR = "client_datasets"

# Time window parameters for multi-input LSTM
TP = 12  # Prediction window (1 hour = 12 timesteps)
TH = 24  # Recent past (last 2 hours = 24 timesteps)
TD = 12  # Daily period (same time slot from 1 day ago = 12 timesteps)
TW = 24  # Weekly period (same time slot from 1 week ago = 24 timesteps)
