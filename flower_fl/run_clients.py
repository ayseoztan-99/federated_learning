# run_clients.py

import subprocess
import multiprocessing
import time

NUM_CLIENTS = 10  # Number of clients to launch
PYTHON_CMD = "python"  

def run_client(client_id):
    print(f"Client {client_id} is starting...")
    subprocess.run([PYTHON_CMD, "client.py", str(client_id)])

if __name__ == "__main__":
    processes = []
    for i in range(1, NUM_CLIENTS + 1):
        p = multiprocessing.Process(target=run_client, args=(i,))
        p.start()
        processes.append(p)
        time.sleep(1) 

    for p in processes:
        p.join()

    print("All client processes completed.")
