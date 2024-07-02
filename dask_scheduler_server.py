import os
import random
import numpy as np
import pandas as pd
from dask.distributed import Client, wait
from dask_ml.cluster import KMeans
import multiprocessing
import time

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Set environment variable for deterministic hashing in Python
    os.environ['PYTHONHASHSEED'] = str(0)  # Seed value of 0

    # Set seeds for random number generation
    random.seed(0)
    np.random.seed(0)

    # Connect to the scheduler with the IP address of your main computer
    scheduler_address = 'tcp://172.20.10.12:8786' 

    # Create the Dask client
    client = Client(scheduler_address) 

    # Wait for both worker computers to connect
    while len(client.scheduler_info()['workers']) < 2:
        print("Menunggu worker terhubung...")
        time.sleep(5)  # Check every 5 seconds

    print("Kedua worker telah terhubung. Memulai clustering...")

    # Start the timer
    start_time = time.time()

    # Load the dataset (only on the main computer/server)
    file_path = 'trimmed_online_retail_II.csv' 
    data = pd.read_csv(file_path)
    features = data[['Quantity', 'Price']].fillna(0)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=0)  
    kmeans.fit(features) 

    # Stop the timer and calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Get cluster labels and save results (only on the main computer/server)
    labels = kmeans.labels_
    data['Cluster'] = labels
    data.to_csv('clustered_online_retail_II.csv', index=False)

    print("Clustering selesai dan hasil disimpan.")
    print(f"Waktu eksekusi clustering: {execution_time:.2f} detik")

    # Close the Dask client
    client.close()
