import pandas as pd
from dask.distributed import Client, wait
from dask_ml.cluster import KMeans
import multiprocessing
import time


if __name__ == '__main__':
    multiprocessing.freeze_support()

    scheduler_address = 'tcp://172.20.10.12:8786'  # Ganti dengan alamat IP Komputer 1

    client = Client(scheduler_address) 

    # Tunggu hingga kedua worker terhubung
    while len(client.scheduler_info()['workers']) < 2:
        print("Menunggu worker terhubung...")
        time.sleep(5)  # Cek setiap 5 detik

    print("Kedua worker telah terhubung. Memulai clustering...")

    # Load dataset (hanya di Komputer 1/server)
    file_path = 'trimmed_online_retail_II.csv' 
    data = pd.read_csv(file_path)
    features = data[['Quantity', 'Price']].fillna(0)

    # Clustering dengan Dask KMeans
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(features) 

    # Dapatkan label cluster dan simpan (hanya di Komputer 1/server)
    labels = kmeans.labels_
    data['Cluster'] = labels
    data.to_csv('clustered_online_retail_II.csv', index=False)
    print("Clustering selesai dan hasil disimpan.")

    client.close()
