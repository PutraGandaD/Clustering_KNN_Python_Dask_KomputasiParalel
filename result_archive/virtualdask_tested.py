import pandas as pd
from dask.distributed import Client, LocalCluster
from dask_ml.cluster import KMeans
import multiprocessing  # Tambahkan import ini

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Tambahkan jika perlu

    # Buat klaster lokal dengan 2 worker
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)

    # Load dataset
    file_path = 'trimmed_online_retail_II.csv'  # Sesuaikan dengan path file Anda
    data = pd.read_csv(file_path)
    features = data[['Quantity', 'Price']].fillna(0)

    # Clustering dengan Dask KMeans (gunakan DataFrame asli)
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(features)

    # Dapatkan label cluster
    labels = kmeans.labels_
    data['Cluster'] = labels

    # Simpan hasil
    data.to_csv('clustered_online_retail_II.csv', index=False)  # Sesuaikan path Anda
    print("Clustering selesai dan hasil disimpan.")

    # Tutup klaster lokal
    client.close()
    cluster.close()
