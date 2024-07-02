import pandas as pd
from dask.distributed import Client
from dask_ml.cluster import KMeans

if __name__ == '__main__':
    client = Client('tcp://192.168.1.10:8786')  # Gunakan alamat IP scheduler yang sebenarnya

    file_path = 'trimmed_online_retail_II.csv'
    data = pd.read_csv(file_path)
    features = data[['Quantity', 'Price']].fillna(0)

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(features)
    labels = kmeans.labels_
    data['Cluster'] = labels

    data.to_csv('clustered_online_retail_II.csv', index=False)
    print("Clustering selesai dan hasil disimpan.")
    client.close()
