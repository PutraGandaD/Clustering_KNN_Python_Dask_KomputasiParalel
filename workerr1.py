from dask.distributed import Worker
import sys

if __name__ == "__main__":
    scheduler_address = 'tcp://192.168.1.10:8786'  # Ganti dengan alamat scheduler yang sebenarnya
    worker = Worker(scheduler_address)
    worker.start()
