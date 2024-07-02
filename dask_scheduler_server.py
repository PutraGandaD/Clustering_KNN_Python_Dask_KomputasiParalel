from dask.distributed import Scheduler
scheduler = Scheduler()
scheduler.start()
print("Scheduler running at:", scheduler.address)
