from .datasets import LmdbDataset, LmdbDatasetForSE
from .writer import LmdbWriter
from .prepare import DataPreProcess

def infinite_data_loader(data_loader):
    while True:
        for data in data_loader:
            yield data
