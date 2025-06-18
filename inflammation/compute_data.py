"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views

class CSVDataSource:
    '''Loads all the inflammation CSV files within a specified directory.'''
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        '''Loads inflammation data from CSV files within a specified directory'''
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(models.load_csv, data_file_paths)
        return list(data)
    
def analyse_data(data_dir):
    """Calculates the standard deviation by day between datasets.
    Gets all the inflammation data from CSV files within a directory, works out the mean
    inflammation value for each day across all datasets, then visualises the
    standard deviation of these means on a graph."""
    data_file_paths = glob.glob(os.path.join(data_dir, 'inflammation*.csv'))
    if len(data_file_paths) == 0:
        raise ValueError(f"No inflammation csv's found in path {data_dir}")
    data = map(models.load_csv, data_file_paths)
    daily_standard_deviation = compute_standard_deviation_by_day(data)

    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    views.visualize(graph_data)
    return daily_standard_deviation

def compute_standard_deviation_by_day(data):
    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation
