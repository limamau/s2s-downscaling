import ml_collections
import numpy as np

def get_config():
    config = ml_collections.ConfigDict()
    config.extent = (2255000, 2965000, 840000, 1480000)  # xmin, xmax, ymin, ymax
    config.new_extent = (5.9, 10.6, 45.8, 47.9) # lon_min, lon_max, lat_min, lat_max
    config.num_workers = 5
    # config.years = [2018, 2019, 2020, 2021, 2022, 2023]
    config.years = [2018, 2021]
    # config.months = [6, 7, 8]
    config.months = [6]
    config.validation_dates = (
        (np.datetime64('2023-01-01T06'), np.datetime64('2024-01-01T00')),
    )
    config.test_dates = (
        (np.datetime64('2018-06-11T06'), np.datetime64('2018-06-13T00')),
        (np.datetime64('2021-06-28T06'), np.datetime64('2021-06-30T00')),
    )
    
    return config
