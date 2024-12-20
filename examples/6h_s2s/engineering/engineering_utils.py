import numpy as np
import xarray as xr

from data.surface_data import ForecastSurfaceData, ForecastEnsembleSurfaceData


def aggregate_det_s2s_precip(lead_time_files, storm_dates):
    lead_times = []
    times = []
    raw_s2s_lats = None
    aux_raw_s2s_data = None
    raw_s2s_lons = None
    raw_s2s_precip = None
    
    # Read and aggregate
    first = True
    for lead_time, file_tuple in lead_time_files.items():
        lead_times.append(lead_time)
        
        for i, file in enumerate(file_tuple):
            ds = xr.open_dataset(file)
            idxs_to_keep = get_s2s_idxs_to_keep(ds.time.values, storm_dates[i])
            idxs_to_keep_diff = np.concatenate([np.array([idxs_to_keep[0]-1]), idxs_to_keep])
            precip = np.expand_dims(np.diff(ds.tp.values[idxs_to_keep_diff], axis=0), axis=0)
            precip = precip / 6
            
            if raw_s2s_lats is None:
                raw_s2s_lats = ds.latitude.values
                raw_s2s_lons = ds.longitude.values
                raw_s2s_precip = precip
                print("First precip shape:", raw_s2s_precip.shape)
                times = ds.time.values[idxs_to_keep]
            
            elif first:
                raw_s2s_precip = np.concatenate([raw_s2s_precip, precip], axis=1)
                times = np.concatenate([times, ds.time.values[idxs_to_keep]])
            
            else:
                if aux_raw_s2s_data is None:
                    aux_raw_s2s_data = precip
                else:
                    aux_raw_s2s_data = np.concatenate([aux_raw_s2s_data, precip], axis=1)
        
        if not first:
            raw_s2s_precip = np.concatenate([raw_s2s_precip, aux_raw_s2s_data], axis=0)
            aux_raw_s2s_data = None
            
        first = False
    
    return ForecastSurfaceData(
        lead_times, times, raw_s2s_lats, raw_s2s_lons, precip=raw_s2s_precip,
    )


# TODO: break this function with a for loop to aleviate the processing
def aggregate_ens_s2s_precip(lead_time_files, storm_dates):
    lead_times = []
    numbers = []
    times = []
    raw_s2s_lats = None
    aux_raw_s2s_data = None
    raw_s2s_lons = None
    raw_s2s_precip = None
    
    # Read and aggregate
    first = True
    for lead_time, file_tuple in lead_time_files.items():
        lead_times.append(lead_time)
        
        for i, file in enumerate(file_tuple):
            ds = xr.open_dataset(file)
            idxs_to_keep = get_s2s_idxs_to_keep(ds.time.values, storm_dates[i])
            idxs_to_keep_diff = np.concatenate([np.array([idxs_to_keep[0]-1]), idxs_to_keep])
            precip = np.expand_dims(np.diff(ds.tp.values[idxs_to_keep_diff], axis=0), axis=0)
            precip = np.reshape(precip, (1, precip.shape[2], precip.shape[1], *precip.shape[3:]))
            precip = precip / 6
            
            if raw_s2s_lats is None:
                numbers = ds.number.values
                raw_s2s_lats = ds.latitude.values
                raw_s2s_lons = ds.longitude.values
                raw_s2s_precip = precip
                print("First precip shape:", raw_s2s_precip.shape)
                times = ds.time.values[idxs_to_keep]
            
            elif first:
                raw_s2s_precip = np.concatenate([raw_s2s_precip, precip], axis=2)
                times = np.concatenate([times, ds.time.values[idxs_to_keep]])
            
            else:
                if aux_raw_s2s_data is None:
                    aux_raw_s2s_data = precip
                else:
                    aux_raw_s2s_data = np.concatenate([aux_raw_s2s_data, precip], axis=2)
        
        if not first:
            raw_s2s_precip = np.concatenate([raw_s2s_precip, aux_raw_s2s_data], axis=0)
            aux_raw_s2s_data = None
            
        first = False
    
    print("precip shape:", raw_s2s_precip.shape)
    return ForecastEnsembleSurfaceData(
        lead_times, numbers, times, raw_s2s_lats, raw_s2s_lons, precip=raw_s2s_precip,
    )


def get_s2s_idxs_to_keep(times, bound_dates):
    idxs_to_keep = []
    start, end = bound_dates
    start = np.datetime64(start) + np.timedelta64(6, 'h')
    end = np.datetime64(end) + np.timedelta64(24, 'h')
    idxs_to_keep = [i for i, t in enumerate(times) if start <= t <= end]
    return idxs_to_keep