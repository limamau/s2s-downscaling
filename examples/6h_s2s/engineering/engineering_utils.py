import numpy as np
import xarray as xr
from netCDF4 import Dataset

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
        print("lead_time:", lead_time)
        lead_times.append(lead_time)
        
        for i, file in enumerate(file_tuple):
            ds = xr.open_dataset(file)
            idxs_to_keep = get_s2s_idxs_to_keep(ds.time.values, storm_dates[i])
            idxs_to_keep_diff = np.concatenate([np.array([idxs_to_keep[0]-1]), idxs_to_keep])
            print("idxs_to_keep_diff:", idxs_to_keep_diff)
            precip = np.expand_dims(
                np.diff(ds.tp.values[idxs_to_keep_diff], n=1, axis=0), 
                axis=0,
            )
            precip = precip / 6
            precip = np.where(precip < 0, 0, precip)
            
            # check for negative values
            if (precip < -1).any():
                print("Negative values found in the data")
                times = ds.time.values[idxs_to_keep]
                # print("times:", times)
            
            if raw_s2s_lats is None:
                raw_s2s_lats = ds.latitude.values
                raw_s2s_lons = ds.longitude.values
                raw_s2s_precip = precip
                # print("First precip shape:", raw_s2s_precip.shape)
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
        
    # clip values
    raw_s2s_precip = np.where(raw_s2s_precip < 0, 0, raw_s2s_precip)
    
    return ForecastSurfaceData(
        lead_times, times, raw_s2s_lats, raw_s2s_lons, precip=raw_s2s_precip,
    )


def aggregate_single_ens_s2s_precip(lead_time_files, storm_dates, num_idx):
    lead_times = []
    times = []
    raw_s2s_lats = None
    aux_raw_s2s_data = None
    raw_s2s_lons = None
    raw_s2s_precip = None
    
    # Read and aggregate
    first = True
    for lead_time, file_tuple in lead_time_files.items():
        print("lead_time:", lead_time)
        lead_times.append(lead_time)
        
        for i, file in enumerate(file_tuple):
            with Dataset(file, 'r') as nc_file:
                tp = nc_file.variables['tp'][:,num_idx,...]
                # tp_var = nc_file.variables['tp']
                # scale_factor = getattr(tp_var, 'scale_factor', None)
                # add_offset = getattr(tp_var, 'add_offset', None)
                # tp = tp * scale_factor + add_offset
                lat = nc_file.variables['latitude'][:]
                lon = nc_file.variables['longitude'][:]
            print("tp shape:", tp.shape)
            time = xr.open_dataset(file).time.values
            idxs_to_keep = get_s2s_idxs_to_keep(time, storm_dates[i])
            problematic_idxs = check_monotonic(tp)
            # print("tp[0]:", tp[problematic_idxs[0], 600:605, 1200:1205])
            # print("tp[0+1]:", tp[problematic_idxs[0]+1, 600:605, 1200:1205])
            idxs_to_keep_diff = np.concatenate(
                [np.array([idxs_to_keep[0]-1]), idxs_to_keep]
            )
            # print("idxs_to_keep_diff:", idxs_to_keep_diff)
            precip = np.expand_dims(
                np.diff(tp[idxs_to_keep_diff], n=1, axis=0), 
                axis=0,
            )
            precip = precip / 6
            # precip = np.where(precip < 0, 0, precip)
            
            # check for negative values
            if (precip < -1).any():
                print("Negative values found in the data")
                times = time[idxs_to_keep]
                # print("times:", times)
            
            if raw_s2s_lats is None:
                raw_s2s_lats = lat
                raw_s2s_lons = lon
                raw_s2s_precip = precip
                # print("First precip shape:", raw_s2s_precip.shape)
                times = time[idxs_to_keep]
            
            elif first:
                raw_s2s_precip = np.concatenate([raw_s2s_precip, precip], axis=1)
                times = np.concatenate([times, time[idxs_to_keep]])
            
            else:
                if aux_raw_s2s_data is None:
                    aux_raw_s2s_data = precip
                else:
                    aux_raw_s2s_data = np.concatenate([aux_raw_s2s_data, precip], axis=1)
        
        if not first:
            raw_s2s_precip = np.concatenate([raw_s2s_precip, aux_raw_s2s_data], axis=0)
            aux_raw_s2s_data = None
            
        first = False
        
    # clip values
    raw_s2s_precip = np.where(raw_s2s_precip < 0, 0, raw_s2s_precip)
    
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
            precip = np.where(precip < 0, 0, precip)
            
            if raw_s2s_lats is None:
                numbers = ds.number.values
                raw_s2s_lats = ds.latitude.values
                raw_s2s_lons = ds.longitude.values
                raw_s2s_precip = precip
                # print("First precip shape:", raw_s2s_precip.shape)
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
        
    # clip values
    raw_s2s_precip = np.where(raw_s2s_precip < 0, 0, raw_s2s_precip)
    
    # print("precip shape:", raw_s2s_precip.shape)
    return ForecastEnsembleSurfaceData(
        lead_times, numbers, times, raw_s2s_lats, raw_s2s_lons, precip=raw_s2s_precip,
    )
    

def check_monotonic(tp):
    problematic_idxs = []
    for i in range(tp.shape[0]-1):
        if not np.all(tp[i+1] >= tp[i]):
            problematic_idxs.append(i)
            # print("Non-monotonic values found in the data at idx ", i)
    return problematic_idxs


def get_s2s_idxs_to_keep(times, bound_dates):
    idxs_to_keep = []
    start, end = bound_dates
    start = np.datetime64(start) + np.timedelta64(6, 'h')
    end = np.datetime64(end) + np.timedelta64(24, 'h')
    idxs_to_keep = [i for i, t in enumerate(times) if start <= t <= end]
    return np.sort(idxs_to_keep)
