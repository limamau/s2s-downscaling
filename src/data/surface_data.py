import h5py
import numpy as np
from typing import Tuple
import xarray as xr

from engineering.regridding import interpolate_data
from engineering.spectrum import low_pass_filter
from utils import haversine

class SurfaceData:
    def __init__(
        self, 
        time: np.ndarray[np.datetime64],
        latitude: np.ndarray[np.float64],
        longitude: np.ndarray[np.float64],
        **variables: np.ndarray,
    ):
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        
        for var_name, var_data in variables.items():
            setattr(self, var_name, var_data)
            
        self._var_names = list(variables.keys())
        
    
    def low_pass_filter(self, sfc_data: 'SurfaceData'):
        """Apply a low-pass filter to the data based on the spatial resolution of the input data."""
        data_lengths = self.get_spatial_lengths()
        ref_lengths = sfc_data.get_spatial_lengths()
        
        for var_name in self._var_names:
            setattr(self, var_name, low_pass_filter(
                getattr(self, var_name),
                data_lengths, ref_lengths,
            ))

    
    def cut_data(self, extent):
        """Cut data to a specific extent."""
        lon_min, lon_max, lat_min, lat_max = extent
        lon_idx = (self.longitude >= lon_min) & (self.longitude <= lon_max)
        lat_idx = (self.latitude >= lat_min) & (self.latitude <= lat_max)
        self.longitude = self.longitude[lon_idx]
        self.latitude = self.latitude[lat_idx]
        for var_name in self._var_names:
            setattr(self, var_name, 
                getattr(self, var_name)[..., lat_idx, :][..., lon_idx]
            )


    def get_extent(self):
        """Get the extent of the data."""
        return (
            self.longitude.min(), self.longitude.max(),
            self.latitude.min(), self.latitude.max(),
        )
        
        
    def get_spatial_lengths(self):
        """Calculates the spatial lengths (x_length and y_length)."""
        x_length = haversine(self.longitude.min(), self.latitude.min(), self.longitude.max(), self.latitude.min())
        y_length = haversine(self.longitude.min(), self.latitude.min(), self.longitude.min(), self.latitude.max())
    
        return x_length, y_length
    

    @classmethod
    def load_from_h5(cls, filename, variables_names):
        """Load data from an HDF5 file using h5py for dimensions."""
        with h5py.File(filename, 'r') as f:
            latitude = f['latitude'][:]
            longitude = f['longitude'][:]
            for var_name in variables_names:
                setattr(cls, var_name, f[var_name][...])
        time = xr.open_dataset(filename, engine='h5netcdf').time.values

        return cls(
            time, latitude, longitude, 
            **{var_name: getattr(cls, var_name) for var_name in variables_names}
        )
        
        
    def regrid(self, sfc_data: 'SurfaceData', method: str='linear'):
        """Regrid variables to a new grid."""
        old_lon_2d, old_lat_2d = np.meshgrid(self.longitude, self.latitude)
        new_lon_2d, new_lat_2d = np.meshgrid(sfc_data.longitude, sfc_data.latitude)
        
        for var_name in self._var_names:
            setattr(self, var_name, interpolate_data(
                getattr(self, var_name),
                old_lon_2d, old_lat_2d,
                new_lon_2d, new_lat_2d,
                method=method
            ))
        
        self.latitude = sfc_data.latitude
        self.longitude = sfc_data.longitude

    
    def save_to_h5(self, filename: str):
        """Save data to an HDF5 file using xarray."""
        data_vars = {
            var_name: (['time', 'latitude', 'longitude'], getattr(self, var_name))
            for var_name in self._var_names
        }
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': self.time,
                'latitude': self.latitude,
                'longitude': self.longitude
            }
        )

        ds.to_netcdf(filename, engine='h5netcdf')
        
        
    def take_out_from_date_range(self, date_ranges: Tuple[Tuple[np.datetime64, np.datetime64],...]):
        """Take out data from specific date ranges, returning it as a new object,
        and modify the original object by removing the selected data."""
        # get indexes outside of date_ranges
        date_idx = np.ones(len(self.time), dtype=bool)
        for start_date, end_date in date_ranges:
            date_idx &= (self.time < start_date) | (self.time > end_date)

        # extract data to new array
        new_sfc_data = self.__class__(
            self.time[~date_idx], self.latitude, self.longitude, 
            **{var_name: getattr(self, var_name)[...,~date_idx,:,:] for var_name in self._var_names}
        )

        # update current object
        self.time = self.time[date_idx]
        for var_name in self._var_names:
            setattr(self, var_name, getattr(self, var_name)[...,date_idx,:,:])

        return new_sfc_data
        
        
    def unflip_latlon(self):
        """Sort data by latitude and longitude."""
        if self.latitude[0] > self.latitude[-1]:
            self.latitude = np.flip(self.latitude)
            for var_name in self._var_names:
                setattr(self, var_name, np.flip(getattr(self, var_name), axis=-1))
        if self.longitude[0] > self.longitude[-1]:
            self.longitude = np.flip(self.longitude)
            for var_name in self._var_names:
                setattr(self, var_name, np.flip(getattr(self, var_name), axis=-1))


class EnsembleSurfaceData(SurfaceData):
    def __init__(
        self,
        number: np.ndarray,
        time: np.ndarray, 
        latitude: np.ndarray, 
        longitude: np.ndarray, 
        **variables: np.ndarray,
    ):
        super().__init__(time, latitude, longitude, **variables)
        self.number = number
        
        
    @classmethod
    def load_from_h5(cls, filename, variables_names):
        """Load data from an HDF5 file using h5py for dimensions."""
        with h5py.File(filename, 'r') as f:
            number = f['number'][:]
            latitude = f['latitude'][:]
            longitude = f['longitude'][:]
            for var_name in variables_names:
                setattr(cls, var_name, f[var_name][...])
        time = xr.open_dataset(filename, engine='h5netcdf').time.values

        return cls(
            number, time, latitude, longitude, 
            **{var_name: getattr(cls, var_name) for var_name in variables_names}
        )


    def save_to_h5(self, filename):
        """Save data to an HDF5 file using xarray."""
        data_vars = {
            var_name: (['number', 'time', 'latitude', 'longitude'], getattr(self, var_name))
            for var_name in self._var_names
        }
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'number': self.number,
                'time': self.time,
                'latitude': self.latitude,
                'longitude': self.longitude
            }
        )

        ds.to_netcdf(filename, engine='h5netcdf')


class ForecastSurfaceData(SurfaceData):
    def __init__(
        self, 
        lead_time: np.ndarray,
        time: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        **variables: np.ndarray,
    ):
        super().__init__(time, latitude, longitude, **variables)
        self.lead_time = lead_time
        
    
    @classmethod
    def load_from_h5(cls, filename, variables_names):
        """Load data from an HDF5 file using h5py for dimensions."""
        with h5py.File(filename, 'r') as f:
            latitude = f['latitude'][:]
            longitude = f['longitude'][:]
            for var_name in variables_names:
                setattr(cls, var_name, f[var_name][...])
        lead_time = xr.open_dataset(filename, engine='h5netcdf').lead_time.values
        time = xr.open_dataset(filename, engine='h5netcdf').time.values

        return cls(
            lead_time, time, latitude, longitude, 
            **{var_name: getattr(cls, var_name) for var_name in variables_names}
        )


    def save_to_h5(self, filename):
        """Save data to an HDF5 file using xarray."""
        data_vars = {
            var_name: (['lead_time', 'time', 'latitude', 'longitude'], getattr(self, var_name))
            for var_name in self._var_names
        }
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'lead_time': self.lead_time,
                'time': self.time,
                'latitude': self.latitude,
                'longitude': self.longitude
            }
        )

        ds.to_netcdf(filename, engine='h5netcdf')
        
        
class ForecastEnsembleSurfaceData(SurfaceData):
    def __init__(
        self, 
        lead_time: np.ndarray,
        number: np.ndarray,
        time: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        **variables: np.ndarray,
    ):
        super().__init__(time, latitude, longitude, **variables)
        self.lead_time = lead_time
        self.number = number


    @classmethod
    def load_from_h5(cls, filename, variables_names):
        """Load data from an HDF5 file using h5py for dimensions."""
        with h5py.File(filename, 'r') as f:
            number = f['number'][:]
            latitude = f['latitude'][:]
            longitude = f['longitude'][:]
            for var_name in variables_names:
                setattr(cls, var_name, f[var_name][...])
        lead_time = xr.open_dataset(filename, engine='h5netcdf').lead_time.values
        time = xr.open_dataset(filename, engine='h5netcdf').time.values

        return cls(
            lead_time, number, time, latitude, longitude, 
            **{var_name: getattr(cls, var_name) for var_name in variables_names}
        )


    def save_to_h5(self, filename):
        """Save data to an HDF5 file using xarray."""
        data_vars = {
            var_name: (['lead_time', 'number', 'time', 'latitude', 'longitude'], getattr(self, var_name))
            for var_name in self._var_names
        }
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'lead_time': self.lead_time,
                'number': self.number,
                'time': self.time,
                'latitude': self.latitude,
                'longitude': self.longitude
            }
        )

        ds.to_netcdf(filename, engine='h5netcdf')
