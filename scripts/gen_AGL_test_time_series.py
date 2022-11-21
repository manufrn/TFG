import h5py
import netCDF4
import numpy as np
from pathlib import Path
from config import data_dir

original_ts = data_dir / 'time_series' / 'processed' / 'AGL_20181116_chain.nc'

output_ts = data_dir / 'time_series' / 'test' / 'test_1.nc'

slice = np.s_[349200:362052]

with netCDF4.Dataset(original_ts, 'r') as ds:
    latitude = ds.variables['lat'][:]
    longitude = ds.variables['lon'][:]
    depth = ds.variables['depth'][:]
    temp = ds.variables['temp'][:]
    date = ds.variables['date'][:]

_depth = depth[slice]
_temp = temp[slice]
_date = date[slice]
dim_time = len(_date)
dim_depth = _depth.shape[1]

with netCDF4.Dataset(output_ts, 'w') as ds:
        ds.createDimension('time', dim_time)
        ds.createDimension('depth', dim_depth)
        ds.createDimension('lat', 1)
        ds.createDimension('lon', 1)
        
        # variables
        lat = ds.createVariable('lat', 'f4', ('lat',))
        lon = ds.createVariable('lon', 'f4', ('lon',))
        temp = ds.createVariable('temp', 'f8', ('time', 'depth',))
        date = ds.createVariable('date', 'i4', ('time', ))
        depth = ds.createVariable('depth', 'i4', ('time', 'depth',))
        
        # asign data
        lat[:] = latitude
        lon[:] = longitude
        temp[:, :] = _temp
        date[:] = _date
        depth[:, :] = _depth

print(f'test time series saved to {output_ts}')

