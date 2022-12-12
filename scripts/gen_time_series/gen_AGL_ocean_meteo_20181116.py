import sys
import numpy as np
import netCDF4
from datetime import datetime, timedelta
from scipy.io import loadmat
sys.path.append('/home/manu/TFG_repo/scripts')
from analysis_routines import * 
from processing_routines import datenum_to_epoch


output_path = data_dir / 'buoy_time_series' / 'AGL_ocean_meteo_20181181.nc'
print(output_path)
print(output_path.is_file())
raw_data_path = data_dir / 'raw/AGL_buoy/AGL_ocean_meteo.mat'
temp_chain, pres_chain, date_chain = load_time_series('processed/AGL_20181116_chain.nc')

mat_data = loadmat(raw_data_path)
del mat_data['__header__']
del mat_data['__globals__']
del mat_data['__version__']

date_epoch = datenum_to_epoch(np.squeeze(mat_data['datet']))
date_datetime = np.array([datetime.utcfromtimestamp(i) for i in date_epoch])

loc_0 = np.where(date_datetime == date_chain[0])[0][0]
loc_f = np.where(date_datetime == date_chain[-1])[0][0]
data = dict({i: np.squeeze(mat_data[i])[loc_0:loc_f+1] for i in mat_data})
date_datetime = date_datetime[loc_0:loc_f+1]
date_epoch = date_epoch[loc_0:loc_f+1]

dim_time = len(data['sea_tem'])

with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as ds:
    # dimensions
    ds.createDimension('time', dim_time)
    
    #variables
    date = ds.createVariable('date', 'i4', ('time',))
    lat = ds.createVariable('lat', 'f4', ('time',))
    lon = ds.createVariable('lon', 'f4', ('time',))
    sea_tem = ds.createVariable('sea_tem', 'f4', ('time',))
    salt = ds.createVariable('salt', 'f8', ('time',))
    wind_dir = ds.createVariable('wind_dir', 'f8', ('time',))
    wind_speed = ds.createVariable('wind_speed', 'f8', ('time',))
    air_pr = ds.createVariable('air_pr', 'f8', ('time',))
    air_tem = ds.createVariable('air_tem', 'f8', ('time',))
    hum = ds.createVariable('hum', 'f8', ('time',))
    
    # asign data
    date[:] = date_epoch
    lat[:] = data['lat']
    lon[:] = data['lon']
    sea_tem[:] = data['sea_tem']
    salt[:] = data['salt']
    wind_dir[:] = data['wind_dir']
    wind_speed[:] = data['wind_speed']
    air_pr[:] = data['air_pr']
    air_tem[:] = data['air_tem']
    hum[:] = data['hum']
    
