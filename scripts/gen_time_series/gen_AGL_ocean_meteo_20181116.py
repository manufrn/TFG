import sys
import numpy as np
import netCDF4
from datetime import datetime, timedelta
from scipy.io import loadmat
sys.path.append('/home/manu/TFG_repo/scripts')
from analysis_routines import * 
from processing_routines import datenum_to_epoch


output_path = data_dir / 'buoy_time_series' / 'AGL_ocean_meteo_20181116.nc'
raw_data_path = data_dir / 'raw/AGL_buoy/AGL_ocean_meteo.mat'
raw_fluxes_path = data_dir / 'raw/AGL_buoy/air_sea_fluxes.mat'
# temp_chain, pres_chain, date_chain = load_time_series('processed/AGL_20181116_chain.nc')
chain_data = load_time_series_xr('processed/AGL_20181116_chain_xrcompatible.nc')
date_chain = chain_data.date.data

mat_data = loadmat(raw_data_path)
del mat_data['__header__']
del mat_data['__globals__']
del mat_data['__version__']

mat_fluxes = loadmat(raw_fluxes_path)
del mat_fluxes['__header__']
del mat_fluxes['__globals__']
del mat_fluxes['__version__']

date_epoch = datenum_to_epoch(np.squeeze(mat_data['datet']))
date_datetime = np.array(date_epoch, dtype='datetime64[s]')

loc_0 = np.where(date_datetime == date_chain[0])[0][0]
loc_f = np.where(date_datetime == date_chain[-1])[0][0]

data = dict({i: np.squeeze(mat_data[i])[loc_0:loc_f+1] for i in mat_data})
fluxes = dict({i: np.squeeze(mat_fluxes[i])[loc_0:loc_f+1] for i in mat_fluxes})

date_datetime = date_datetime[loc_0:loc_f+1]
date_epoch = date_epoch[loc_0:loc_f+1]

dim_date = len(data['sea_tem'])

with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as ds:
    # dimensions
    ds.createDimension('date', dim_date)
    
    #variables
    date = ds.createVariable('date', 'i4', ('date',))
    lat = ds.createVariable('lat', 'f4', ('date',))
    lon = ds.createVariable('lon', 'f4', ('date',))
    sea_tem = ds.createVariable('sea_tem', 'f4', ('date',))
    salt = ds.createVariable('salt', 'f8', ('date',))
    wind_dir = ds.createVariable('wind_dir', 'f8', ('date',))
    wind_speed = ds.createVariable('wind_speed', 'f8', ('date',))
    air_pr = ds.createVariable('air_pr', 'f8', ('date',))
    air_tem = ds.createVariable('air_tem', 'f8', ('date',))
    hum = ds.createVariable('hum', 'f8', ('date',))
    Qe = ds.createVariable('Qe', 'f8', ('date',))
    Qh = ds.createVariable('Qh', 'f8', ('date',))
    stress = ds.createVariable('stress', 'f8', ('date',))
    tx = ds.createVariable('tx', 'f8', ('date', ))
    ty = ds.createVariable('ty', 'f8', ('date', ))

    
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
    Qe[:] = fluxes['Qe']
    Qh[:] = fluxes['Qh']
    stress[:] = fluxes['stress']
    tx[:] = fluxes['tx']
    ty[:] = fluxes['ty']


    
