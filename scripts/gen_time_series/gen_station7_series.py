import h5py
import netCDF4
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from config import data_dir
from gen_AGL_time_series import datenum_to_epoch

data_path = data_dir / 'raw' / 'oceanographic_station' / 'station7.mat'
output_path = data_dir / 'time_series' / 'station7_more_profiles.nc'

station7_complete = loadmat(data_path)  
# date_slice = np.s_[238:244]
# date_slice = np.s_[180:244]
# pres_slice = np.s_[:450]

masked_temp = np.ma.masked_invalid(np.squeeze(station7_complete['tems'])).T
masked_temp = masked_temp[date_slice, pres_slice]

date_series = datenum_to_epoch(np.squeeze(station7_complete['dates'])[date_slice])

pres = np.squeeze(station7_complete['pres'])[pres_slice]
pres_2d = np.vstack([pres for _ in range(len(date_series))])
masked_pres = np.ma.masked_where(masked_temp.mask, pres_2d)
masked_pres = masked_pres.astype('int32')

latitude = np.squeeze(station7_complete['lat'])
longitude = np.squeeze(station7_complete['lon'])

dim_time = len(date_series)
dim_pres = masked_pres.shape[1]
with netCDF4.Dataset(output_path, mode='w', format='NETCDF4') as ds:
    # ds.description = '''Station 7 series from 2018-10-2 12:1117h to
    #     2019-4-8 15:57:11)'''
    
    ds.title = 'Station 7 CTD temperature'

    # dimensions
    ds.createDimension('time', dim_time)
    ds.createDimension('pres', dim_pres)
    ds.createDimension('lat', 1)
    ds.createDimension('lon', 1)

    # variables
    lat = ds.createVariable('lat', 'f4', ('lat',))
    lon = ds.createVariable('lon', 'f4', ('lon',))
    temp = ds.createVariable('temp', 'f8', ('time', 'pres',))
    date = ds.createVariable('date', 'i4', ('time', ))
    pres = ds.createVariable('pres', 'i4', ('time', 'pres',))

    # asign data
    lat[:] = latitude
    lon[:] = longitude
    temp[:, :] = masked_temp
    date[:] = date_series
    pres[:, :] = masked_pres

print(f'Completed. Time series saved in {output_path}')


