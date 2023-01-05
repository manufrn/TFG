import h5py
import netCDF4
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from config import data_dir
from gen_AGL_time_series import datenum_to_epoch

data_path = data_dir / 'raw' / 'oceanographic_station' / 'station7.mat'
output_path = data_dir / 'time_series' / 'station7_complete.nc'

station7_complete = loadmat(data_path)  

temp_ = np.squeeze(station7_complete['tems']).T
pres_ = np.squeeze(station7_complete['pres'])
date_ = datenum_to_epoch(np.squeeze(station7_complete['dates']))
sal_ = np.squeeze(station7_complete['sals']).T
latitude_ = np.squeeze(station7_complete['lat'])
longitude_ = np.squeeze(station7_complete['lon'])

print(np.shape(temp_))

dim_date = len(date_)
dim_pres = len(pres_)
with netCDF4.Dataset(output_path, mode='w', format='NETCDF4') as ds:
    # ds.description = '''Station 7 series from 2018-10-2 12:1117h to
    #     2019-4-8 15:57:11)'''
    
    ds.title = 'Station 7 CTD temperature'

    # dimensions
    ds.createDimension('date', dim_date)
    ds.createDimension('pres', dim_pres)

    # variables
    lat = ds.createVariable('lat', 'f4', ('date',))
    lon = ds.createVariable('lon', 'f4', ('date',))
    temp = ds.createVariable('temp', 'f8', ('date', 'pres',))
    date = ds.createVariable('date', 'i4', ('date', ))
    pres = ds.createVariable('pres', 'i4', ('pres',))
    sal = ds.createVariable('sal', 'f8', ('date', 'pres'))

    # asign data
    lat[:] = latitude_
    lon[:] = longitude_
    temp[:, :] = temp_
    date[:] = date_
    pres[:] = pres_ 

print(f'Completed. Time series saved in {output_path}')
#
#
