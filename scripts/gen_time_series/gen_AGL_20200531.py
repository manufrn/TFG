import sys
import netCDF4
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime, timedelta
sys.path.append('/home/manu/TFG_repo/scripts')
from processing_routines import datenum_to_epoch
from analysis_routines import date_to_idx
from config import data_dir

output_path = data_dir / 'time_series' / 'unprocessed' / 'AGL_20200531_chain.nc'

data_path_56 = data_dir / 'raw' / 'thermistor_chain' / 'AGL_Octubre_2020' / 'SBE56' / 'Data_May_Oct_2020_SBE56.mat'
low_37_path = data_dir / 'raw' / 'thermistor_chain' / 'AGL_Octubre_2020' / 'SBE37_3465' / 'SBE37SM-RS232_3465_2020_10_10.asc'
up_37_path = data_dir / 'raw' / 'thermistor_chain' / 'AGL_Octubre_2020' / 'SBE37_5674' / 'SBE37567420201010.mat'


data_56 = loadmat(data_path_56)

del data_56['__header__']
del data_56['__version__']
del data_56['__globals__']
del data_56['cmap']
del data_56['cc']

min_length = min([len(data_56[i]) for i in data_56])
n_depths = len(data_56)

date = np.zeros((min_length, n_depths))
temp = np.zeros_like(date)
depths = np.array([1, 8, 23, 28, 33, 38, 43, 53, 63, 78, 93, 108, 126, 151, 176])
for i in data_56:
    depth_str = i.split('_')[1]
    if 'm' in depth_str:
        depth_str = depth_str[0]
    depth = int(depth_str)
    idx = np.argmax(depths==depth)
    
    temp_date_i = data_56[i].T
    date[:, idx] = temp_date_i[0][:min_length]
    temp[:, idx] = temp_date_i[1][:min_length]
    
real_depths = depths
real_depths[2:] += 9

if (date[:, 1:]==date[:, :-1]).all():
    date = datenum_to_epoch(date[:, 0])

date[122:] += 1
date_datetime = np.array([datetime.utcfromtimestamp(i) for i in date])

date_start = datetime(2020, 5, 31, 18)
idx_start = date_to_idx(date_datetime, date_start)
date_crop = datetime(2020, 10, 2, 18)
idx_crop = date_to_idx(date_datetime, date_crop) + 1

temp = temp[idx_start:idx_crop]
date = date[idx_start:idx_crop]
date_datetime = date_datetime[idx_start:idx_crop]

up_37 = loadmat(up_37_path)
up_37_epoch = datenum_to_epoch(np.squeeze(up_37['dates']))[6:]
up_37_datetime = np.array([datetime.utcfromtimestamp(i) for i in up_37_epoch])
up_37_pres = np.squeeze(up_37['pre'])[6:]
up_37_temp = np.squeeze(up_37['tem'])[6:]
# we take values starting from 6: to not take lowering of thermistor period and
# to start at 18:00 like SBE56

low_37 = pd.read_csv(low_37_path, names=['temp', 'cond', 'pres', 'date', 'time'], skiprows=58)
low_37['datetime'] = low_37['date'] + low_37['time']
low_37['datetime'] = low_37['datetime'].apply(lambda x: datetime.strptime(x[1:], '%d %b %Y %H:%M:%S'))
low_37 = low_37.drop(['date', 'time'], axis=1)
low_37_datetime = low_37['datetime'].to_numpy(dtype=datetime)[12:]
low_37_pres = low_37['pres'].to_numpy()[12:]
low_37_temp = low_37['temp'].to_numpy()[12:]

@np.vectorize
def remove_seconds(date):
    if date.second == 1:
        return date - timedelta(seconds=1)
    else:
        return date

low_37_datetime = remove_seconds(low_37_datetime)

last_up = np.where(up_37_datetime == date_crop)[0][0]
last_low = np.where(low_37_datetime == date_crop)[0][0]

# array de temperaturas
masked_temp = np.ma.masked_all_like(np.zeros((len(date_datetime), 17)))
masked_temp[:, 0:7] = temp[:, :7]
masked_temp[::120, 7] = up_37_temp[:last_up + 1]
masked_temp[::60, -1] = low_37_temp[:last_low + 1]
masked_temp[:, 8:-1] = temp[:, 7:]

# array de presiones
masked_pres = np.ma.masked_all_like(np.zeros((len(date_datetime), 17)))
masked_pres[:, 0:7] = depths[None, :7]
masked_pres[::120, 7] = up_37_pres[:last_up + 1]
masked_pres[::60, -1] = low_37_pres[:last_low + 1]
masked_pres[:, 8:-1] = depths[None, 7:]

print('Cada 120 elementos (10 min) de masked temp, hay 16 medidas:', (masked_temp[::120].count(axis=1)==17).all())
print('Cada 60 elementos (5 min) de masked temp, hay 15 medidas o más:', (masked_temp[::60].count(axis=1) >= 16).all())


dim_depth = masked_pres.shape[1]
dim_time = len(date_datetime)
latitude, longitude = 43.789, 3.782 # latitude and longitude of AGL buoy

_date = date # to mess it up with date in .nc dataset
del date


with netCDF4.Dataset(output_path, mode='w', format='NETCDF4') as ds:
    ds.description = 'Time series of AGL buy thermistor chain from 2018-11-16 11:00:00 to 2019-04-08 11:00:00'
    ds.time_coverage_start = '2018-11-16T11:00:00Z'
    ds.time_coverage_end = '2019-04-08T13:00:00Z'
    ds.title = 'AGL_1 thermistor chain series'

    # dimensions
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
    temp[:, :] = masked_temp
    date[:] = _date
    depth[:, :] = masked_pres
