from scipy.io import loadmat
import datetime
from pathlib import Path
import numpy as np
import h5py

n = 300000 # max number of meassures in time series. for debugging purposes

# paths
data_path = '../data/raw/thermistor_chain/AGL_1/SBE56'
output_path = '../data/time_series/'
output_fn = 'AGL_1_SB56_300000.h5'


lat, lon = 43.789, 3.782 # latitude and longitude of AGL buoy

# define depths of thermistors and asign correct order to their identifiers
pres = np.array([1, 8, 23, 28, 33, 43, 53, 63, 78, 96, 108, 126, 151, 176])
order = ['0218', '5894', '0221', '5895', '0222', '0225', '0226',
         '5897', '5899', '0235', '5900', '5901', '5902', '5903']

def datenum_to_epoch(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
   
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    datetime_format = datetime.datetime.fromordinal(int(datenum)) \
           + datetime.timedelta(days=int(days)) \
           + datetime.timedelta(hours=int(hours)) \
           + datetime.timedelta(minutes=int(minutes)) \
           + datetime.timedelta(seconds=round(seconds)) \
           - datetime.timedelta(days=366)

    return datetime_format.timestamp()

# get and load all .mat files for given thermistors (order) in a list
mat_files = sorted(Path(data_path).glob('*.mat'))
mat_files = [file for x in order for file in mat_files if x in str(file)]
raw_data = [loadmat(file) for file in mat_files]

# find shortest time series for a single thermistor and get index
max_idx = min(len(thermistor['tem']) for thermistor in raw_data)
max_idx = n if n < max_idx else max_idx

# extract temperature and dates in a 2D numpy array each
temp = np.vstack([np.squeeze(thermistor['tem'])[:max_idx] for thermistor in raw_data])
date = np.vstack([np.squeeze(thermistor['dates'])[:max_idx] for thermistor in raw_data])

# check that all thermistor dates are synced
if (date[1:, :] == date[:-1, :]).all():
    date = date[0, :] # we dont need dates to be a 2d array 

date = list(map(datenum_to_epoch, date))

file = Path(output_path) / Path(output_fn)
if file.is_file():
    print('Output file already exists. Overwriting.')

with h5py.File(file, 'w') as f:
    temperature = f.create_dataset('temperature', data=temp, dtype='float64')
    pressure = f.create_dataset('pressure', data=pres, dtype='float64')
    date = f.create_dataset('date', data=date, dtype='int')
    lat = f.create_dataset('lat', data=lat, dtype='float64')
    lon = f.create_dataset('lon', data=lon, dtype='float64')

print(f"{file} generated succesfully")


with h5py.File(file, 'r') as f:
    date = np.array(f['date'])
    for i in date:
        print(i)
        print(datetime.datetime.fromtimestamp(i))

