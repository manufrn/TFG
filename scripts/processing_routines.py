import h5py
import netCDF4
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from scipy.io import loadmat

@np.vectorize
def datenum_to_epoch(datenum):
    """Convert Matlab datenum into posix time.
    """

    days = datenum % 1
    datetime_format = (datetime.fromordinal(int(datenum)) \
           + timedelta(days=days) \
           - timedelta(days=366))
    date64 = np.datetime64(datetime_format)
    posix_time = ((date64 - np.datetime64('1970-01-01T00:00:00')) 
                  / (np.timedelta64(1, 's')))
    return round(posix_time)


def get_SBE56(path, n_min = 0, n_max = None):

    depth = np.array([1, 8, 23, 28, 33, 43, 53, 63, 78, 96, 108, 126, 
                           151, 176])
    order = ['0218', '5894', '0221', '5895', '0222', '0225', '0226',
                   '5897', '5899', '0235', '5900', '5901', '5902', '5903']

    # get all mat files in directory
    mat_files = sorted(Path(path).glob('*mat')) 
    print(mat_files)
    
    # order list by preasure
    mat_files = [file for x in order for file in mat_files if x in str(file)]
    raw_data = [loadmat(file) for file in mat_files]
    
    # find shortest time series for a single thermistor and get index
    # not really necesarry if n_max is specified
    max_idx = min(len(thermistor['tem']) for thermistor in raw_data)
    longest = max(len(thermistor['tem']) for thermistor in raw_data)

    if n_max != None:
        max_idx = n_max if n_max < max_idx else max_idx

    # extract temperature and dates in a 2D numpy array each
    temp = np.vstack([np.squeeze(thermistor['tem'])[n_min:max_idx] for thermistor in raw_data]).T
    date = np.vstack([np.squeeze(thermistor['dates'])[n_min:max_idx] for thermistor in raw_data]).T

    # check that all thermistor dates are synced
    if (date[:, 1:] == date[:, :-1]).all():
        print('All SBE56 dates are synced. Generating 1d dates array')
        date = date[:, 0] # we dont need dates to be a 2d array 

    else:
        print('SBE56 dates are not synced. Generating a 2d dates array with '
              'dates for each thermistor.')
    
    date = datenum_to_epoch(date)

    return temp, depth, date
