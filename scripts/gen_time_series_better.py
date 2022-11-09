import h5py
import datetime
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from config import data_dir

def datenum_to_epoch(datenum):
    """
    Convert Matlab datenum into epoch.
    :param datenum: Date in datenum format
    :return:        Epoch time. Seconds from the 1st of January 1970
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

def get_SBE56():
    pres = np.array([1, 8, 23, 28, 33, 43, 53, 63, 78, 96, 108, 126, 151, 176])
    order = ['0218', '5894', '0221', '5895', '0222', '0225', '0226',
         '5897', '5899', '0235', '5900', '5901', '5902', '5903']

    # get all mat files in directory
    mat_files = sorted(Path(data_path).glob('*mat')) 
    
    # order list by preasure
    mat_files = [file for x in order for file in mat_files if x in str(file)]
    
    # find shortest time series for a single thermistor and get index
    max_idx = min(len(thermistor['tem']) for thermistor in raw_data)
    max_idx = n_max if n_max < max_idx else max_idx

    # extract temperature and dates in a 2D numpy array each
    temp = np.vstack([np.squeeze(thermistor['tem'])[n_min:max_idx] for thermistor in raw_data])
    date = np.vstack([np.squeeze(thermistor['dates'])[n_min:max_idx] for thermistor in raw_data])

    # check that all thermistor dates are synced
    if (date[1:, :] ==[:-1, :]).all():
        print('All SBE56 dates are synced. Generating 1d dates array')
        date = date[0, :] # we dont need dates to be a 2d array 

    else:
        print('SBE56 dates are not synced. Generating a 2d dates array with \n
              dates for each thermistor.')
    
    date = list(map(datenum_to_epoch, date))

    return temp, pres, date

def write_file():

    

    
