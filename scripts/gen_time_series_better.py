import h5py
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from config import data_dir

    
@np.vectorize
def datenum_to_epoch(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    datetime_format = (datetime.fromordinal(int(datenum)) \
           + timedelta(days=days) \
           - timedelta(days=366))
    date64 = np.datetime64(datetime_format)
    posix_time = (date64 - np.datetime64('1970-01-01T00:00:00')) / (np.timedelta64(1, 's'))
    return round(posix_time)


def get_SBE(path, pres, order, n_min = 0, n_max = None):
    # get all mat files in directory
    mat_files = sorted(Path(path).glob('*mat')) 
    
    # order list by preasure
    mat_files = [file for x in order for file in mat_files if x in str(file)]
    raw_data = [loadmat(file) for file in mat_files]
    
    # find shortest time series for a single thermistor and get index
    max_idx = min(len(thermistor['tem']) for thermistor in raw_data)
    longest = max(len(thermistor['tem']) for thermistor in raw_data)
    print(max_idx, longest)
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
    
    # date = list(map(datenum_to_epoch, date))
    datenum_to_epoch_vec = np.vectorize(datenum_to_epoch)
    date = datenum_to_epoch_vec(date)

    return temp, pres, date


if __name__ == '__main__':

    # SBE56
    path_SBE56 = data_dir / 'raw' / 'thermistor_chain' / 'AGL_1' / 'SBE56'
    pres_SBE56 = np.array([1, 8, 23, 28, 33, 43, 53, 63, 78, 96, 108, 126, 
                           151, 176])
    order_SBE56 = ['0218', '5894', '0221', '5895', '0222', '0225', '0226',
                   '5897', '5899', '0235', '5900', '5901', '5902', '5903']
    
    

    temp_SBE56, pres_SBE56, date_SBE56 = get_SBE(path_SBE56, pres_SBE56, 
                                                 order_SBE56)

   
    date_SBE56 = np.asarray(date_SBE56)
    np.save(data_dir / 'SBE56', date_SBE56)
    # SBE37
    
    path_SBE37 = data_dir / 'raw' / 'thermistor_chain' / 'AGL_1' / 'SBE37'
    pres_SBE37 = np.array([51, 215])
    order_SBE37 = ['5674', '3465']

    temp_SBE37, press_SBE37, date_SBE37 = get_SBE(path_SBE37, pres_SBE37,
                                               order_SBE37)
    
    date_SBE37 = np.asarray(date_SBE37)
    np.save(data_dir / 'SBE37', date_SBE37)

    
