import h5py
import netCDF4
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from config import data_dir

    
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

    pres = np.array([1, 8, 23, 28, 33, 43, 53, 63, 78, 96, 108, 126, 
                           151, 176])
    order = ['0218', '5894', '0221', '5895', '0222', '0225', '0226',
                   '5897', '5899', '0235', '5900', '5901', '5902', '5903']

    # get all mat files in directory
    mat_files = sorted(Path(path).glob('*mat')) 
    
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

    return temp, pres, date


if __name__ == '__main__':

    filename = data_dir / 'time_series' / 'processed' / 'AGL_1_37_56_cropped.nc'
    latittude, longitude = 43.789, 3.782 # latitude and longitude of AGL buoy
    n_max = 2471041 # to crop data before thermistors are taken out of the water

    path_SBE56 = data_dir / 'raw' / 'thermistor_chain' / 'AGL_1' / 'SBE56'
    path_SBE37 = data_dir / 'raw' / 'thermistor_chain' / 'AGL_1' / 'SBE37'
    fn_up_SBE37 = 'SBE37567420190409.mat' # upper SBE37 (around 50 m)
    fn_low_SBE37 = 'SBE37346520190409.mat'

    ### SBE56 ###
    temp_SBE56, pres_SBE56, date_SBE56 = get_SBE56(path_SBE56, n_max = n_max)
    
    ### SBE37 ###
    low_SBE37 = loadmat(path_SBE37 / fn_low_SBE37)
    up_SBE37 = loadmat(path_SBE37 / fn_up_SBE37)

    temp_up_37, date_up_37 = np.squeeze(up_SBE37['tem']), np.squeeze(up_SBE37['dates'])
    temp_low_37, date_low_37 = np.squeeze(low_SBE37['tem']), np.squeeze(low_SBE37['dates'])
    date_up_37 = datenum_to_epoch(date_up_37)
    date_low_37 = datenum_to_epoch(date_low_37)

    press_low_37, press_up_37 = np.squeeze(low_SBE37['pre']), np.squeeze(up_SBE37['pre'])
    
    # dates of SBE56 are out of sync by a year. Subtract this year
    date_SBE56 -= int(timedelta(days=365).total_seconds())

    # move SBE56 dates to multiples of 5 seconds after meassure 122
    date_SBE56[123:] += 1

    # crop SBE37 to extent of SBE56
    last_up = np.nonzero(np.in1d(date_SBE56, date_up_37))[0][-1]
    last_low = np.nonzero(np.in1d(date_SBE56, date_low_37))[0][-1]
    temp_up_37 = temp_up_37[:np.where(date_up_37 == date_SBE56[last_up])[0][0] + 1]
    temp_low_37 = temp_low_37[:np.where(date_low_37 == date_SBE56[last_low])[0][0] + 1]


    ### GENERATE MASKED ARRAYS ###
    # temperature array
    masked_temp = np.ma.masked_all_like(np.zeros((len(date_SBE56), 16)))
    up_slice = np.s_[:last_up+1:120, 6] # 6th column of array, values up till tast up every 120 points
    low_slice = np.s_[:last_low+1:60, -1] # last column of array, values every 60 points

    masked_temp[:, 0:6] = temp_SBE56[:, 0:6]
    masked_temp[up_slice] = temp_up_37    
    masked_temp[:, 7:-1] = temp_SBE56[:, 6:]
    masked_temp[low_slice] = temp_low_37
    
    print('Cada 120 elementos (10 min) de masked temp, hay 16 medidas?:', 
          (masked_temp[::120].count(axis=1)==16).all())
    print('Cada 60 elementos (5 min) de masked temp, hay 15 medidas o más?:', 
          (masked_temp[::60].count(axis=1) >= 15).all())

    # preassure array
    masked_pres = np.ma.masked_all_like(masked_temp)
    masked_pres[:, 0:6] = pres_SBE56[None, :6]
    masked_pres[up_slice] = press_up_37[:np.where(date_up_37 == date_SBE56[last_up])[0][0] +1]
    masked_pres[:, 7:-1] = pres_SBE56[None, 6:]
    masked_pres[low_slice] = press_low_37[:np.where(date_low_37 == date_SBE56[last_low])[0][0] +1]

    dim_pres = masked_pres.shape[1]
    dim_time = len(date_SBE56)

    ### SAVE SERIES AS netCDF4 ###
    with netCDF4.Dataset(filename, mode='w', format='NETCDF4') as ds:
        ds.description = 'Time series of AGL buy thermistor chain from 2018-11-16 11:00:00 to 2019-04-08 11:00:00'
        ds.title = 'AGL_1 thermistor chain series'
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
        pres = ds.createVariable('pres', 'i4', ('time', 'pres', ))
        
        # asign data
        lat[:] = latittude
        lon[:] = longitude
        temp[:, :] = masked_temp
        date[:] = date_SBE56
        pres[:, :] = masked_pres

    print(f'Completed. Time series saved in {filename}')
