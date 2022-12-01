import h5py
import netCDF4
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import data_dir, reports_dir

def load_time_series(filename, latlon=False, convert_date=True):
    '''Load time series saved as filename in data_dir/time_series
    '''

    file_path = data_dir / 'time_series' / filename

    if filename.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            temp = f['temperature'][:]
            depth = f['pressure'][:]
            date = f['date'][:]
            lat = f['lat'][:]
            lon = f['lon'][:]

    elif filename.endswith('.nc'):
        with netCDF4.Dataset(file_path, 'r') as ds:
            temp = ds.variables['temp'][:]
            try:
                depth = ds.variables['pres'][:]
            except:
                depth = ds.variables['depth'][:]
            date = ds.variables['date'][:]
            lat = ds.variables['lat'][:]
            lon = ds.variables['lon'][:]
    else:
        raise Exception('Only .h5 and .nc time series supported')

    if convert_date:
        date = np.array([datetime.utcfromtimestamp(i) for i in date])
    
    if latlon:
        return temp, depth, date, lat, lon

    else:
        return temp, depth, date


def load_SHDR_fit(filename):
    '''Load  saved SHDR fit as filename in data_dir/SHDR_fit.
    Returns data frame containing fit of time series.
    '''

    file_path = data_dir / 'SHDR_fit' / filename
    df_fit = pd.read_csv(file_path)

    try:
        df_fit['Dates'] = df_fit['Dates'].apply(lambda x: datetime.utcfromtimestamp(x))

    except:
        df_fit['Dates'] = df_fit['date'].apply(lambda x: datetime.utcfromtimestamp(x))

    return df_fit


def fit_function(z, df, loc):
    '''Return value of idealized fit function for datapoint at loc. loc can be
    integer pointing to position of datapoint or datetime object.
    '''
    
    # check if inputed loc is datetime object and convert it to iloc
    if isinstance(loc, datetime):
        loc = date_to_idx(df['Dates'], loc)

    
    fit = df.iloc[loc]
    
    D1, b2, c2 = fit['D1'], fit['b2'], fit['c2']
    b3, a2, a1 = fit['b3'], fit['a2'], fit['a1']

    # print('D1: {:.2f}, a1: {:.2f}, b3: {:.2E}, b2:{:.2E}, c2: {:.2E}'.format(D1, a1, b3, b2, c2))

    pos = np.where(z >= D1, 1.0, 0.0)  # check if z is above or bellow MLD
    zaux = - (z - D1) * (b2 + (z - D1) * c2)
    return a1 + pos * (b3 * (z - D1) + a2 *(np.exp(zaux) - 1.0))


def if_masked_to_array(array):
    '''CHech if given array is maked and return a standard ndarray version
    of it whithout the false values of the mask'''

    if isinstance(array, np.ma.core.MaskedArray):
        return np.asarray(array[array.mask == False])

    else:
        return array


def date_to_idx(dates, date):
    ''' Use fit dataframe to convert a given date to the closest iloc. 
    '''
    
    try:
        idx = np.where(dates == date)[0][0]

    except:
        sign = +1
        value = 1
        while True:
            dt = sign*value
            possible_date = date + timedelta(seconds=dt) 

            if date + timedelta(seconds=dt) in dates:
                idx =  np.where(dates==possible_date)[0][0]
                break
            
            if sign == -1:
                value += 1
            sign *= -1

    return idx

def timedelta_to_interval(timedelta, dt=5):
        '''Return interval in rows of df fit for a given timedelta
        '''

        interval = np.int(np.round(timedelta.total_seconds() / dt))
        return interval


def mean_and_std(df_fit, variable):
    variable = df_fit[variable]
    return variable.mean(), variable.std()
