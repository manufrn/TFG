import h5py
import netCDF4
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pandarallel import pandarallel
from config import data_dir, reports_dir
from scipy.interpolate import interp1d

pandarallel.initialize(verbose=0)

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

def load_time_series_xr(filename):

    file_path = data_dir / 'time_series' / filename
    data = xr.open_dataset(file_path)
    data.coords['date'] = pd.to_datetime(data['date'], unit='s')
    return data

    

def load_SHDR_fit(filename, convert_date=True):
    '''Load  saved SHDR fit as filename in data_dir/SHDR_fit.
    Returns data frame containing fit of time series.
    '''

    file_path = data_dir / 'SHDR_fit' / filename
    skiprows = 0
    with open(file_path, 'r+') as f:
        first_line = f.readline()
        if first_line.startswith('SHDR'):
            skiprows = 14

    df_fit = pd.read_csv(file_path, skiprows=skiprows)

    if convert_date:
        try:
            df_fit['date'] = pd.to_datetime(df_fit['Dates'], unit='s')

        except:
            df_fit['date'] = pd.to_datetime(df_fit['date'], unit='s')
    
    df_fit.set_index('date', inplace=True)

    return df_fit

def load_buoy_series(filename):
    ds = xr.open_dataset(data_dir / 'buoy_time_series' / filename)
    df = ds.to_dataframe()
    df['date'] = df['date'].apply(lambda x: datetime.utcfromtimestamp(x))
    df.set_index('date', inplace=True)
    return df


def get_fit_metadata(filename):
    file_path = data_dir / 'SHDR_fit' / filename
    with open(file_path, 'r+') as f:
        first_line = f.readline()
        if first_line.startswith('SHDR'):
            metadata_list = [first_line]
            for _ in range(13):
                metadata_list.append(next(f))
            metadata_string = ''.join(metadata_list)
            return metadata_string


def fit_function(z, df, date_i):
    '''Return value of idealized fit function for datapoint at loc. loc can be
    integer pointing to position of datapoint or datetime object.
    '''
    
    if isinstance(date_i, int):
        fit = df.iloc[date_i]

    else:
        fit = df.loc[date_i]
    
    D1, b2, c2 = fit['D1'], fit['b2'], fit['c2']
    b3, a2, a1 = fit['b3'], fit['a2'], fit['a1']

    pos = np.where(z >= D1, 1.0, 0.0)  # check if z is above or bellow MLD
    zaux = - (z - D1) * (b2 + (z - D1) * c2)
    f = a1 + pos * (b3 * (z - D1) + a2 *(np.exp(zaux) - 1.0))
    return f

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
        idx = np.searchsorted(dates, date)
        # sign = +1
        # value = 1
        # while True:
        #     dt = sign*value
        #     possible_date = date + timedelta(seconds=dt) 
        #
        #     if date + timedelta(seconds=dt) in dates:
        #         idx =  np.where(dates==possible_date)[0][0]
        #         break
        #     
        #     if sign == -1:
        #         value += 1
        #     sign *= -1

    return idx

def timedelta_to_interval(tdelta, dt=5):
        '''Return interval in rows of df fit for a given timedelta
        '''

        interval = np.int(np.round(tdelta.total_seconds() / dt))
        return interval


def mean_and_std(df_fit, variable):
    variable = df_fit[variable]
    return variable.mean(), variable.std()


def distance(df_fit, variable, n, value):
    '''Given variable of df_fit, return the locs where the diference between
    slices [n:] - [:-n] in that variable are greater than value.
    '''
    array = df_fit[variable].to_numpy()
    locs = np.where(abs(array[n:] - array[:-n]) > value)[0]
    ratio = len(locs)/len(df_fit)
    return locs, ratio
                    

def physical_RMS(df_fit, temp, pres, loc):
    y = if_masked_to_array(temp[loc])
    z = if_masked_to_array(pres[loc])
    fitnes = np.sqrt(np.sum((y - fit_function(z, df_fit, loc))**2) / len(y))
    return fitnes

def n_worst_profiles(df_fit, n):
    em = df_fit['em'].to_numpy()
    indices = np.argpartition(em, -n)[-n:]
    return indices

def interpolate(z, y, z_values, insert=False):
    
    if isinstance(z, np.ma.core.MaskedArray):
        z = np.asarray(z[z.mask==False])
        y = np.asarray(y[y.mask==False])

    if len(z) != len(np.unique(z)):
        idx = np.argmin((z[1:] - z[:-1])) + 1
        z = np.delete(z, idx)
        y = np.delete(y, idx)

    interp = interp1d(z, y, 'cubic')

    y_values = interp(z_values)
    if insert:
        idx = np.searchsorted(z, z_values)
        z = np.insert(z, idx, z_values)
        y = np.insert(y, idx, y_values)

    else:
        z = z_values
        y = y_values
    return z, y

def find_MLD_threshold(temp, pres, threshold=0.2, interpolation=True):
    temp_loc = if_masked_to_array(temp)
    pres_loc = if_masked_to_array(pres)
    
    if interpolation:
        zz = np.linspace(np.min(pres_loc), np.max(pres_loc), 200)
        pres_loc, temp_loc = interpolate(pres_loc, temp_loc, zz)

    dif = temp_loc[0] - temp_loc
    a = np.searchsorted(dif, threshold)

    if a == len(pres_loc):
        MLD = pres_loc[-1]
    else:
        MLD = pres_loc[a]
    return MLD
