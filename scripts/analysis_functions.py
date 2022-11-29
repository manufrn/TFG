import h5py
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from config import data_dir, reports_dir

def load_time_series(filename, latlon= False, convert_date=True):
    '''Load time series saved as filename in data_dir/time_series
    '''

    file_path = data_dir / 'time_series' / filename

    if filename.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            temp = np.array(f['temperature'])
            depth = np.array(f['pressure'])
            date = np.array(f['date'])
            lat = np.array(f['lat'])
            lon = np.array(f['lon'])

    if filename.endswith('.nc'):
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

def if_masked_to_array(array):
    '''CHech if given array is maked and return a standard ndarray version
    of it whithout the false values of the mask'''

    if isinstance(array, np.ma.core.MaskedArray):
        return np.asarray(array[array.mask == False])
    else:
        return array
    
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
     


# def date_to_iloc(dates, date):
#     ''' Use fit dataframe to convert a given date to the closest iloc. 
#     '''
#     
#     try:
#         iloc = np.where(dates == date)[0][0]
#
#     except:
#         possible_dates = [date + timedelta(seconds=dt) for dt in (-2, -1, 1, 2)]
#         iloc = np.in1d(dates, possible_dates).argmax()
    # return iloc

def date_to_iloc(dates, date):
    ''' Use fit dataframe to convert a given date to the closest iloc. 
    '''
    
    try:
        iloc = np.where(dates == date)[0][0]
    except:
        sign = +1
        value = 1
        while True:
            dt = sign*value
            possible_date = date + timedelta(seconds=dt) 
            if date + timedelta(seconds=dt) in dates:
                iloc =  np.where(dates==possible_date)[0][0]
            
            if sign == -1:
                value += 1
                
            sign *= -1

    return iloc

def timedelta_to_interval(timedelta, dt=5):
        '''Return interval in rows of df fit for a given timedelta
        '''

        interval = np.int(np.round(timedelta.total_seconds() / dt))
        return interval


def mean_and_std(df_fit, variable):
    variable = df_fit[variable]
    return variable.mean(), variable.std()


def fit_function(z, df, loc):
    '''Return value of idealized fit function for datapoint at loc. loc can be
    integer pointing to position of datapoint or datetime object.
    '''
    
    # check if inputed loc is datetime object and convert it to iloc
    if isinstance(loc, datetime):
        loc = date_to_iloc(df['Dates'], loc)

    
    fit = df.iloc[loc]
    
    D1, b2, c2 = fit['D1'], fit['b2'], fit['c2']
    b3, a2, a1 = fit['b3'], fit['a2'], fit['a1']

    # print('D1: {:.2f}, a1: {:.2f}, b3: {:.2E}, b2:{:.2E}, c2: {:.2E}'.format(D1, a1, b3, b2, c2))

    pos = np.where(z >= D1, 1.0, 0.0)  # check if z is above or bellow MLD
    zaux = - (z - D1) * (b2 + (z - D1) * c2)
    return a1 + pos * (b3 * (z - D1) + a2 *(np.exp(zaux) - 1.0))


def plot_fit_variable(df, variable, lims = [None, None], interval=None, plot=True):
    '''Plot given fit variable (e.g D1, a1, ...) for time between lims and 
    with given interval.
    '''
    
    if isinstance(interval, timedelta):
        interval = timedelta_to_interval(df['Dates'], interval)

    if isinstance(lims[0], datetime):
        lims = [date_to_iloc(df['Dates'], lim) for lim in lims]

    dic = {'D1': 'MLD (m)', 'a1': 'SST (ºC)'}
    var = df[variable][lims[0]:lims[1]:interval]
    dates = df['Dates'][lims[0]:lims[1]:interval]

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)

    fig, ax = plt.subplots()
    ax.scatter(dates, var, s=8)
    ax.set_xlabel('Date')
    if variable in dic:
        ax.set_ylabel(dic[variable])
    else:
        ax.set_ylabel(variable)

    if variable == 'D1':
        ax.set_ylim((max(var) + 4, min(var) - 4))

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    if plot:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax


def plot_profile_fit(df, temp, depth, loc):
    '''Plot measured vertical profile and fit for measure at loc
    '''

    if isinstance(loc, datetime):
        loc = date_to_iloc(df['Dates'], loc)
        print(loc)

    zz = np.linspace(0, depth[-1] + 5, 300)

    fig, ax = plt.subplots(figsize=(4, 4.6875))
    ax.scatter(temp[loc], depth[loc], marker='o', fc='None', ec='tab:red', s=22)
    ax.axhline(df.iloc[loc, 3], c='grey', ls='--') # plot MLD
    ax.set_ylim(depth[-1] + 10, 0)
    ax.set_xlim(9.5, 18)
    ax.plot(fit_function(zz, df, loc), zz)
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (mb)')
    ax.set_title(df['Dates'].iloc[loc])
    fig.tight_layout()
    plt.show()


def fitness(df, temp, depth, loc):
    ''' Get RMS for profile in loc at height z 
    ''' 

    if isinstance(loc, datetime):
        loc = date_to_iloc(df['Dates'], loc)

    temp_loc = temp[loc]
    fitness = np.sqrt(np.sum((temp_loc - fit_function(depth, df, loc))**2) / len(temp_loc))
    return fitness
    

def plot_RMS_fit(df, temp, depth, loc):
    ''' Plot experimental profile and fit with diference between fit and profile,
    and square of that difference.
    '''

    if isinstance(loc, datetime):
        loc = date_to_iloc(df['Dates'], loc)
    
    delta = temp[:, loc] - fit_function(depth, df, loc)
    zz = np.linspace(0, depth[-1] + 5, 300)
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (7.5, 3.75))
    
    ax1.scatter(temp[:, loc], depth, marker='o', fc='None', ec='tab:red', s=22)
    ax1.axhline(df.iloc[loc, 3], c='grey', ls='--') # plot MLD
    ax1.set_ylim(depth[-1] + 10, 0)
    ax1.set_xlim(9.5, 18)
    ax1.plot(fit_function(zz, df, loc), zz)
    ax1.set_xlabel('Temperature (ºC)')
    ax1.set_ylabel('Depth (mb)')
    
    ax2.set_xlabel('$\Delta$')
    ax2.scatter(delta, depth)
    ax2.set_ylim(depth[-1] + 10, 0)
    ax2.set_xlim(-max(abs(delta)) -0.01, max(abs(delta)) + 0.01)
    ax2.tick_params(left=False)
    
    ax3.barh(depth, delta**2, height=2)
    ax3.set_xlabel('$\Delta^2$')
    ax3.set_ylim(depth[-1] + 10, 0)
    ax3.tick_params(left=False)
    fig.suptitle(df['Dates'].iloc[loc])
    fig.tight_layout()
    plt.show()


def penalty_f(z, a, c, MLD):
    return a * np.exp(- (z - MLD)**2 / 2 / c**2)


def modified_fitness(individuals, z, y, args, MLD, a, c,):
    ''' Modfied version to implement higher error weights to points in
    the MLD '''
    
    alpha = fitness(individuals, z, y, args) * np.sqrt(len(y)) \
            / np.sum(((y - fit_function(individuals, z, args))**2 
            * penalty_f(z, a, c, MLD)), axis=1)

    fitness = np.sqrt(np.sum(((y - fit_function(individuals, z, args))**2)
              * (1 + alpha)) / len(y))

    return fitness


def plot_multiple_profiles(df, temp, depth, locs):
    
    n = len(locs)
    for loc in locs:
        if isinstance(loc, datetime):
            loc = date_to_iloc(df, loc)

    if n % 2 != 0:
        raise Exception('This function can only plot an even number \
                        of profiles, {:.0f} were given'.format(n))

    max_ylim = np.max(depth) + 10
    zz = np.linspace(0, max_ylim, 300)

    fig, axes = plt.subplots(int(n/2), 2, figsize=(6.5, n/2*3))
    axes = axes.reshape(n)

    for ax, loc in zip(axes, locs):
        temp_loc = if_masked_to_array(temp[loc])
        depth_loc = if_masked_to_array(depth[loc])
        ax.scatter(temp_loc, depth_loc, marker='o', fc='None', ec='tab:red')
        ax.axhline(df.iloc[loc, 3], c='grey', ls='--')
        ax.set_ylim(max_ylim, 0)
        ax.set_xlim(9.5, 18)

        ax.plot(fit_function(zz, df, loc), zz)

        ax.text(0.7, 0.1, 'em{:.2f}'.format(df.loc[loc, 'em']), bbox={'facecolor': 'w', 'alpha': 0.5,
                                         'pad': 5}, transform=ax.transAxes, ha='center')
        ax.set_xlabel('Temperature (ºC)')
        ax.set_ylabel('Depth (mb)')
        ax.set_title(df['Dates'].iloc[loc])

    fig.tight_layout()
    plt.show()


def animate_profile_evolution(df, tems, depth, start_number, final_number, number_plots, filename):
    numbers = np.linspace(start_number, final_number, number_plots, dtype='int')
    zz = np.linspace(0, 175, 300)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.set_xlim((10, 20))
    ax.set_xlabel('Temperatura (ºC)')
    ax.set_ylabel('Profundidad (mb)')
    ax.set_ylim(np.max(depth) + 5, 0)
    ax.set_xlim(9.5, 18)
    fig.tight_layout()

    points, = ax.plot([], [], 'o', mfc='None', mec='tab:red')
    line, = ax.plot([], [], c='tab:blue')
    mld, = ax.plot([], [], c='grey', ls='--')
    title = ax.text(0.7, 0.1, '', bbox={'facecolor': 'w', 'alpha': 0.5,
                                         'pad': 5}, transform=ax.transAxes, ha='center')

    def animate(i):
        points.set_data(tems[i], depth[i])
        line.set_data(fit_function(zz, df, i), zz)
        mld.set_data((9.5, 18), (df.iloc[i, 3], df.iloc[i, 3]))
        title.set_text('{}'.format(df['Dates'][i]))

    ani = FuncAnimation(fig, animate, frames=numbers, interval=70)
    ani.save(reports_dir / 'movies' / filename)


def plot_thermistor_temperature(temp, depth, date, i, wide='True', lims=[None, None], interval=None):
    '''Plot a single thermistor temperature series 
    '''

    if isinstance(temp, np.ma.core.MaskedArray):
        date = date[temp[:, i].mask==False]
        temp = if_masked_to_array(temp[:, i])
        depth = if_masked_to_array(depth[:, i])


    if isinstance(lims[0], datetime):
        lims[0] = date_to_iloc(date, lims[0])

    if isinstance(lims[1], datetime):
        lims[1] = date_to_iloc(date, lims[1])

    if isinstance(interval, timedelta):
        inteval = timedelta_to_interval()


    date = date[lims[0]:lims[1]:interval]
    temp = temp[lims[0]:lims[1]:interval]
    depth = depth[lims[0]:lims[1]:interval]

    
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)

    if wide:
        fig, ax = plt.subplots(figsize=(7, 3.75))
    else: 
        fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(date, temp)
    ax.set_title(f'Temperature at depth {depth[0]} db (ºC)')
    ax.set_xlabel('Date')
    fig.tight_layout()
    plt.show()
    

def plot_column_temperature(temp, depth, date, lims = [None, None], interval=None):

    if isinstance(lims[0], datetime):
        lims = [date_to_iloc(date, lim) for lim in lims]

    if isinstance(interval, timedelta):
        inteval = timedelta_to_interval()

    date = date[lims[0]:lims[1]:interval]
    temp = temp[:, lims[0]:lims[1]:interval]

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    cmap = plt.colormaps['viridis']
    levels = MaxNLocator(nbins=256).tick_values(temp.min(), temp.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylim(depth[-1] + 10, 0)
    # for i in depth:
    #     ax.axhline(i)
    im = ax.pcolormesh(date, depth, temp, cmap=cmap, norm=norm)
    fig.colorbar(im)
    fig.tight_layout()
    plt.show()


def find_thermocline_thermistors(df, depth, temp, loc):
    '''
    TODO
    ''' 
    df = df.iloc[loc]
    mld = df['D1'][loc]
    idx = np.where(depth < mld)
    depth = np.delete(depth, idx, axis=0)

    perm_thermocline = lambda z: df['a3'][loc] - df['a2'][loc] + df['b3'][loc] * (z - mld)
    
    max_delta = 1
    
    thermocline_therm = []
    for i, depth in enumerate(depth):
        if abs(perm_thermocline(depth) - temp[i, loc]) >= max_delta:
            thermocline_therm.append(depth)

    return thermocline_therm
    
if __name__ == '__main__':
    pass
