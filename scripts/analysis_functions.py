import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from config import data_dir, reports_dir

def load_time_series(filename, convert_date=True):
    '''Load time series saved as filename in data_dir/time_series
    '''

    file_path = data_dir / 'time_series' / filename
    if not filename.endswith('.h5'):
        raise Exception('Only .h5 time series supported')

    with h5py.File(file_path, 'r') as f:
        temperature = np.array(f['temperature'])
        pressure = np.array(f['pressure'])
        date = np.array(f['date'])
        lat = np.array(f['lat'])
        lon = np.array(f['lon'])

    if convert_date:
        date = np.array([datetime.fromtimestamp(i) for i in date])

    return temperature, pressure, date, lat, lon

    
def load_SHDR_fit(filename):
    '''Load  saved SHDR fit as filename in data_dir/SHDR_fit.
    Returns data frame containing fit of time series.
    '''

    file_path = data_dir / 'SHDR_fit' / filename
    df_fit = pd.read_csv(file_path)
    df_fit['Dates'] = df_fit['Dates'].apply(lambda x: datetime.fromtimestamp(x))
    return df_fit
     


def date_to_iloc(dates, date):
    ''' Use fit dataframe to convert a given date to the closest iloc. 
    '''
    
    try:
        iloc = np.where(dates == date)[0][0]

    except:
        possible_dates = [date + timedelta(seconds=dt) for dt in (-2, -1, 1, 2)]
        iloc = np.in1d(dates, possible_dates).argmax()
    return iloc


def timedelta_to_interval(timedelta, dt=5):
        '''Return interval in rows of df fit for a given timedelta
        '''

        interval = np.int(np.round(timedelta.total_seconds() / dt))
        return interval



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


def plot_fit_variable(df, variable, lims = [None, None], interval=None):
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

    fig, ax = plt.subplots()
    ax.scatter(dates, var, s=8)
    ax.set_xlabel('Date')
    if variable in dic:
        ax.set_ylabel(dic[variable])
    else:
        ax.set_ylabel(variable)
    fig.tight_layout()
    plt.show()


def plot_profile_fit(df, temp, pres, loc):
    '''Plot measured vertical profile and fit for measure at loc
    '''

    if isinstance(loc, datetime):
        loc = date_to_iloc(df['Dates'], loc)
        print(loc)

    zz = np.linspace(0, pres[-1] + 5, 300)

    fig, ax = plt.subplots(figsize=(4, 4.6875))
    ax.scatter(temp[:, loc], pres, marker='o', fc='None', ec='tab:red', s=22)
    ax.axhline(df.iloc[loc, 3], c='grey', ls='--') # plot MLD
    ax.set_ylim(pres[-1] + 10, 0)
    ax.set_xlim(9.5, 18)
    ax.plot(fit_function(zz, df, loc), zz)
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (mb)')
    ax.set_title(df['Dates'].iloc[loc])
    fig.tight_layout()
    plt.show()


def fitness(df, temp, pres, loc):
    ''' Get RMS for profile in loc at height z 
    ''' 

    if isinstance(loc, datetime):
        loc = date_to_iloc(df['Dates'], loc)

    temp_loc = temp[:, loc]
    fitness = np.sqrt(np.sum((temp_loc - fit_function(pres, df, loc))**2) / len(temp_loc))
    return fitness
    

def plot_RMS_fit(df, temp, pres, loc):
    ''' Plot experimental profile and fit with diference between fit and profile,
    and square of that difference.
    '''

    if isinstance(loc, datetime):
        loc = date_to_iloc(df['Dates'], loc)
    
    delta = temp[:, loc] - fit_function(pres, df, loc)
    zz = np.linspace(0, pres[-1] + 5, 300)
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (7.5, 3.75))
    
    ax1.scatter(temp[:, loc], pres, marker='o', fc='None', ec='tab:red', s=22)
    ax1.axhline(df.iloc[loc, 3], c='grey', ls='--') # plot MLD
    ax1.set_ylim(pres[-1] + 10, 0)
    ax1.set_xlim(9.5, 18)
    ax1.plot(fit_function(zz, df, loc), zz)
    ax1.set_xlabel('Temperature (ºC)')
    ax1.set_ylabel('Depth (mb)')
    
    ax2.set_xlabel('$\Delta$')
    ax2.scatter(delta, pres)
    ax2.set_ylim(pres[-1] + 10, 0)
    ax2.set_xlim(-max(abs(delta)) -0.01, max(abs(delta)) + 0.01)
    ax2.tick_params(left=False)
    
    ax3.barh(pres, delta**2, height=2)
    ax3.set_xlabel('$\Delta^2$')
    ax3.set_ylim(pres[-1] + 10, 0)
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


def plot_multiple_profiles(df, temp, pres, locs):
    
    n = len(locs)
    for loc in locs:
        if isinstance(loc, datetime):
            loc = date_to_iloc(df, loc)

    if n % 2 != 0:
        raise Exception('This function can only plot an even number \
                        of profiles, {:.0f} were given'.format(n))

    zz = np.linspace(0, 185, 300)

    fig, axes = plt.subplots(int(n/2), 2, figsize=(6.5, n/2*3))
    axes = axes.reshape(n)

    for ax, loc in zip(axes, locs):
        ax.scatter(temp[:, loc], pres, marker='o', fc='None', ec='tab:red')
        ax.axhline(df.iloc[loc, 3], c='grey', ls='--')
        ax.set_ylim(pres[-1] + 10, 0)
        ax.set_xlim(9.5, 18)

        ax.plot(fit_function(zz, df, loc), zz)
        ax.set_xlabel('Temperature (ºC)')
        ax.set_ylabel('Depth (mb)')
        ax.set_title(df['Dates'].iloc[loc])

    fig.tight_layout()
    plt.show()


def animate_profile_evolution(df, tems, pres, start_number, final_number, number_plots, filename):
    numbers = np.linspace(start_number, final_number, number_plots, dtype='int')
    zz = np.linspace(0, 175, 300)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.set_xlim((10, 20))
    ax.set_xlabel('Temperatura (ºC)')
    ax.set_ylabel('Profundidad (mb)')
    ax.set_ylim(pres[-1] + 10, 0)
    ax.set_xlim(9.5, 18)
    fig.tight_layout()

    points, = ax.plot([], [], 'o', mfc='None', mec='tab:red')
    line, = ax.plot([], [], c='tab:blue')
    mld, = ax.plot([], [], c='grey', ls='--')
    title = ax.text(0.7, 0.1, '', bbox={'facecolor': 'w', 'alpha': 0.5,
                                         'pad': 5}, transform=ax.transAxes, ha='center')

    def animate(i):
        points.set_data(tems[:, i], pres)
        line.set_data(fit_function(zz, df, i), zz)
        mld.set_data((9.5, 18), (df.iloc[i, 3], df.iloc[i, 3]))
        title.set_text('{}'.format(df['Dates'][i]))

    ani = FuncAnimation(fig, animate, frames=numbers, interval=80)
    ani.save(reports_dir / 'movies' / filename)


def plot_thermistor_series(temp, pres, date, i, wide='True', lims=[None, None], interval=None):
    '''Plot a single thermistor temperature series 
    '''
    if isinstance(lims[0], datetime):
        lims[0] = date_to_iloc(date, lims[0])

    if isinstance(lims[1], datetime):
        lims[1] = date_to_iloc(date, lims[1])

    if isinstance(interval, timedelta):
        inteval = timedelta_to_interval()

    date = date[lims[0]:lims[1]:interval]
    temp = temp[i, lims[0]:lims[1]:interval]

    
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)

    if wide:
        fig, ax = plt.subplots(figsize=(7, 3.75))
    else: 
        fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(date, temp)
    ax.set_title(f'Temperature at depth {pres[i]} db (ºC)')
    ax.set_xlabel('Date')
    fig.tight_layout()
    plt.show()
    

def plot_column_temperature(temp, pres, date, lims = [None, None], interval=None):

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
    ax.set_ylim(pres[-1] + 10, 0)
    # for i in pres:
    #     ax.axhline(i)
    im = ax.pcolormesh(date, pres, temp, cmap=cmap, norm=norm)
    fig.colorbar(im)
    fig.tight_layout()
    plt.show()


def find_thermocline_thermistors(df, pres, temp, loc):
    '''
    TODO
    ''' 
    df = df.iloc[loc]
    mld = df['D1'][loc]
    idx = np.where(pres < mld)
    pres = np.delete(pres, idx, axis=0)

    perm_thermocline = lambda z: df['a3'][loc] - df['a2'][loc] + df['b3'][loc] * (z - mld)
    
    max_delta = 1
    
    thermocline_therm = []
    for i, depth in enumerate(pres):
        if abs(perm_thermocline(depth) - temp[i, loc]) >= max_delta:
            thermocline_therm.append(depth)

    return thermocline_therm
    
if __name__ == '__main__':
    pass
