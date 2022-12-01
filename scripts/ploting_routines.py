import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from analysis_routines import *


def plot_fit_variable(df, variable, lims = [None, None], interval=None, plot=True):
    '''Plot given fit variable (e.g D1, a1, ...) for time between lims and 
    with given interval.
    '''
    
    if isinstance(interval, timedelta):
        interval = timedelta_to_interval(df['Dates'], interval)

    if isinstance(lims[0], datetime):
        lims = [date_to_idx(df['Dates'], lim) for lim in lims]

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
        loc = date_to_idx(df['Dates'], loc)

    zz = np.linspace(1, depth[-1] + 5, 300)

    temp_loc = if_masked_to_array(temp[loc])
    pres_loc = if_masked_to_array(depth[loc])

    fig, ax = plt.subplots(figsize=(4, 4.6875))
    ax.scatter(temp_loc, pres_loc, marker='o', fc='None', ec='tab:red', s=22)
    ax.axhline(df.iloc[loc, 3], c='grey', ls='--') # plot MLD
    ax.set_ylim(pres_loc[-1] + 10, 0)
    ax.set_xlim(9.5, 18)
    ax.plot(fit_function(zz, df, loc), zz, lw=1)
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (mb)')
    ax.set_title(df['Dates'].iloc[loc])
    fig.tight_layout()
    plt.show()


def plot_RMS_fit(df, temp, depth, loc):
    ''' Plot experimental profile and fit with diference between fit and profile,
    and square of that difference.
    '''

    if isinstance(loc, datetime):
        loc = date_to_idx(df['Dates'], loc)
    
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


def plot_multiple_profiles(df, temp, depth, locs):
    
    n = len(locs)
    for loc in locs:
        if isinstance(loc, datetime):
            loc = date_to_idx(df, loc)

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


def animate_profile_evolution(df, tems, depth, filename, optional_mld=None,
                              start_loc=0, final_loc=None, number_plots=300):

    if isinstance(start_loc, datetime):
        start_loc = date_to_idx(df['Dates'], start_loc)

    if isinstance(final_loc, datetime):
        final_loc = date_to_idx(df['Dates'], final_loc)

    if final_loc == None:
        final_loc = len(df) - 1
    numbers = np.linspace(start_loc, final_loc, number_plots, dtype='int')
    zz = np.linspace(0, 175, 300)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.set_xlim((10, 20))
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (db)')
    ax.set_ylim(np.max(depth) + 3, 0)
    ax.set_xlim(9.5, 18)
    fig.tight_layout()

    points, = ax.plot([], [], 'o', mfc='None', mec='tab:red')
    line, = ax.plot([], [], c='tab:blue')
    mld, = ax.plot([], [], c='grey', ls='--')
    if isinstance(optional_mld, np.ndarray):
        opt_mld, = ax.plot([], [], c='grey', ls=':')
    title = ax.text(0.7, 0.1, '', bbox={'facecolor': 'w', 'alpha': 0.5,
                                         'pad': 5}, transform=ax.transAxes, ha='center')

    def animate(i):
        points.set_data(tems[i], depth[i])
        line.set_data(fit_function(zz, df, i), zz)
        mld.set_data((9.5, 18), (df.iloc[i, 3], df.iloc[i, 3]))
        if isinstance(optional_mld, np.ndarray):
            opt_mld.set_data((9.5, 18), (optional_mld[i], optional_mld[i]))
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
        lims[0] = date_to_idx(date, lims[0])

    if isinstance(lims[1], datetime):
        lims[1] = date_to_idx(date, lims[1])

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
        lims = [date_to_idx(date, lim) for lim in lims]

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
