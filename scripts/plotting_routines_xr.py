import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from analysis_routines import *
from IPython.display import Video
from config import data_dir, reports_dir


def plot_fit_variable(df, variable, period=None, ylim=None, plot=True, wide=True):
    '''Plot given fit variable (e.g D1, a1, ...) for time between lims and 
    with given interval.
    '''
    
    slice_ = slice(*period)
    var = df[variable][slice_]
    date = df[slice_].index

    locator = mdates.AutoDateLocator(minticks=4, maxticks=None)
    formatter = mdates.ConciseDateFormatter(locator)
    minor_locator = mdates.AutoDateLocator(minticks=6)

    if wide:
        fig, ax = plt.subplots(figsize=(7, 3.75))
    else: 
        fig, ax = plt.subplots()

    ax.scatter(date, var, s=8)

    if variable == 'D1':
        ax.set_ylim((max(var) + 4, min(var) - 4))
        ax.set_ylabel('MLD (db)')

    else:
        ax.set_ylabel(variable)

    if ylim is not None:
        ax.set_ylim(*ylim)
    

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(minor_locator)
    # ax.xaxis.set_minor_locator(formatter)

    if plot:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

def plot_arbitrary_variable(variable, date=None, period=[None, None], ylim=None, type='scatter'):
    if date is None:
        slice_ = slice(*period)
        variable = variable[slice_]
        date = variable[slice_].index

    else:
        slice_ = slice(*period)
        variable = variable[slice_]
        date = date[slice_]


    locator = mdates.AutoDateLocator(minticks=4, maxticks=None)
    formatter = mdates.ConciseDateFormatter(locator)
    minor_locator = mdates.AutoDateLocator(minticks=6)

    fig, ax = plt.subplots(figsize=(7, 3.75))
    if type=='scatter':
        ax.scatter(date, variable, s=8)

    elif type=='plot':
        ax.plot(date, variable)

    if ylim is not None:
        ax.set_ylim(*ylim)


    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(minor_locator)

    fig.tight_layout()
    plt.show()

def plot_profile_fit(df, data, date_i, xlim=None, save=False):
    '''Plot measured vertical profile and fit for measure at loc
    '''

    if isinstance(date_i, int):
        data_i = data.isel(date=date_i)
        mld = df.iloc[date_i]['D1']

    else:
        data_i = data.sel(date=date_i)
        mld = df.loc[date_i, 'D1']

    pres_i = data_i.meassured_depth
    temp_i = data_i.temp
    date_i_str = np.datetime_as_string(data_i.date, unit='s')

    pres_i = pres_i[np.isfinite(temp_i)]
    temp_i = temp_i[np.isfinite(temp_i)]

    zz = np.linspace(1, pres_i[-1] + 5, 300)

    fig, ax = plt.subplots(figsize=(4, 4.6875))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.scatter(temp_i, pres_i, marker='o', fc='None', ec=colors[1], s=24)
    ax.axhline(mld, c='grey', ls='--') # plot MLD
    ax.set_ylim(pres_i[-1] + 10, 0)

    if xlim is None:
        ax.set_xlim(11, 16)
    
    else:
        ax.set_xlim(*xlim)

    ax.plot(fit_function(zz, df, date_i), zz, lw=1, c=colors[0])
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (db)')
    ax.set_title(date_i_str)
    fig.tight_layout()
    if save:
        fig.savefig(str(save))
    plt.show()


def plot_RMS_fit(df, temp, depth, loc):
    ''' Plot experimental profile and fit with diference between fit and profile,
    and square of that difference.
    '''

    if isinstance(loc, datetime):
        loc = date_to_idx(df['date'], loc)
    
    delta = temp[:, loc] - fit_function(depth, df, loc)
    zz = np.linspace(0, depth[-1] + 5, 300)
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (7.5, 3.75))
    
    ax1.scatter(temp[:, loc], depth, marker='o', fc='None', ec='tab:red', s=22)
    ax1.axhline(df.iloc[loc, 3], c='grey', ls='--') # plot MLD
    ax1.set_ylim(depth[-1] + 10, 0)
    ax1.set_xlim(11, 16)
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
    fig.suptitle(df['date'].iloc[loc])
    fig.tight_layout()
    plt.show()


def plot_multiple_profiles(df, data, dates_i, xlim=None):
    
    n = len(dates_i)

    if n % 2 != 0:
        raise Exception('This function can only plot an even number \
                        of profiles, {:.0f} were given'.format(n))


    fig, axes = plt.subplots(int(n/2), 2, figsize=(6.5, n/2*3))
    axes = axes.reshape(n)

    for ax, date_i in zip(axes, dates_i):

        if isinstance(date_i, int):
            data_i = data.isel(date=date_i)
            mld = df.iloc[date_i, 3]
            em = df.iloc[date_i]['em']

        else:
            data_i = data.sel(date=date_i)
            mld = df.loc[date_i, 'D1']
            em = df.loc[date_i, 'em']

        pres_i = data_i.meassured_depth
        temp_i = data_i.temp
        date_i_str = np.datetime_as_string(data_i.date, unit='s')

        pres_i = pres_i[np.isfinite(temp_i)]
        temp_i = temp_i[np.isfinite(temp_i)]


        max_ylim = np.max(pres_i) + 10
        zz = np.linspace(0, max_ylim, 300)

        ax.scatter(temp_i, pres_i, marker='o', fc='None', ec='tab:red')
        ax.axhline(mld, c='grey', ls='--')
        ax.set_ylim(max_ylim, 0)
        if xlim is None:
            ax.set_xlim(11, 16)

        else:
            ax.set_xlim(*xlim)

        ax.plot(fit_function(zz, df, date_i), zz)

        ax.text(0.7, 0.1, 'em{:.2f}'.format(em), bbox={'facecolor': 'w', 'alpha': 0.5,
                                         'pad': 5}, transform=ax.transAxes, ha='center')
        ax.set_xlabel('Temperature (ºC)')
        ax.set_ylabel('Depth (mb)')
        ax.set_title(date_i_str)

    fig.tight_layout()
    plt.show()


def animate_profile_evolution(df, data, filename, optional_mld=None, xlim=None,
                              period=[None, None], num_plots=300, interval=70):

    slice_ = slice(*period)
    data_period = data.sel(date=slice_)
    temp = data_period.temp
    pres = data_period.meassured_depth
    date = np.datetime_as_string(data_period.date, unit='s')
    fit = df[slice_]

    frames = np.linspace(0, len(date) - 1, num_plots, dtype='int')
    zz = np.linspace(0, 175, 300)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.set_xlim(11, 16)
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (db)')
    ax.set_ylim(np.max(pres) + 3, 0)
    if xlim is None:
        ax.set_xlim(11, 16)
    else:
        ax.set_xlim(*xlim)
    fig.tight_layout()

    points, = ax.plot([], [], 'o', mfc='None', mec='tab:red')
    line, = ax.plot([], [], c='tab:blue')
    mld, = ax.plot([], [], c='grey', ls='--')
    if isinstance(optional_mld, np.ndarray):
        opt_mld, = ax.plot([], [], c='grey', ls=':')
    title = ax.text(0.7, 0.1, '', bbox={'facecolor': 'w', 'alpha': 0.5,
                                         'pad': 5}, transform=ax.transAxes, ha='center')

    def animate(i):
        points.set_data(temp[i], pres[i])
        line.set_data(fit_function(zz, fit, i), zz)
        mld.set_data((9.5, 18), (fit.iloc[i]['D1'], fit.iloc[i]['D1']))

        if isinstance(optional_mld, np.ndarray):
            opt_mld.set_data((9.5, 18), (optional_mld[i], optional_mld[i]))

        title.set_text('{}'.format(date[i]))

    ani = FuncAnimation(fig, animate, frames=frames, interval=interval)
    ani.save(reports_dir / 'movies' / filename)



def plot_thermistor_temperature(data, idxs, period, wide=True):
    '''Plot a single thermistor temperature series 
    '''
    locator = mdates.AutoDateLocator(minticks=4, maxticks=None)
    formatter = mdates.ConciseDateFormatter(locator)
    minor_locator = mdates.AutoDateLocator(minticks=6)

    slice_ = slice(*period)
    data_period = data.sel(date=slice_)

    if wide:
        fig, ax = plt.subplots(figsize=(7, 3.75))
    else: 
        fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(minor_locator)

    if isinstance(idxs, list):
        for i in idxs:
            temp_i = data_period.temp[:, i]

            date_i = data_period.date[np.isfinite(temp_i)]
            depth_i = data_period.meassured_depth[:, i][np.isfinite(temp_i)][0].data
            temp_i = temp_i[np.isfinite(temp_i)]
            
            ax.plot(date_i, temp_i, label='{:.1f} db'.format(depth_i))
            ax.legend()
            ax.set_ylabel('Temperature ºC')

    else:
        i = idxs
        temp_i = data_period.temp[:, i]
        date_i = data_period.date[np.isfinite(temp_i)]
        depth_i = data_period.meassured_depth[:, i][np.isfinite(temp_i)][0].data
        temp_i = temp_i[np.isfinite(temp_i)]

        ax.plot(date_i, temp_i)
        ax.set_title('Temperature at depth {:.1f} db (ºC)'.format(depth_i))
    fig.tight_layout()
    plt.show()


def plot_column_temperature(data, df_fit=None, period=[None, None], smooth=True,
                            ylims=None, wide=True, save=False):

    if len(period) != 3:
        slice_ = slice(period[0], period[1], 120)

    else:
        slice_ = slice(*period)

    data_period = data.sel(date=slice_)
    temp = data_period.temp
    date = data_period.date
    depths = data_period.depth

    xx = np.linspace(0, len(date) - 1, 500, dtype='int')
    date = date[xx]
    temp = temp[xx]
    if df_fit is not None:
        mld_array = df_fit[slice_]['D1'].to_numpy()[xx]




    locator = mdates.AutoDateLocator(minticks=4, maxticks=None)
    formatter = mdates.ConciseDateFormatter(locator)
    minor_locator = mdates.AutoDateLocator(minticks=6)

    cmap = plt.colormaps['viridis']
    levels = MaxNLocator(nbins=256).tick_values(temp.min(), temp.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    if wide:
        fig, ax = plt.subplots(figsize=(8.6, 3.75))
    else: 
        fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(minor_locator)

    if ylims is not None:
        ax.set_ylim(ylims[0], ylims[1])
    else:
        ax.set_ylim(depths[-1], 0)

    for i in depths:
        ax.axhline(i, ls=':', c='grey', lw=0.3)

    x, y = np.meshgrid(date, depths)
    if smooth:
        im = ax.pcolormesh(x.T, y.T, temp, cmap=cmap, norm=norm, shading='gouraud')

    else:
        im = ax.pcolormesh(x.T, y.T, temp, cmap=cmap, norm=norm, shading='nearest')

    if df_fit is not None:
        ax.plot(date, mld_array, c='black', lw=0.45, ls='--')

    ax.set_title('Column temperature (ºC)')
    cbar = fig.colorbar(im, extend='both')
    cbar.ax.tick_params(which='minor', axis='y', right=False)
    fig.tight_layout()
    if save is not False:
        fig.savefig(reports_dir / str(save))
    plt.show()


def display_video(filename):
    video_path = reports_dir / 'movies' / filename
    return Video(video_path, embed=True, width=500)


def plot_interpolation(z, y, z_int, y_int):
    if isinstance(z, np.ma.core.MaskedArray):
        z = np.asarray(z[z.mask==False])
        y = np.asarray(y[y.mask==False])

    z = z[np.isfinite(y)]
    y = y[np.isfinite(y)]

    fig, ax = plt.subplots(figsize=(4, 4.6875))
    ax.scatter(y, z, marker='o', fc='None', ec='tab:blue', s=22)
    ax.scatter(y_int, z_int, marker='x')
    #ax.plot(y_int, z_int)
    ax.set_ylim(z[-1] + 10, 0)
    ax.set_xlim(11, 16)
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (mb)')
    fig.tight_layout()
    plt.show()


