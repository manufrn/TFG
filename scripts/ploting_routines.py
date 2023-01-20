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


def plot_fit_variable(df, variable, lims = [None, None], interval=None, plot=True, wide=True):
    '''Plot given fit variable (e.g D1, a1, ...) for time between lims and 
    with given interval.
    '''
    
    if isinstance(interval, timedelta):
        interval = timedelta_to_interval(interval)

    if isinstance(lims[0], datetime):
        lims[0] = date_to_idx(df['date'], lims[0])

    if isinstance(lims[1], datetime):
        lims[1] = date_to_idx(df['date'], lims[1])

    var = df[variable][lims[0]:lims[1]:interval]
    dates = df['date'][lims[0]:lims[1]:interval]

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)

    if wide:
        fig, ax = plt.subplots(figsize=(7, 3.75))
    else: 
        fig, ax = plt.subplots()

    ax.scatter(dates, var, s=8)

    if variable == 'D1':
        ax.set_ylim((max(var) + 4, min(var) - 4))
        ax.set_ylabel('MLD (db)')

    else:
        ax.set_ylabel(variable)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    if plot:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax


def plot_profile_fit(df, temp, depth, loc, save=False):
    '''Plot measured vertical profile and fit for measure at loc
    '''



    temp_loc = if_masked_to_array(temp[loc])
    pres_loc = if_masked_to_array(depth[loc])

    zz = np.linspace(1, pres_loc[-1] + 5, 300)

        
    fig, ax = plt.subplots(figsize=(4, 4.6875))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.scatter(temp_loc, pres_loc, marker='o', fc='None', ec=colors[1], s=24)
    ax.axhline(df.iloc[loc, 3], c='grey', ls='--') # plot MLD
    ax.set_ylim(pres_loc[-1] + 10, 0)
    if depth.shape[1] == 17:
        ax.set_xlim(11, 25)
    else:
        ax.set_xlim(11, 16)

    ax.plot(fit_function(zz, df, loc), zz, lw=1, c=colors[0])
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (db)')
    ax.set_title(df['date'].iloc[loc])
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


def plot_multiple_profiles(df, temp, depth, locs, em=None):
    
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
        em_loc = em[loc] if isinstance(em, np.ndarray) else df['em'][loc]
        ax.scatter(temp_loc, depth_loc, marker='o', fc='None', ec='tab:red')
        ax.axhline(df.iloc[loc, 3], c='grey', ls='--')
        ax.set_ylim(max_ylim, 0)
        ax.set_xlim(11, 16)

        ax.plot(fit_function(zz, df, loc), zz)

        ax.text(0.7, 0.1, 'em{:.2f}'.format(em_loc), bbox={'facecolor': 'w', 'alpha': 0.5,
                                         'pad': 5}, transform=ax.transAxes, ha='center')
        ax.set_xlabel('Temperature (ºC)')
        ax.set_ylabel('Depth (mb)')
        ax.set_title(df['date'].iloc[loc])

    fig.tight_layout()
    plt.show()


def animate_profile_evolution(df, tems, depth, filename, optional_mld=None,
                              start_loc=0, final_loc=None, number_plots=300):

    if isinstance(start_loc, datetime):
        start_loc = date_to_idx(df['date'], start_loc)

    if isinstance(final_loc, datetime):
        final_loc = date_to_idx(df['date'], final_loc)

    if final_loc == None:
        final_loc = len(df) - 1
    numbers = np.linspace(start_loc, final_loc, number_plots, dtype='int')
    zz = np.linspace(0, 175, 300)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.set_xlim(11, 16)
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (db)')
    ax.set_ylim(np.max(depth) + 3, 0)
    ax.set_xlim(11, 16)
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
        title.set_text('{}'.format(df['date'][i]))

    ani = FuncAnimation(fig, animate, frames=numbers, interval=70)
    ani.save(reports_dir / 'movies' / filename)



def plot_thermistor_temperature(temp, depth, date, idxs, wide='True', lims=[None, None], interval=None):
    '''Plot a single thermistor temperature series 
    '''
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)

    if wide:
        fig, ax = plt.subplots(figsize=(7, 3.75))
    else: 
        fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    if isinstance(idxs, list):
        for i in idxs:
            if isinstance(temp, np.ma.core.MaskedArray):
                date_i = date[temp[:, i].mask==False]
                temp_i = if_masked_to_array(temp[:, i])

            if isinstance(lims[0], datetime):
                lim_0 = date_to_idx(date_i, lims[0])

            if isinstance(lims[1], datetime):
                lim_1 = date_to_idx(date_i, lims[1])

            if isinstance(interval, timedelta):
                interval_i = timedelta_to_interval(interval)

            else:
                interval_i = interval

            date_i = date_i[lim_0:lim_1:interval_i]
            temp_i = temp_i[lim_0:lim_1:interval_i]

            ax.plot(date_i, temp_i, label=f'{depth[0, i]} m')
            ax.legend()
            ax.set_ylabel('Temperature ºC')

    else:
        i = idxs
        if isinstance(temp, np.ma.core.MaskedArray):
            date_i = date[temp[:, i].mask==False]
            temp_i = if_masked_to_array(temp[:, i])

            if isinstance(lims[0], datetime):
                lims[0] = date_to_idx(date_i, lims[0])

            if isinstance(lims[1], datetime):
                lims[1] = date_to_idx(date_i, lims[1])

            if isinstance(interval, timedelta):
                interval = timedelta_to_interval(interval)

            date_i = date_i[lims[0]:lims[1]:interval]
            temp_i = temp_i[lims[0]:lims[1]:interval]

            ax.plot(date_i, temp_i)
            ax.set_title(f'Temperature at depth {depth[0, i]} db (ºC)')
    fig.tight_layout()
    plt.show()


def plot_column_temperature(temp, date, pres, df_fit=None, interval=120, lims=[None, None], 
                            smooth=True, ylims=None, wide=True, save=False):

    if isinstance(lims[0], datetime):
        lims[0] = date_to_idx(date, lims[0])
    
    if isinstance(lims[1], datetime):
        lims[1] = date_to_idx(date, lims[1])

    if isinstance(interval, timedelta):
        interval = timedelta_to_interval(interval)
        date = date[lims[0]:lims[1]:interval]
        temp = temp[lims[0]:lims[1]:interval]

        if df_fit is not None:
            mld_array = df_fit['D1'].to_numpy()
            mld_array = mld_array[lims[0]:lims[1]:interval]


    else:
        date = date[lims[0]:lims[1]:interval]
        temp = temp[lims[0]:lims[1]:interval]
        xx = np.linspace(0, len(date) - 1, 500, dtype='int')
        date = date[xx]
        temp = temp[xx]

        if df_fit is not None:
            mld_array = df_fit['D1'].to_numpy()
            mld_array = mld_array[lims[0]:lims[1]:interval][xx]

    if lims[0] is not None:
        depths = pres[lims[0]] 

    else:
        depths = pres[0]
    
    # temp = temp[:-1]
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    cmap = plt.colormaps['viridis']
    levels = MaxNLocator(nbins=256).tick_values(temp.min(), temp.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    if wide:
        fig, ax = plt.subplots(figsize=(8.6, 3.75))
    else: 
        fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
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
    # cbar.set_ticks([])
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
    ax.set_xlim(9.5, 18)
    ax.set_xlabel('Temperature (ºC)')
    ax.set_ylabel('Depth (mb)')
    fig.tight_layout()
    plt.show()


def interact_profile(data_chain, fit_chain, range_dates, dn=24):
    slice_ = slice(*range_dates, dn)
    temp_chain_range = data_chain.temp.loc[slice_].data
    date_chain_range = data_chain.date.loc[slice_].data
    fit_chain_range = fit_chain.loc[slice_]
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    N = len(date_chain_range)
    
    int_wdgt = IntSlider(
    description='Number:',
    value=0,
    min=0, max=N-1, step=1,
    layout = Layout(width='100%'))
    
    zz = np.linspace(0, 200, 300)

    
    def plot_(i):
        fig = plt.figure(figsize=(4, 4.6875))
        plt.scatter(temp_chain_range[i], data_chain.depth, marker='o', fc='None', ec=colors[1], s=24)
        plt.plot(fit_function(zz, fit_chain_range, i), zz, lw=1)
        plt.title(np.datetime_as_string(date_chain_range[i], unit='s'))
        plt.ylim(200, 0)
        plt.show()
        del fig
        
    interact(plot_, i=int_wdgt)
