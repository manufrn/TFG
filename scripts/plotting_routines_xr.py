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
from ipywidgets import Layout, interact, IntSlider
from scipy.stats import chi2
from statsmodels.tsa.stattools import acf, adfuller, ccf, ccovf
from scipy.signal import find_peaks
from config import data_dir, reports_dir


def plot_fit_variable(df, variable, period=None, ylim=None, xlim=None, plot=True, wide=True):
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

    if xlim is not None:
        ax.set_xlim(*xlim)
    

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(minor_locator)
    # ax.xaxis.set_minor_locator(formatter)

    if plot:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

def plot_arbitrary_variable(variable, date=None, period=[None, None], ylim=None, kind='scatter', lw=None):
    if date is None:
        slice_ = slice(*period)
        variable = variable.loc[slice_]
        if type(variable) is pd.Series:
            date = variable.loc[slice_].index
        elif type(variable) is xr.DataArray:
            date = variable.loc[slice_].date

    else:
        slice_ = slice(*period)
        variable = variable[slice_]
        date = date[slice_]


    locator = mdates.AutoDateLocator(minticks=4, maxticks=None)
    formatter = mdates.ConciseDateFormatter(locator)
    minor_locator = mdates.AutoDateLocator(minticks=6)

    fig, ax = plt.subplots(figsize=(9.2, 3.75))
    if kind=='scatter':
        ax.scatter(date, variable, s=8)

    elif kind=='plot':
        if lw is None:
            ax.plot(date, variable)
        else: 
            ax.plot(date, variable, lw=lw)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlim(np.min(date), np.max(date))

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(minor_locator)

    fig.tight_layout()
    plt.show()

def plot_profile_fit(df, data, date_i, xlim=None, save=False):
    '''Plot measured vertical profile and fit for measure at loc
    '''

    if isinstance(date_i, (int, np.integer)):
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

        if isinstance(date_i, (int, np.integer)):
            data_i = data.isel(date=date_i)
            mld = df.iloc[date_i]['D1']
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
        ax.set_ylabel('Depth (db)')
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



def plot_thermistor_temperature(data, idxs, period, xlim=None, wide=True):
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


    if xlim is not None:
        ax.set_xlim(*xlim)

    fig.tight_layout()
    plt.show()


def plot_column_temperature(data, df_fit=None, period=[None, None], smooth=True,
                            ylims=None, wide=True, save=False):

    if len(period) != 3:
        slice_ = slice(period[0], period[1], 120)

    else:
        slice_ = slice(*period)

    data_period = data.sel(date=slice_)

    if ylims is not None:
        bfill = data_period.depth.sel(depth=ylims[0], method='bfill')
        # data_period = data_period.where(data_period.depth < ylims[0], drop=True)
        data_period = data_period.sel(depth=(slice(ylims[1], bfill)))
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
        fig.savefig(reports_dir / 'figures' / str(save))
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

def plot_AGL_data(ds, var, period=[None, None]):
    
    slice_ = slice(*period)
    
    if isinstance(var, list):
        var_arr = np.sum([ds[i].loc[slice_] for i in var], axis=0)
        
    else:
        var_arr = ds[var].loc[slice_]
    date = ds.date.loc[slice_]
    
    locator = mdates.AutoDateLocator(minticks=4, maxticks=None)
    formatter = mdates.ConciseDateFormatter(locator)
    minor_locator = mdates.AutoDateLocator(minticks=6)
    
    fig, ax = plt.subplots(figsize=(7, 3.75))
    ax.xaxis.set_major_locator(locator)
    ax.set_xlim(date[0], date[-1])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.plot(date, var_arr)
    ax.set_title(var)
    fig.tight_layout()
    plt.show()


def plot_spectrum(freqs, pxx, dof, x_units, y_units=None, xlim=None, ylim=None, x='freqs', vlines=None):
    period = 1/freqs

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots()

    if x == 'freqs':
        ax.loglog(freqs, pxx)
        ax.set_xlabel(r'frecuency ({})'.format(x_units))
        
    elif x == 'period':
        ax.loglog(period, pxx)
        ax.set_xlabel(r'Period ({})'.format(x_units))


    if xlim != None:
        ax.set_xlim(*xlim)

    if ylim != None:
        ax.set_ylim(*ylim)

    if vlines != None:
        for i in vlines:
            ax.axvline(i, ls='--', c='grey')

    if y_units is not None:
        y_units = y_units + r'$^{2}$ ' + x_units + r'$^{-1}$'
        ax.set_ylabel('Spectral density ({})'.format(y_units))
    else:
        ax.set_ylabel('Spectral density')


    ax.text(0.06, 0.07, '{} degrees of freedom'.format(int(dof)), transform=ax.transAxes, ha='left',
            fontsize=9, color=colors[0])

    plt.show()


def plot_column_oscilation(column_coefs, component, mld_coef, delta05_coef, pos1, pos2, x_arrow, 
                           ylim=None, rel=False):
    x = []
    ci = []
    depths = column_coefs.depths
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    
    mld_mean = mld_coef.attrs['mean']
    mld_ampl = mld_coef.loc[component]['A']
    delta_mean = delta05_coef.attrs['mean']

    delta_ampl = delta05_coef.loc[component]['A']
   
    for depth in depths:
        if not rel:
            coef = getattr(column_coefs, 'd' + str(depth))
            value = coef.loc[component]['A']
            confidence = coef.loc[component]['A_ci']

        elif rel:
            coef = getattr(column_coefs, 'd' + str(depth))
            value = getattr(column_coefs, 'd' + str(depth) + 'rel')
            confidence = 0
            
        ci.append(confidence)
        x.append(value)

        
    fig, ax = plt.subplots()
    ax.scatter(x, depths)
    ax.errorbar(x, depths, xerr=ci, marker='o', linestyle='none', lw=0.8, capsize=3, c='k')
    ax.axhline(mld_mean, ls='--')
    ax.axhline(mld_mean + delta_mean, ls='--', c=colors[1])
    xlim = ax.get_xlim()
    ax.set_ylim(max(depths) + 2, min(depths) - 2)
    mld_mean_arr = np.full(2, mld_mean)
    plt.fill_between((xlim[0], xlim[1]), (mld_mean_arr-mld_ampl), (mld_mean_arr+mld_ampl), color=colors[0], alpha=0.35)
    plt.fill_between((xlim[0], xlim[1]), (mld_mean + delta_mean-delta_ampl), (mld_mean + delta_mean+delta_ampl), color=colors[1], alpha=0.35)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)


    norm_x = xlim[1] - xlim[0]
    norm_y = ax.get_ylim()[0] - ax.get_ylim()[1]

    ax.set_xlabel('Oscilation amplitude (ºC)')
    ax.set_ylabel('Depth')
    ax.set_title('Tidal component ' + component)
    ax.arrow(x = x_arrow, y = mld_mean - mld_ampl, dy=2*mld_ampl, dx=0, width=0.0008*norm_x,
             length_includes_head=True, head_width=0.01*norm_x, head_length=0.02*norm_y, color='k')
    ax.arrow(x = x_arrow, y = mld_mean + mld_ampl, dy=-2*mld_ampl, dx=0, width=0.0008*norm_x,
             length_includes_head=True, head_width=0.01*norm_x, head_length=0.02*norm_y, color='k')

    ax.arrow(x = x_arrow, y = mld_mean + delta_mean - delta_ampl, dy=2*delta_ampl, dx=0, width=0.0008*norm_x,
             length_includes_head=True, head_width=0.01*norm_x, head_length=0.02*norm_y, color='k')
    ax.arrow(x = x_arrow, y = mld_mean + delta_mean + delta_ampl, dy=-2*delta_ampl, dx=0, width=0.0008*norm_x,
             length_includes_head=True, head_width=0.01*norm_x, head_length=0.02*norm_y, color='k')
    ax.text(pos1[0], pos1[1], 'MLD', transform=ax.transAxes)
    ax.text(pos2[0], pos2[1], r'$\Delta_{0.05}$', transform=ax.transAxes)
    
    plt.show()

def plot_D1_and_G005(D1, G005, period=[None, None], xlim=None, ylim=None, save=None):
    slice_ = slice(*period)
    D1_ = D1.loc[slice_]
    G005_ = G005.loc[slice_]
    date = D1[slice_].index

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax1 = plt.subplots(figsize=(9.2, 3.75))
    locator = mdates.AutoDateLocator(minticks=4, maxticks=None)
    formatter = mdates.ConciseDateFormatter(locator)
    minor_locator = mdates.AutoDateLocator(minticks=6)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_minor_locator(minor_locator)
    ax1.plot(date, D1_)
    ax1.set_ylabel(r'MLD (db)', color=colors[0])  # we already handled the x-label with ax1
    #ylim = ax1.get_ylim()
    #ax1.set_ylim(ylim[1], ylim[0])
    ax1.tick_params(axis='y', colors=colors[0], which='both')


    ax2 = ax1.twinx()
    ax2.plot(date, G005_, color=colors[1])
    ax2.tick_params(axis='y', colors=colors[1], which='both')
    ax2.set_ylabel('$G_{0.05}$', color=colors[1])  # we already handled the x-label with ax1
    fig.tight_layout()
    plt.show()

def plot_multiple_profiles_ax(df, data, date_i, ylim=None):
        
    markers = ['o', 's', '^', 'v', '<', '>', 'd']
    ls = ['-', '--', ':', '-.']

    fig, ax = plt.subplots(figsize=(4.5, 4.6875))

    for (i, date_) in enumerate(date_i): 
        if isinstance(date_i, (int, np.integer)):
            data_i = data.isel(date=date_, method='nearest')
            mld = df.iloc[date_]['D1']

        else:
            data_i = data.sel(date=date_, method='nearest')
                #mld = df.loc[i, 'D1']
                
        pres_i = data_i.meassured_depth.data
        temp_i = data_i.temp.data
        date_i_str = np.datetime_as_string(data_i.date, unit='s')
        pres_i = pres_i[np.isfinite(temp_i)]
        temp_i = temp_i[np.isfinite(temp_i)]
        
        # switch statement for seting label
        match i:
            case 0:
                label_i = 't=0'
                
            case 1|2:
                label_i = 't=' + str(i) + 'M$_2$/3'
                
            case 3:
                label_i = 't=$M_2$'
                
        zz = np.linspace(1, pres_i[-1] + 5, 300)
        ax.scatter(temp_i, pres_i, marker=markers[i], label=label_i)
        ax.plot(fit_function(zz, df, date_), zz, lw=1.2, marker=' ', alpha=0.9, ls=ls[i])
        
        if ylim is None:
            ax.set_ylim(135, 0)
        else:
            ax.set_ylim(*ylim)
        ax.set_xlabel('Temperature (ºC)')
        ax.set_ylabel('Profundidad (dbar)')
    
    ax.legend()
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
        
    interact(plot_, i=int_wdgt)

def lag_correlation(var1, var2, period=[None, None]):
    slice_ = slice(*period)
    ccf_12 = ccf(var1[slice_], var2[slice_], adjusted=False)
    date = var1[slice_].index.to_numpy()
    lag_hours = np.array(np.array(date - date[0], dtype='timedelta64[s]'), dtype='int')/3600

    fig, ax = plt.subplots()
    ax.plot(lag_hours, ccf_12)
    ax.set_xlabel('lag (h)')
    ax.set_ylabel('correlation')
    
    dt = date[1] - date[0]
    dt = np.timedelta64(dt, 's')/np.timedelta64(1, 's')
    max_ccf = np.argmax(ccf_12)*dt/60/60
    min_ccf = np.argmin(ccf_12)*dt/60/60
    peaks = find_peaks(ccf_12, height=0.005, prominence=0.08)[0]
    print(lag_hours[peaks])

    ax.axvline(max_ccf, ls='--', c='grey')
    ax.axvline(min_ccf, ls='--', c='grey')
    print(min_ccf)
    print(max_ccf)
