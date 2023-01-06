import numpy as np
import matplotlib.pyplot as plt
import pycwt
import sys
import xarray as xr
from datetime import datetime, timedelta
from analysis_routines import *
from harmonic_analysis import lowpass_filter
from config import data_dir
import matplotlib.gridspec as gs
from scipy.signal.windows import boxcar
from scipy.stats import chi2

def detrend_and_normalize(temp, time):
    pseudo_time = range(len(temp))
    pseudo_time = time
    p = np.polyfit(pseudo_time, temp, 1)
    temp_notrend = temp - np.polyval(p, pseudo_time)
    std = temp_notrend.std()  # standard deviation
    var = std ** 2  # variance
    temp_norm = temp_notrend / std

    return temp_norm, var


def power_of_two(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                return 2 ** i
    else:
        return 1


def wavelet_spectrum(temp, mother, dt, dj):
    n = temp.size
    s0 = 2 * dt
    j = int(1/dj * np.log2(n*dt/s0))

    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(temp, dt, dj, s0, j, mother)
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs
    return power, scales, period, coi, fft_power, fftfreqs


def significance(temp, power, mother, dt, scales, var):

    n = temp.size
    alpha, _, _ = pycwt.ar1(temp)
    signif, fft_theor = pycwt.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
    sig95 = np.ones([1, n]) * signif[:, None]
    sig95 = power / sig95

    glbl_power = power.mean(axis=1)
    dof = n - scales  # correction for padding at edges
    glbl_signif, tmp = pycwt.significance(var, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)
    return sig95, glbl_power, glbl_signif, fft_theor


def complete_plot(temp, time, power, period, coi, sig95, levels, dt):
    time = np.array([datetime.fromtimestamp(t) for t in time])
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 8), GridSpec_kw={'height_ratios': [1, 3]}, sharex=True)
    ax1.plot(time, temp, 'k', lw=1)

    ax2.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
                extend='both', cmap='viridis')
    
    extent = [time.min(), time.max(), min(period), max(period)]
    
    ax2.contour(time, np.log2(period), sig95, [-99, 1], colors='k', linewidths=0.5,
               extent=extent)
    ax2.set_ylim(max(np.log2(period)), min(np.log2(period)))
    dt = timedelta(seconds=dt)
    ax2.fill(np.concatenate([time, time[-1:] + dt, time[-1:] + dt,
                           time[:1] - dt, time[:1] - dt]),
        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                           np.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')

    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    ax2.set_yticks(np.log2(yticks))
    ax2.set_yticklabels(yticks)
    fig.tight_layout()
    plt.show()

def new_complete_plot(temp, time, power, period, coi, sig95, levels, glbl_power, glbl_signif, fft_theor, fftfreqs, fft_power, var, ylim=None):
    dt = int(time[1] - time[0])

    # dt = int(np.asarray((time[1] - time[0]), dtype='timedelta64[s]').item().total_seconds())
    
    plt.rcParams.update({'font.size': 12})
    time = np.array([datetime.fromtimestamp(t) for t in time])
    gs = plt.GridSpec(2, 3, width_ratios=[1,1, 0.5], height_ratios=[1, 3])
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time, temp, 'k', lw=1)
    # ax1.set_title('a) Water temperature at depth {:.0f} m'.format(depth))
    ax1.set_ylabel('Temperature (ºC)')

    period /= 60
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
    ax2.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
                extend='both', cmap='viridis')
    extent = [time.min(), time.max(), min(period), max(period)]
    
    ax2.contour(time, np.log2(period), sig95, [-99, 1], colors='k', linewidths=0.5,
               extent=extent)

    # dt = ntimedelta64(dt, 's')
    dt = timedelta(seconds=dt)
    ax2.fill(np.concatenate([time, time[-1:] + dt, time[-1:] + dt,
                           time[:1] - dt, time[:1] - dt]),
        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                           np.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')


    if ylim is not None:
        ax2.set_ylim(*ylim)
        yticks = 2 ** np.arange(np.log2(ylim[1]), np.log2(ylim[0]))

    else:
        ax2.set_ylim(max(np.log2(period)), min(np.log2(period)))
        yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))

    ax2.set_yticks(np.log2(yticks))
    ax2.set_yticklabels(yticks)
    ax2.set_ylabel('Period (min)')
    ax2.set_title('b) Wavelet power spectrum')

    ax3 = fig.add_subplot(gs[1, 2], sharey=ax2)
    ax3.plot(glbl_signif, np.log2(period), 'k--')
    ax3.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
    ax3.plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
        linewidth=1.)
    ax3.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
    ax3.set_title('c) Global wavelet spectrum')
    # ax3.set_xlabel(r'power [({})^2]'.format(units))
    ax3.set_xlim([0, glbl_power.max() * var * 1.25])
    ax3.set_yticks(np.log2(yticks))
    ax3.set_yticklabels(yticks)
    ax3.set_xlabel(u'Power (ºC\u00b2)')
    plt.setp(ax3.get_yticklabels(), visible=False)

    fig.tight_layout()
    # fig.savefig('Hey.png', dpi=100)
    plt.show()

def smooth(y, n):
    y_smoothed = np.convolve(y, boxcar(n), mode='valid')
    return y_smoothed


def wavelet_power_spectrum(variable, date, period=[None, None, 6], ylim=None, norm_levels=2**7):

    slice_ = slice(*period)

    if isinstance(variable, xr.core.dataarray.DataArray):
        variable = variable.loc[slice_].data
        date = date.loc[slice_].data

    elif isinstance(variable, pd.Series):
        variable = variable.loc[slice_].to_numpy()
        date = date.loc[slice_].to_numpy()

    elif isinstance(variable, np.ndarray):
        if period[0] is None:
            i_0 = 0

        else:   
            i_0 = date.get_loc(period[0])

        if period[1] is None:
            i_f = len(variable)

        else:
            i_f = date.get_loc(period[1])
        date = date[i_0:i_f:period[2]].to_numpy()
        variable = variable[i_0:i_f:period[2]]



    mother = pycwt.Morlet(6)
    dt = np.asarray((date[1] - date[0]), dtype='timedelta64[s]').item().total_seconds()
    dj = 0.25

    date_seconds = (date - np.datetime64('1970-01-01T00:00:00Z'))/ np.timedelta64(1, 's')
    variable_norm, var = detrend_and_normalize(variable, date_seconds)


    power, scales, period, coi, fft_power, fft_freqs = wavelet_spectrum(variable_norm, mother,
                                                                        dt, dj)
    
    sig95, glbl_power, glbl_signif, fft_theor = significance(variable_norm, power, mother, dt, scales, var)

    power /= scales[:, None]

    levels = np.array([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8])/norm_levels
    new_complete_plot(variable, date_seconds, power, period, coi, sig95, levels, glbl_power, glbl_signif, fft_theor, fft_freqs, fft_power, var, ylim=ylim)

    
if __name__ == '__main__':
     
    
    temp, pres, time = load_time_series('processed/AGL_20181116_chain.nc', convert_date=False)
    date = np.array([datetime.utcfromtimestamp(i) for i in time])

    # i_0 = date_to_idx(date, datetime(2018, 11, 22, 20))
    # i_f = date_to_idx(date, datetime(2018, 11, 23, 8))
    i_0 = date_to_idx(date, datetime(2018, 11, 16, 11))
    i_f = date_to_idx(date, datetime(2018, 11, 25, 12))
    temp = temp[i_0:i_f:6, 8]


    time = time[i_0:i_f:6]
    depth = pres[0, 8]
    # url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
    # temp = np.genfromtxt(url, skip_header=19)
    # title = 'NINO3º Sea Surface Temperature'
    # label = 'NINO3 SST'
    # units = 'degC'
    # t0 = 1871.0
    # dt = 0.25  # In years
    # N = temp.size
    # time = np.arange(0, N) * dt + t0
    mother = pycwt.Morlet(6)
    dt = float(time[1] - time[0])
    dj = 0.25

    temp_norm, var = detrend_and_normalize(temp, time)
    # temp_norm = temp
    var = temp.std()**2

    # temp = smooth(temp, 3)


    power, scales, period, coi, fft_power, fft_freqs = wavelet_spectrum(temp_norm, mother,
                                                                        dt, dj)
    
    sig95, glbl_power, glbl_signif, fft_theor = significance(temp_norm, power, mother, dt, scales, var)

    power /= scales[:, None]

    levels = np.array([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8])/2**7
    new_complete_plot(temp, time, power, period, coi, sig95, levels, glbl_power, glbl_signif, fft_theor, fft_freqs)
