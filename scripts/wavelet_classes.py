import numpy as np
import matplotlib.pyplot as plt
import pycwt
import xarray as xr
from datetime import datetime, timedelta
from analysis_routines import *
from config import data_dir
import matplotlib.gridspec as gs

class Wavelet:
    def __init__(self, signal, date, period, mother=None):
        slice_ = slice(*period)

        self.signal = signal.loc[slice_].data
        self.date = date.loc[slice_].data
        self.dt = int(self.date[1] - self.date[0]) / 1e9
        if mother is None:
            self.mother = pycwt.Morlet(6)
        else:
            self.mother = mother

    def detrend_and_normalize(self):
        signal = self.signal.copy()
        pseudo_time = range(len(signal))
        p = np.polyfit(pseudo_time, signal, 1)
        signal_notrend = signal - np.polyval(p, pseudo_time)
        std = signal_notrend.std()  # standard deviation
        var = std**2
        signal_norm = signal_notrend / std

        self.signal = signal_norm
        self.orig_signal = signal
        self.var = var

        del signal


    def wavelet_spectrum(self):
        n = self.signal.size
        s0 = 2 * self.dt
        dj = 0.25
        j = int(1/dj * np.log2(n * self.dt / s0))

        wave, scales, freqs, coi, fft, fft_freqs = pycwt.cwt(self.signal,
                                                             self.dt,
                                                             dj,
                                                             s0,
                                                             j,
                                                             self.mother)

        power = np.abs(wave) ** 2
        fft_power = np.abs(fft) ** 2
        period = 1 / freqs
        self.power = power
        self.scales = scales
        self.period = period
        self.coi = coi
        self.fft_power = fft_power
        self.fft_freqs = fft_freqs
        del power, scales, period, coi, fft_power, fft_freqs

    def significance(self):
        n = self.signal.size
        alpha, _, _ = pycwt.ar1(self.signal)

        signif, fft_theor = pycwt.significance(1.0, 
                                               self.dt,
                                               self.scales,
                                               0, 
                                               alpha,
                                               significance_level=0.95,
                                               wavelet=self.mother)

        sig95 = np.ones([1, n]) * signif[:, None]
        sig95 = self.power / sig95

        glbl_power = self.power.mean(axis=1)
        dof = n - self.scales  # correction for padding at edges
        glbl_signif, tmp = pycwt.significance(self.var, self.dt, self.scales, 1, 
                                              alpha, significance_level=0.95, 
                                              dof=dof, wavelet=self.mother)

        self.power /= self.scales[:, None]
        self.sig95 = sig95
        self.glbl_power = glbl_power
        self.glbl_signif = glbl_signif
        self.fft_theor = fft_theor

        
    def compute(self):
        self.detrend_and_normalize()
        self.wavelet_spectrum()
        self.significance()

    def plot_spectrum_ax(self, ax, norm_levels=2**7, skip=10, lw=0.5):
        date = self.date.copy()
        period = self.period.copy()

        levels = np.array([0.25, 0.5, 1, 2, 4, 8])/norm_levels

        period /= 60 # minutes for plot
        
        im = ax.contourf(date, np.log2(period), np.log2(self.power),
                          np.log2(levels), extend='both', cmap='viridis')

        extent = [date.min(), date.max(), min(period), max(period)]

        ax.contour(date, np.log2(period)[skip:], self.sig95[skip:],
                    [-99, 1], colors='k', linewidths=lw, extent=extent)

        dt = date[1] - date[0]
        ax.fill(np.concatenate([date, date[-1:] + dt, date[-1:] + dt,
                                date[:1] - dt, date[:1] - dt]),
                 np.concatenate([np.log2(self.coi), [1e-9], np.log2(period[-1:]),
                                 np.log2(period[-1:]), [1e-9]]),
                 'k', alpha=0.3, hatch='x')

        yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), 
                                np.ceil(np.log2(period.max())))
        
        ax.set_ylim(np.log2(period).max(), np.log2(period).min())

        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.set_ylabel('Periodo (min)')


    def complete_plot(self, units=None, skip=10, norm_levels=2**7, hlines=None):
        date = self.date.copy()
        period = self.period.copy()

        levels = np.linspace(0.25, 8, 60)/norm_levels


        gs = plt.GridSpec(2, 3, width_ratios=[1, 1, 0.5], height_ratios=[1,3])
        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(date, self.signal)
        if units is not None:
            ax1.set_ylabel(units)

        period /= 60 # minutes for plot
        
        ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
        im = ax2.contourf(date, np.log2(period), np.log2(self.power),
                          np.log2(levels), extend='both', cmap='viridis')

        extent = [date.min(), date.max(), min(period), max(period)]

        ax2.contour(date, np.log2(period)[skip:], self.sig95[skip:],
                    [-99, 1], colors='k', linewidths=0.5, extent=extent)

        dt = date[1] - date[0]
        ax2.fill(np.concatenate([date, date[-1:] + dt, date[-1:] + dt,
                                date[:1] - dt, date[:1] - dt]),
                 np.concatenate([np.log2(self.coi), [1e-9], np.log2(period[-1:]),
                                 np.log2(period[-1:]), [1e-9]]),
                 'k', alpha=0.3, hatch='x')

        yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), 
                                np.ceil(np.log2(period.max())))
        
        ax2.set_ylim(np.log2(period).max(), np.log2(period).min())

        ax2.set_yticks(np.log2(yticks))
        ax2.set_yticklabels(yticks)
        ax2.set_ylabel('Periodo (min)')
        ax2.set_title('b) Wavelet power spectrum')
        
        
        ax3 = fig.add_subplot(gs[1, 2], sharey=ax2)
        ax3.plot(self.glbl_signif, np.log2(period), 'k--')
        ax3.plot(self.var * self.fft_theor, np.log2(period), '--', 
                 color='#cccccc')
        ax3.plot(self.var * self.fft_power, np.log2(1./self.fft_freqs), 
                 '-', color='#cccccc', linewidth=1.)
        ax3.plot(self.var * self.glbl_power, np.log2(period), 'k-', linewidth=1.5)
        ax3.set_title('c) Global wavelet spectrum')
        # ax3.set_xlabel(r'power [({})^2]'.format(units))
        ax3.set_xlim([0, self.glbl_power.max() * self.var * 1.25])
        ax3.set_yticks(np.log2(yticks))
        ax3.set_yticklabels(yticks)
        ax3.set_xlabel(u'Power (ºC\u00b2)')
        plt.setp(ax3.get_yticklabels(), visible=False)

        if hlines is not None:
            for i in hlines:
                ax3.axhline(i, color='k', lw=2)


        fig.tight_layout()
        plt.show()



