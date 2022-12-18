import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq, ifft, rfft
from scipy.signal.windows import boxcar
from scipy.signal import welch, filtfilt, butter, lfilter, sosfilt
from utide import solve


#### UTILITIES ####

def smooth(y, n):
    y_smoothed = np.convolve(y, boxcar(n), mode='valid') / n
    return y_smoothed

def detrend(x):
    n = len(x)
    t = np.arange(n)
    p = np.polyfit(t, x, 1)
    x_detrended = x - np.polyval(p, t)
    return x_detrended


def period_to_freq(period, period_units):
    if period_units == 's':
        freq = 1/period
        
    elif period_units == 'm':
        freq = 1/period/60
    
    elif period_units == 'h':
        freq = 1/period/60/60
    
    return freq

#### FFT ####

def spectrum(x, dt, n_smooth):
    '''Perform fft of series with spacing dt smoothin the 
    series with a quadratic window and block smoothing the 
    result with blocks of lenght n_smooth
    '''
    n = len(x)
    x = detrend(x)
    #win_weights = quadwin(n)
    #x *= win_weights
    
    pslice = slice(1, n//2)
    freqs = fftfreq(n, d=dt)[pslice]
    amplitude = rfft(x)[pslice]
    
    power = 2 * np.abs(amplitude)**2 / n**2
    psd = power * dt * n # power spectral density
    #psd *= n / (win_weights**2).sum()
    #power *= n**2 / win_weights.sum()**2
    
    freqs = smooth(freqs, n_smooth)
    psd = smooth(psd, n_smooth)
    power = smooth(power, n_smooth)
    
    return freqs, amplitude, power, psd

def windowed_spectrum(x, dt, window_time, n_smooth, window='hann'):
    ''' Perform a windowed fourier transmor over time series x with
    spacing dt. Smooth out the results.
    '''
    N = len(x)
    window_n = window_time/dt
    freqs, psd = welch(x, fs=1/dt, nperseg=window_n, window=window, detrend='linear')
    #freqs = smooth(freqs, n_smooth)
    #psd = smooth(psd, n_smooth)
    
    return freqs, psd

def plot_spectrum(freqs, pxx, units, lims=None, x='freqs', vlines=None):
    period = 1/freqs
    fig, ax = plt.subplots()

    if x == 'freqs':
        ax.loglog(freqs, pxx)
        ax.set_xlabel(r'frecuency ({})'.format(units))
        
    elif x == 'period':
        ax.loglog(period, pxx)
        ax.set_xlabel(r'Period ({})'.format(units))

    if lims != None:
        ax.set_xlim(*lims)

    if vlines != None:
        for i in vlines:
            ax.axvline(i, ls='--', c='grey')

    ax.set_ylabel(r'Power spectral density'.format(units))
    plt.show()


#### FILTERING ####

def bandstop_filter(signal, sampling_rate, lowcut, highcut, order=4):
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandstop', output='sos')

    filtered_signal = sosfilt(sos, signal)
    return filtered_signal


#### UTIDE ####

def coef_dataframe(date, value, lat=43.789):
    coef = solve(date, value, lat=lat, nodal=False, verbose=False)
    columns = ['A', 'A_ci', 'g', 'g_ci']
    data_dict = dict((k, coef[k]) for k in ['name', 'PE', 'SNR', 'A', 'A_ci', 'g', 'g_ci'] if k in coef)
    df = pd.DataFrame(data_dict)
    df.insert(1, 'period', 1/coef.aux['frq'])
    df.set_index('name', inplace=True)
    df.attrs['mean'] = coef['mean']
    df.attrs['slope'] = coef['slope']
    return df

class TidalComponentsFit:
    def __init__(self, D1, b2, c2, a2):
        self.D1 = D1
        self.b2 = b2
        self.c2 = c2
        self.a2 = a2
        
    @classmethod
    def compute(cls, df, variables=None, period=[None, None], lat=43.789):
        
        if len(period) != 3:
            slice_ = slice(period[0], period[1], 6)
        
        else:
            slice_ = slice(*period)
        
        df = df[slice_]
        date = df.index
        
        if variables is None:
            variables = ['D1', 'b2', 'c2', 'a2']
        
        if 'D1' in variables:
            D1 = df['D1'].to_numpy()
            df_D1 = coef_dataframe(date, D1, lat=lat)
        
        else:
            df_D1 = None
        
        if 'b2' in variables:
            b2 = df['b2'].to_numpy()
            df_b2 = coef_dataframe(date, b2, lat=lat)
        
        else:
            df_b2 = None
            
        if 'c2' in variables:
            c2 = df['c2'].to_numpy()
            df_c2 = coef_dataframe(date, c2, lat=lat)
        
        else:
            df_c2 = None
            
        if 'a2' in variables:
            a2 = df['a2'].to_numpy()
            df_a2 = coef_dataframe(date, a2, lat=lat)
        
        else:
            df_a2 = None
            
        return cls(df_D1, df_b2, df_c2, df_a2)
    
    def clean(self):
        
        for var in ['D1', 'b2', 'c2', 'a2']:
            df_var = getattr(self, var)
            if df_var is not None:
                
                df_var = df_var[df_var['SNR'] > 2.0]
                df_var = df_var[df_var['A'] > df_var['A_ci']]
                df_var = df_var[df_var['PE'] > 1]
                setattr(self, var, df_var)
