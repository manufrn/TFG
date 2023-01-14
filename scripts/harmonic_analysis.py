import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq, ifft, rfft
from scipy.signal.windows import boxcar
from scipy.signal import welch, filtfilt, butter, lfilter, sosfilt, sosfiltfilt
from utide import solve
from multitaper import MTSpec, MTCross


#### UTILITIES ####

def smooth(y, n, mode='valid'):
    y_smoothed = np.convolve(y, boxcar(n), mode=mode) / n
    return y_smoothed

def detrend(x):
    n = len(x)
    t = np.arange(n)
    p = np.polyfit(t, x, 1)
    x_detrended = x - np.polyval(p, t)
    return x_detrended

def detrend_normalize(signal):
    signal_detrended = detrend(signal)
    signal_detrend_norm = signal_detrended/np.std(signal_detrended)
    return signal_detrend_norm


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
    dof = 2*n_smooth
    
    return freqs, amplitude, power, psd, dof

def windowed_spectrum(x, dt, window_time, n_smooth):
    ''' Perform a windowed fourier transmor over time series x with
    spacing dt. Smooth out the results.
    '''
    # x = smooth(x, n_smooth)

    N = len(x)
    window_n = window_time/dt
    window = ('kaiser', 3.0)
    freqs, psd = welch(x, fs=1/dt, nperseg=window_n, window=window, detrend='linear')
    freqs = smooth(freqs, n_smooth)
    psd = smooth(psd, n_smooth)

    dof = (N//(window_n//2) - 1)*2*n_smooth   

    return freqs, psd, dof


def multitapping_spectrum(x, dt, n_smooth=None, nw=3.5, kspec=4):
    spectrum = MTSpec(x=x, nw=nw, kspec=kspec, dt=dt, iadapt=0)
    freq, psd = spectrum.rspec()

    if n_smooth is None:
        dof = spectrum.se[0]

    else:
        psd = smooth(np.squeeze(psd), n_smooth)
        freqs = freq[n_smooth//2:-n_smooth//2+1]
        dof = spectrum.se[0]*n_smooth

    del spectrum

    return freq, psd, dof


def smooth_spectrum(freq, pxx, dof, n_smooth):
    pxx_s = smooth(pxx, n_smooth)
    freq_s = freq[n_smooth//2:-n_smooth//2+1]
    dof_s = dof*n_smooth
    return freq_s, pxx_s, dof_s


#### FILTERING ####

def bandstop_filter(signal, date, sampling_rate, lowcut, highcut, order=4):
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandstop', output='sos')

    filtered_signal = sosfiltfilt(sos, signal)
    series_filtered_signal = pd.Series(filtered_signal, index=date)
    return series_filtered_signal


def bandpass_filter(signal, date, sampling_rate, lowcut, highcut, order=4):
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')

    filtered_signal = sosfiltfilt(sos, signal)
    series_filtered_signal = pd.Series(filtered_signal, index=date)
    return series_filtered_signal

def lowpass_filter(signal, date, sampling_rate, highcut, order=4):
    # signal = np.pad(signal, (500, 500), 'constant', constant_values=(0, 0))
    nyq = 0.5 * sampling_rate
    high = highcut / nyq
    sos = butter(order, high, btype='lowpass', output='sos', analog=False)

    filtered_signal = sosfiltfilt(sos, signal)
    # filtered_signal = filtered_signal[1000:-500]
    series_filtered_signal = pd.Series(filtered_signal, index=date)
    return series_filtered_signal

def highpass_filter(signal, date, sampling_rate, lowcut, order=4):
    # signal = np.pad(signal, (500, 500), 'constant', constant_values=(0, 0))
    nyq = 0.5 * sampling_rate
    high = lowcut / nyq
    sos = butter(order, high, btype='highpass', output='sos', analog=False)

    filtered_signal = sosfiltfilt(sos, signal)
    # filtered_signal = filtered_signal[1000:-500]
    series_filtered_signal = pd.Series(filtered_signal, index=date)
    return series_filtered_signal




#### UTIDE ####

def coef_dataframe(value, date=None, period=[None, None], n_smooth=None, lat=43.789):

    if date is None:
        slice_ = slice(*period)

        if isinstance(value, pd.core.series.Series ):
            date = np.asarray(value.loc[slice_].index)
            value = np.squeeze(value.loc[slice_].to_numpy())

        elif isinstance(value, xr.DataArray):
            date = value.loc[slice_].indexes['date']
            value = value.loc[slice_].data

    else:
        slice_ = slice(*period)
        value = np.asarray(value[slice_])
        date = date[slice_]

    if n_smooth is not None:
        value = smooth(value, n_smooth)
        date = date[n_smooth//2:len(value)+n_smooth//2]


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
        
        
        if variables is None:
            variables = ['D1', 'a2']
        
        if 'D1' in variables:
            D1 = df['D1']
            df_D1 = coef_dataframe(D1, period=period, lat=lat)
        
        else:
            df_D1 = None
        
        if 'b2' in variables:
            b2 = df['b2']
            df_b2 = coef_dataframe(b2, period=period, lat=lat)
        
        else:
            df_b2 = None
            
        if 'c2' in variables:
            c2 = df['c2']
            df_c2 = coef_dataframe(c2, period=period, lat=lat)
        
        else:
            df_c2 = None
            
        if 'a2' in variables:
            a2 = df['a2']
            df_a2 = coef_dataframe(a2, period=period, lat=lat)
        
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

class column_coefs:
    def __init__(self, depths):
        self.depths = depths

    def compute(self, data, period=[None, None, 6], lat=43.789):
        slice_ = slice(*period)
        for depth in self.depths:
            temp = data.temp.sel(date=slice_, depth=depth)
            coef = coef_dataframe(temp, period=period, lat=lat)
            setattr(self, 'd' + str(depth), coef)
            
    def clean(self):
        for depth in self.depths:
            coef = getattr(self, 'd' + str(depth))
            coef = coef[coef['SNR'] > 2.0]
            coef = coef[coef['A'] > coef['A_ci']]
            coef = coef[coef['PE'] > 1]
            setattr(self, 'd' + str(depth), coef)        

