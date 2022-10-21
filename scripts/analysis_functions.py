import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from scipy.io import loadmat
from numpy.fft import fft, fftfreq
import os
import sys

def import_data(file_name):
    results_path = f'../results/{file_name}_R.csv'
    df_fit = pd.read_csv(results_path)
    time_series_path = f'../data/thermistor_chain/AGL_Abril_2019/Time_series/{file_name}.mat'
    data = loadmat(time_series_path)

    tems = data['tems']
    pres = data['pres']

    return tems, pres, df_fit

def fit_fun(z, df):
    D1, b2, c2 = df['D1m'], df['b2m'], df['c2m']
    b3, a2, a1 = df['b3m'], df['a2m'], df['a1m']
    print('D1: {:.2f}, a1: {:.2f}, b3: {:.2E}, b2:{:.2E}, c2: {:.2E}'.format(D1, a1, b3, b2, c2))

    pos = np.where(z >= D1, 1.0, 0.0)  # check if z is above or bellow MLD
    zaux = - (z - D1) *(b2 + (z - D1) *c2)
    return a1 + pos *(b3 *(z - D1) + a2 *(np.exp(zaux) - 1.0))


def plot_fit_variable(df, variable, interval=None, save=False):

    dic = {'D1m': 'MLD (m)', 'a1m': 'SST (ºC)'}
    n = len(df[variable])
    max_date = 737888.413299
    idx = int(np.where(df['Dates'] == max_date)[0])
    var = df[variable][:idx:interval]
    dates = df['Dates'][:idx:interval]
    fig, ax = plt.subplots()
    ax.scatter(dates, var, s=8)
    ax.set_xlabel('Date')
    if variable in dic:
        ax.set_ylabel(dic[variable])
    else:
        ax.set_ylabel(variable)
    fig.tight_layout()
    plt.show()


def plot_profile_fit(df, tems, pres, number):
    temp = tems[:, number]
    zz = np.linspace(0, 200, 300)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(temp, pres, marker='o', fc='None', ec='tab:red')
    ax.axhline(df.iloc[number, 3], c='grey', ls='--')
    ax.set_ylim(pres[-1] + 10, 0)
    ax.set_xlim(9.5, 18)

    ax.plot(fit_fun(zz, df.iloc[number]), zz)
    ax.set_xlabel('Temperatura (ºC)')
    ax.set_ylabel('Profundidad (mb)')

    plt.show()


# DOESN'T WORK, DON'T RUN
def compute_physical_parameters(file_name, df, alfa=0.05):
    b2 = df['b2m']
    c2 = df['c2m']
    D = df['D1m']

    l = 2 *c2 /b2**2
    Delta = -b2 /2 /c2 *(1 - (1 - 2 *l *np.log(alfa))**0.5)

    def compute_G(D, alfa):
        G = (fit_fun(D, df) - fit_fun(D + Delta, df)) /Delta
        return G

    path = '../results/' + file_name

    df = df.assign(G95=lambda x: compute_G(x.D1m, alfa))
    df.to_csv(path, index=False)


def plot_worst_fit_profiles(df, tems, pres):
    em = df['em']
    profiles = df.index[df['em'] > 2]
    for profile in profiles:
        plot_profile_fit(df, tems, pres, profile)

def plot_multiple_profiles(df, tems, pres, profile_numbers):

    if profile_numbers != 4:
        pass

    zz = np.linspace(0, 200, 300)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6))
    axes = axes.reshape(4)

    for ax, number in zip(axes, profile_numbers):
        temp = tems[:, number]
        ax.scatter(temp, pres, marker='o', fc='None', ec='tab:red')
        ax.axhline(df.iloc[number, 3], c='grey', ls='--')
        ax.set_ylim(pres[-1] + 10, 0)
        ax.set_xlim(9.5, 18)

        ax.plot(fit_fun(zz, df.iloc[number]), zz)
        ax.set_xlabel('Temperatura (ºC)')
        ax.set_ylabel('Profundidad (mb)')

    fig.tight_layout()
    plt.show()

def animate_profile_evolution(df, tems, pres, start_number, final_number, number_plots, file_name):
    numbers = np.linspace(start_number, final_number, number_plots, dtype='int')
    zz = np.linspace(0, 200, 300)

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
    title = ax.text(0.85, 0.9, '', bbox={'facecolor': 'w', 'alpha': 0.5,
                                         'pad': 5}, transform=ax.transAxes, ha='center')

    def animate(i):
        points.set_data(tems[:, i], pres)
        line.set_data(fit_fun(zz, df.iloc[i]), zz)
        mld.set_data((9.5, 18), (df.iloc[i, 3], df.iloc[i, 3]))
        title.set_text('nº: {}'.format(i))

    ani = FuncAnimation(fig, animate, frames=numbers, interval=80)
    ani.save(f'../results/{file_name}')


def spectral_analysis(df, variable, dt=5):
    signal = df[variable]
    n = np.size(signal)
    fourier = fft(signal)
    freq = fftfreq(n, d=dt)
    indices = np.where(freq > 0)
    freq = freq[indices]
    fourier = abs(fourier[indices])
    t = np.linspace(1, n, n)
    period = 1 /freq

    fig, ax = plt.subplots()
    ax.plot(freq, fourier)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Magnitud')
    ax.set_xlim(0, 0.0001)
    plt.show()

# 2440000, 2480000

if __name__ == '__main__':
    tems, pres, df_fit = import_data('Time_Series_Abril')
    n = len(df_fit['Dates'])
    animate_profile_evolution(df_fit, tems, pres, 0, n -1, 250, 'Animación_serie_completa.mp4')
    # plot_mutiple_profiles(df_fit, tems, pres, [1, 2, 3, 4])
    # plot_fit_variable(df_fit, 'a3m', 100)
