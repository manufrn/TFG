#!/usr/bin/env python3

'''
Fit an upper ocean temperature profile time series to a functional form consisting
of a constant value (SST) at the surface (MLD); an exponential decay (seasonal pynoclune)
and linear decay (permanent pynocline). The fit is computed performing a differential
evolution search. The program makes use of the multiprocessing module to scale up the
computationto multiple cores. Supported data files: .mat and .nc

USAGE: run file from terminal with time series as argument.
e.g: $ python3 DE-time-series-manu-multithread.py data/Example_Time_Series.mat
A .csv results file will be created in the results folder, named as the Time Series
file it was created from + an appendix to easily distinguish it
'''

import sys
import os
import time
import tqdm
import multiprocessing as mp
from datetime import date
import configparser
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import OptimizeResult
printerr = sys.stderr.write

results_folder = 'results'

### CONSTANTS ###

postfix = "R"     # postfix for distinguishing files:

nvar = 6          # number of variables
nindi = 10 *nvar   # number of individuals
ngene = 1000      # number of generations
BC2m = 0.5        # Maximum value for b2 anc c2 (about stratification in the seasonal pycnocline)

CR = 0.5          # Cross probablility
FF = 0.5          # Mutation factor

ZPLAT = 100.0     # minimum depth of the profiles
ZPROF = 300.0     # maximum depth of the fitting
ZUNDE = 99990.0   # to tag 'no data' in netcdf
LENZMIN = 10      # minimum number of (dens,depths) observations in the profile

LIMEXP = 100.0


### FUNCTION DEFINITION ###

def fun(z, indi):
    '''Estimate the function for one individual'''

    D1, b2, c2 = indi[0], indi[1], indi[2]
    b3, a2, a1 = indi[3], indi[4], indi[5]

    pos = np.where(z >= D1, 1.0, 0.0)  # check if z is above or bellow MLD
    zaux = - (z - D1) *(b2 + (z - D1) *c2)
    zaux = np.where(zaux > LIMEXP, LIMEXP, zaux)
    zaux = np.where(zaux < - LIMEXP, - LIMEXP, zaux)

    return a1 + pos *(b3 *(z - D1) + a2 *(np.exp(zaux) - 1.0))

def calajus(indi, z, y):
    '''Estimate the fitting for each individual'''

    return np.sqrt(np.sum((y - fun(z, indi))**2) /float(y.size))

def funp(z, pob):
    '''Estimate the function for the population'''

    D1, b2, c2 = pob[:, 0], pob[:, 1], pob[:, 2]
    b3, a2, a1 = pob[:, 3], pob[:, 4], pob[:, 5]

    pos = np.where(z >= D1, 1.0, 0.0)
    zaux = -(z -D1) *(b2 + (z - D1) *c2)
    zaux = np.where(zaux > LIMEXP, LIMEXP, zaux)
    zaux = np.where(zaux < - LIMEXP, - LIMEXP, zaux)

    return a1 + pos *(b3 *(z -D1) +a2 *(np.exp(zaux) -1.0))

def muf(k, ngene):
    '''Weigth of the best individual in the recombination'''

    return 0.2 + 0.8 *(float(k) /float(ngene))**2

def calajuspob(pob, z, y):
    '''Estimate the fitting for the population'''

    pop_number = y.size
    nindi = pob.shape[0]
    y1 = y.reshape(pop_number, 1, 1)
    z1 = z.reshape(pop_number, 1, 1)
    return (np.sqrt(np.sum((y1 - funp(z1, pob))**2, axis=0) /float(pop_number))).reshape((nindi,))

def miDE(f, Lmm, args, maxiter=100, popsize=60,
         tol=0.0025, mutation=0.5, recombination=0.5,
         seed=111, mu=lambda g, gmax: 0.6):
    '''Diferential evolution alogorithm (aprox. derivative of difevol.f)'''
    z, y = args[0], args[1]
    Lmin, Lmax = np.array(Lmm[0]), np.array(Lmm[1])

    nvar = np.size(Lmin)
    Lminr = Lmin.reshape((1, nvar))
    DL = (Lmax -Lmin).reshape((1, nvar))

    Lmin1 = Lmin.reshape((1, nvar))
    Lmax1 = Lmax.reshape((1, nvar))

    if seed is not None:
        np.random.seed(seed)

    pob = np.random.rand(popsize, nvar)
    pob = Lminr + DL *pob

    ajus = calajuspob(pob, z, y)
    ibest = ajus.argmin()
    best = pob[ibest, :].reshape(1, nvar)
    bestaj = ajus[ibest]
    pob[0, :], pob[ibest, :] = pob[ibest, :], pob[0, :]
    ajus[0], ajus[ibest] = ajus[ibest], ajus[0]

    for igen in range(maxiter):
        mua = mu(igen, maxiter)

        p1 = np.random.permutation(popsize)
        p2 = np.random.permutation(popsize)
        pobnu = (1 -mua) *pob +mua *best +mutation *(pob[p1, :] -pob[p2, :])

        pobnu = np.where(np.random.rand(popsize, nvar) < recombination, pobnu, pob)

        # limitación de bordes
        pobnu = np.where(pobnu < Lmin1, Lmin1, pobnu)
        pobnu = np.where(pobnu > Lmax1, Lmax1, pobnu)

        ajusnu = calajuspob(pobnu, z, y)

        pob = np.where(ajus.reshape(popsize, 1) < ajusnu.reshape(popsize, 1), pob, pobnu)
        ajus = np.where(ajus < ajusnu, ajus, ajusnu)
        ibest = ajus.argmin()
        best = pob[ibest, :].reshape(1, nvar)
        bestaj = ajus[ibest]

        pob[0, :], pob[ibest, :] = pob[ibest, :], pob[0, :]
        ajus[0], ajus[ibest] = ajus[ibest], ajus[0]

        if ajus.mean() *tol / ajus.std() > 1:
            break

    return OptimizeResult(fun=bestaj, x=best.reshape(nvar))

def extract_data_from_file(sal=False):
    if len(sys.argv) != 2:
        printerr("Error: A file name must be indicated\n")
        exit(1)

    fn = sys.argv[1]

    # for .mat files
    if fn.endswith('.mat'):
        data = scipy.io.loadmat(fn)

        lat = data['lat'][0]
        lon = data['lon'][0]
        pres = data['pres']
        temp = data['tems']
        dates = data['dates']
        if sal:
            sal = data['sals']

        return lat, lon, pres, temp, dates

    # for Argo floats .nc files
    elif fn.endswith('.nc'):
        data = netcdf.NetCDFile(fn, 'r')
        lat = data.variables['LATITUDE']
        lon = data.variables['LONGITUDE']
        pres = data.variables['PRES_ADJUSTED']
        temp = data.variables['TEMP_ADJUSTED']
        juld = data.variables['JULD']  # fecha desde ff.variables['REFERENCE_DATE_TIME']
        plat = data.variables['PLATFORM_NUMBER']
        dateref = data.variables['REFERENCE_DATE_TIME']

        year = '%s%s%s%s' % (dateref[0], dateref[1], dateref[2], dateref[3])
        month = '%s%s' % (dateref[4], dateref[5])
        day = '%s%s' % (dateref[6], dateref[7])

        origin = datetime.date(int(year), int(month), int(day))
        return None

    else:
        raise ValueError('Data format not recognised.')


def fit_profile(press, tem):
    '''Parse data from a single date'''

    press = press[np.isfinite(tem)]
    tem = tem[np.isfinite(tem)]

    z, y = [], []
    for (p, t) in zip(press, tem):
        if p > ZPROF:
            break
        if p > ZUNDE or t > ZUNDE:
            continue
        z.append(p)
        y.append(t)

    # if profile has not enough datapoints, return 9999.99 for all parametres
    if len(z) < LENZMIN:
        return 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.9

    z, y = np.array(z), np.array(y)

    # distinguish between temp and density profiles
    if y[0] > y[-1]:
        sign = 1.0
        ylabel= 'T'
    else:
        sign = -1.0
        ylabel = 'Rho'

    # initial limits for the parameters
    min_values, max_values = np.zeros(nvar), np.zeros(nvar)

    # limits for the diferent parametres
    min_values[0] = 1.0
    max_values[0] = z[-1]

    y_max = y.max()
    y_min = y.min()

    min_values[1] = 0.0
    max_values[1] = BC2m

    min_values[2] = 0.0
    max_values[2] = BC2m

    min_values[3] = 0.0 if z[-1] < ZPLAT else - abs((y[-1] - y[0] /(z[-1] -z[0])))
    max_values[3] = 0.0

    if sign < 0.0:
        min_values[3], max_values[3] = max_values[3], - min_values[3]

    min_values[4] = 0.0
    max_values[4] = y_max - y_min

    min_values[5] = y_min
    max_values[5] = y_max

    # first pass
    res1 = miDE(calajuspob, (min_values, max_values), args=(z, y),
                maxiter=ngene, popsize=nindi, tol=0.00025,
                mutation=FF, recombination=CR, mu=muf)

    # final pass with delta coding
    v_min, v_max = 0.85 *res1.x, 1.15 *res1.x
    for i in range(nvar):
        min_values_d = min(v_min[i], v_max[i])
        max_values_d = max(v_min[i], v_max[i])
        min_values[i] = max(min_values[i], min_values_d)
        max_values[i] = max(max_values[i], max_values_d)

    resd = miDE(calajuspob, (min_values, max_values), args=(z, y),
                maxiter=ngene, popsize=nindi, tol=0.00025,
                mutation=FF, recombination=CR, mu=muf)

    if res1.fun < resd.fun:
        res = res1
        em = res1.fun

    else:
        res = resd
        em = resd.fun

    D1m = res.x[0]
    b2m = res.x[1]
    c2m = res.x[2]
    b3m = res.x[3]
    a2m = res.x[4]
    a1m = res.x[5]
    a3m = a1m - a2m

    return D1m, a1m, a2m, b2m, c2m, a3m, b3m, em

def save_results_to_file(lat, lon, dates, results_fit):
    '''Save to results to a .csv file in results_folder'''

    input_fn = sys.argv[1]
    base_fn = os.path.basename(input_fn)
    output_fn = os.path.join(results_folder, '{}_{}.csv'.format(os.path.splitext(base_fn)[0],
                                                                postfix))

    print('Writing results to {}'.format(output_fn))

    # if time series only has one spatial coordinate, expand it to include it in dataframe
    if np.size(lat) == 1:
        lat = [float(lat) for _ in range(np.size(dates))]
        lon = [float(lon) for _ in range(np.size(dates))]

    columns = ['D1m', 'a1m', 'a2m', 'b2m', 'c2m', 'a3m', 'b3m', 'em']

    # Convert results list to pd.Dataframe and save as .csv
    results_fit = pd.DataFrame(results_fit, columns=columns, dtype='float')
    results_fit.insert(0, 'Dates', dates[0])
    results_fit.insert(1, 'lat', lat)
    results_fit.insert(2, 'lon', lon)

    results_fit.to_csv(output_fn, index=False)


def main():
    t_0 = time.time()

    print('Loading data...')

    lat, lon, pres, temps, dates = extract_data_from_file()
    pool_arguments = [[pres[:, 0], temps[:, i]] for i in range(np.size(dates))]

    print('Begining DE fit...')
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_fit = pool.starmap(fit_profile, tqdm.tqdm(pool_arguments,
                                                          total=len(pool_arguments)), chunksize=1)

    save_results_to_file(lat, lon, dates, results_fit)
    print('Elapsed time: {:.2f} seconds'.format(time.time() - t_0))


if __name__ == '__main__':
    main()
