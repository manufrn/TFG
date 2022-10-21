#!/usr/bin/env python3

'''
Perform a differential evolution search
At the surface (MLD) it adjust to a constant value (sst); deeper
it adjust to an exponential decay (seasonal pycnocline)
plus a linear decay (permanent pycnocline).
it uses the temperature, salinity, and pressure data from a matfile (or netcdf file)
'''
import tqdm
import sys
import os
import time
import scipy.io
from scipy.optimize import OptimizeResult
from numpy import *
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

printerr = sys.stderr.write

### CONSTANTS ###
postfix = "R"     # postfix for distinguishing files:

uno = 1.0
cero = 0.0

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

    pos = where(z >= D1, 1.0, 0.0)  # check if z is above or bellow MLD
    zaux = - (z - D1) *(b2 + (z - D1) *c2)
    zaux = where(zaux > LIMEXP, LIMEXP, zaux)
    zaux = where(zaux < - LIMEXP, - LIMEXP, zaux)

    return a1 + pos *(b3 *(z - D1) + a2 *(exp(zaux) - 1.0))

def calajus(indi, z, y):
    '''Estimate the fitting for each individual'''

    return sqrt(sum((y - fun(z, indi))**2) /float(y.size))

def funp(z, pob):
    '''Estimate the function for the population'''

    D1, b2, c2 = pob[:, 0], pob[:, 1], pob[:, 2]
    b3, a2, a1 = pob[:, 3], pob[:, 4], pob[:, 5]

    pos = where(z >= D1, 1.0, 0.0)
    zaux = -(z -D1) *(b2 + (z - D1) *c2)
    zaux = where(zaux > LIMEXP, LIMEXP, zaux)
    zaux = where(zaux < - LIMEXP, - LIMEXP, zaux)

    return a1 + pos *(b3 *(z -D1) +a2 *(exp(zaux) -1.0))

def muf(k, ngene):
    '''Weigth of the best individual in the recombination'''

    return 0.2 + 0.8 *(float(k) /float(ngene))**2

def calajuspob(pob, z, y):
    '''Estimate the fitting for the population'''

    np = y.size
    nindi = pob.shape[0]
    y1 = y.reshape(np, 1, 1)
    z1 = z.reshape(np, 1, 1)
    return (sqrt(sum((y1 - funp(z1, pob))**2, axis=0) /float(np))).reshape((nindi,))

# diferential evolution (aprox. derivative of difevol.f)
def miDE(f, Lmm, args, maxiter=100, popsize=60,
         tol=0.0025, mutation=0.5, recombination=0.5,
         seed=111, mu=lambda g, gmax: 0.6):

    z, y = args[0], args[1]
    #Lmm = zip(*Lmm)
    Lmin, Lmax = array(Lmm[0]), array(Lmm[1])

    nvar = size(Lmin)
    Lminr = Lmin.reshape((1, nvar))
    DL = (Lmax -Lmin).reshape((1, nvar))

    Lmin1 = Lmin.reshape((1, nvar))
    Lmax1 = Lmax.reshape((1, nvar))

    if seed is not None:
        random.seed(seed)

    pob = random.rand(popsize, nvar)
    pob = Lminr +DL *pob

    ajus = calajuspob(pob, z, y)
    ibest = ajus.argmin()
    best = pob[ibest, :].reshape(1, nvar)
    bestaj = ajus[ibest]
    pob[0, :], pob[ibest, :] = pob[ibest, :], pob[0, :]
    ajus[0], ajus[ibest] = ajus[ibest], ajus[0]

    for igen in range(maxiter):
        mua = mu(igen, maxiter)

        p1 = random.permutation(popsize)
        p2 = random.permutation(popsize)
        pobnu = (1 -mua) *pob +mua *best +mutation *(pob[p1, :] -pob[p2, :])

        pobnu = where(random.rand(popsize, nvar) < recombination, pobnu, pob)

        # limitación de bordes
        pobnu = where(pobnu < Lmin1, Lmin1, pobnu)
        pobnu = where(pobnu > Lmax1, Lmax1, pobnu)

        ajusnu = calajuspob(pobnu, z, y)

        pob = where(ajus.reshape(popsize, 1) < ajusnu.reshape(popsize, 1), pob, pobnu)
        ajus = where(ajus < ajusnu, ajus, ajusnu)
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

    if fn.endswith('.mat'):
        data = scipy.io.loadmat(fn)

        lat = data['lat'][:]
        lon = data['lon'][:]
        pres = data['pres'][:]
        temp = data['tems'][:]
        dates = data['dates'][:]
        if sal:
            sal = data['sals'][:]

        return lat, lon, pres, temp, dates

    #for Argo floats nc files
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
        month  = '%s%s' % (dateref[4], dateref[5])
        day  = '%s%s' % (dateref[6], dateref[7])

        origin = datetime.date(int(year), int(month), int(day))
        return None

    else:
        raise ValueError('Data format not recognised.')


def parse_single_date(press, tem):
    '''Parse data from a single date'''

    press = press[isfinite(tem)]
    tem = tem[isfinite(tem)]

    z, y = [], []
    for (p, t) in zip(press, tem):
        if p > ZPROF:
            break
        if p > ZUNDE or t > ZUNDE:
            continue
        z.append(p)
        y.append(t)

    if len(z) < LENZMIN:
        return(fmt % (9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99))

    z, y = array(z), array(y)

    # distinguish between temp and density profiles
    if y[0] > y[-1]:
        sign = 1.0
        ylabel= 'T'
    else:
        sign = -1.0
        ylabel = 'Rho'

    # initial limits for the parameters
    min_values, max_values = zeros(nvar), zeros(nvar)

    # límites para la profundidad de la capa de mezcla
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
    em = 9.99e99
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
    a1m = res.x[5]
    b2m = res.x[1]
    c2m = res.x[2]
    a2m = res.x[4]
    b3m = res.x[3]
    a3m = a1m -a2m

    return D1m, a1m, a2m, b2m, c2m, a3m, b3m, em



def print_results_to_file(lat, lon, dates, results_fit):
    input_fn = sys.argv[1]
    base = os.path.basename(input_fn)
    output_fn = os.path.join('Results', os.path.splitext(base)[0] + '_' + postfix + '.csv')

    print('Writing results to ' + output_fn)

    if len(list(lat)) == 1:
        lat = [float(lat) for _ in range(size(dates))]
        lon = [float(lon) for _ in range(size(dates))]

    columns = ['D1m', 'a1m', 'b2m', 'c2m', 'a2m', 'b3m', 'a3m', 'em']

    df = pd.DataFrame(results_fit, columns = columns, dtype='float')
    df.insert(0, 'Dates', dates[0])
    df.insert(1, 'lat', lat)
    df.insert(2, 'lon', lon)

    df.to_csv(output_fn, index=False)


if __name__ == '__main__':
    t_0 = time.time()

    print('Loading data...')
    lat, lon, pres, temps, dates = extract_data_from_file()

    pool_arguments = [[pres[:, 0], temps[:, i]] for i in range(size(dates))]

    print('Begining DE fit...')
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_fit = pool.starmap(parse_single_date, tqdm.tqdm(pool_arguments,
                                   total=len(pool_arguments)), chunksize=1)

    print_results_to_file(lat, lon, dates, results_fit)
    print(time.time() - t_0)

