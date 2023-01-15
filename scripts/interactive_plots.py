import numpy as np
import xarray as xr
import hvplot.xarray
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt

from analysis_routines import * 
from plotting_routines_xr import *

fit_chain = load_SHDR_fit('optimal_server_fit/AGL_20181116_fit_fci.csv')
data_chain = load_time_series_xr('processed/AGL_20181116_chain_xrcompatible.nc')

period = [datetime(2018, 11, 19, 20), datetime(2018, 11, 21, 20)]

slice_ = slice(*period)

fit_period = fit_chain.loc[period]
# data_period = data_chain.loc[period]

N = len(fit_period)

zz_ = np.linspace(0, 200, 300)
ii = range(N)

def f_zz(i):
    return zz_

def f_yy(zz, i):
    return fit_function(zz, fit_period, i)

fig, ax = plt.subplots()
controls = iplt.plot(f_zz, f_yy, i=ii)
plt.show()
