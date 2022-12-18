import numpy as np
import xarray
import hvplot.xarray
from analysis_routines import * 
from plotting_routines_xr import *

data = load_time_series_xr('processed/AGL_20181116_chain_xrcompatible.nc')
slice_ = slice(datetime(2018, 11, 28, 11), datetime(2018, 12, 22), 12)
plot = data.temp.sel(date=slice_).hvplot.quadmesh(x='date', y='depth', ylim=[200, 0], cmap='viridis')
hvplot.save(plot, 'test.html')
