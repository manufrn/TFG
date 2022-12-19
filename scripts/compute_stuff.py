import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from numba import njit, prange
import multiprocessing as mp
import tqdm as tqdm
from config import data_dir
from analysis_routines import * 
from ploting_routines import *


import dask.dataframe as dd
from dask.diagnostics import ProgressBar
pbar = ProgressBar()                
pbar.register()

print('Loading time series...')
# temp, pres, date = load_time_series('processed/AGL_20181116_chain.nc')
# df_ci = load_SHDR_fit('optimal_server_fit/AGL_20181116_fit_fci.csv')
# df_c = load_SHDR_fit('optimal_server_fit/AGL_20181116_fit_fc.csv')
df_s = load_SHDR_fit('optimal_server_fit/AGL_20181116_fit_s.csv')


### FIND MLD THRESHOLD
# print('Generating pool arguments...')
# threshold = 0.2  # degrees celsius
# pool_arguments = [[temp[i], pres[i], threshold] for i in range(len(date))]
#
# with mp.Pool(processes=mp.cpu_count()) as pool:
#     mld_threshold = pool.starmap(find_MLD_threshold, tqdm.tqdm(pool_arguments,
#                                                     total=len(pool_arguments)), chunksize=1)
# mld_threshold = np.asarray(mld_threshold)
# save_path = data_dir / 'SHDR_fit' / 'aux' / '02_threshold_i_20181116.npy'
# np.save(save_path, mld_threshold)


# ### QUALITY INDEXES
# def quality_index(MLD, y, z, interpolation=False):
#     z = if_masked_to_array(z)
#     y = if_masked_to_array(y)
#     if interpolation:      
#         zz = np.array([13, 18, 38, 48, 58, 68, 73, 84, 90, 102, 114, 
#                     120, 131, 136, 141, 146, 156, 161, 166, 171])
#         z, y = interpolate(z, y, zz, True)
#         
#     idx_MLD = np.searchsorted(z, MLD)
#     z_ML = z[:idx_MLD]
#     y_ML = y[:idx_MLD]
#     idx_1_5_MLD = np.searchsorted(z, 1.5*MLD)
#     
#     if idx_1_5_MLD == len(z) or idx_1_5_MLD==idx_MLD or idx_MLD==0:
#         return np.nan
#
#     if idx_1_5_MLD - idx_MLD < 3:
#         return np.nan
#
#     z_1_5 = z[:idx_1_5_MLD]
#     y_1_5 = y[:idx_1_5_MLD]
#     
#     return 1 - np.std(y_ML)/np.std(y_1_5)
#
#
# print('Generating Pool arguments...')
# pool_arguments = [[df_s['D1'][i], temp[i], pres[i], True] for i in range(len(date))]
#
# with mp.Pool(processes=mp.cpu_count() - 4) as pool:
#     QI = pool.starmap(quality_index, tqdm.tqdm(pool_arguments,
#                                                     total=len(pool_arguments)), chunksize=1)
#
# QI = np.asarray(QI)
# save_path = data_dir / 'SHDR_fit' / 'aux' / 'QI_s_i_20181116.npy'
# np.save(save_path, QI)

# G_alpha
def G_alpha(row):
    
    alpha = 0.05
    b2 = row['b2']
    c2 = row['c2']
    D1 = row['D1']
    
    if c2 == 0:
        delta_alpha = -np.log(alpha) / b2
    
    elif b2 == 0:
        delta_alpha = np.sqrt(- np.log(alpha) / c2)
    
    else:
        lambda_ = 2 * c2 / b2**2
        delta_alpha = - b2 / 2 / c2 * (1 - np.sqrt(1 - 2*lambda_*np.log(alpha)))
        
    z_alpha = delta_alpha + D1
    f_z_alpha = fit_function_row(z_alpha, row)
    G_alpha = (fit_function_row(D1, row) - f_z_alpha) / delta_alpha
    
    return G_alpha

ddf = dd.from_pandas(df_s, npartitions=10)
series = ddf.apply(G_alpha, axis=1, meta=('x', 'f8'))  
series = series.compute()
series.to_csv(data_dir / 'SHDR_fit' / 'aux' / 'G05_s.csv')
