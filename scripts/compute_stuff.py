import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import multiprocessing as mp
import tqdm as tqdm
from config import data_dir
from analysis_routines import * 

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
pbar = ProgressBar()                

pbar.register()

print('Loading time series...')
data_chain = load_time_series_xr('processed/AGL_20181116_chain_xrcompatible.nc')
# df_ci = load_SHDR_fit('optimal_server_fit/AGL_20181116_fit_fci.csv')
# mld_thrs_02 = pd.read_csv(data_dir / 'SHDR_fit/aux/MLD_trsh_02_i.csv')


df_c = load_SHDR_fit('optimal_server_fit/AGL_20181116_fit_fc.csv')
# df_s = load_SHDR_fit('optimal_server_fit/AGL_20181116_fit_s.csv')


def find_MLD_threshold(y, z, threshold=0.2, interpolation=True):
    z = z[np.isfinite(y)] 
    y = y[np.isfinite(y)]

    if interpolation:
        zz = np.linspace(min(z), max(z), int(max(z) - min(z) + 1))
        new_z, new_y = interpolate(z, y, zz, False)

    dif = new_y[0] - new_y
    idx = np.searchsorted(dif, threshold)

    if idx == len(new_z):
        MLD = new_z[-1]
    else:
        MLD = new_z[idx]

    return MLD

def MLD_threshold_mp(data_chain, output_path):
    temp = data_chain.temp.data
    depth = data_chain.depth.data

    depths = np.repeat(depth[None, :], np.shape(temp)[0], axis=0)

    pool_args = list(zip(temp, depths))

    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        mld_thrs = pool.starmap(find_MLD_threshold, tqdm.tqdm(pool_args, total=len(pool_args)), chunksize=1)

    df_result = pd.DataFrame(zip(data_chain.date.data, mld_thrs), columns=['date', 'x']) 
    df_result.to_csv(output_path, index=False)
    print(f'Results saved to {output_path}')

# MLD_threshold_mp(data_chain, data_dir / 'SHDR_fit/aux' / 'MLD_trsh_02_i.csv')

# ### QUALITY INDEXES

def QI(y, z, mld):
    z = z[np.isfinite(y)]
    y = y[np.isfinite(y)]

    zz = np.linspace(min(z), max(z), int(max(z) - min(z) + 1))

    new_z, new_y = interpolate(z, y, zz, False)
    idx_MLD = np.searchsorted(new_z, mld)
    z_ML = new_z[:idx_MLD]
    y_ML = new_y[:idx_MLD]
    idx_1_5_MLD = np.searchsorted(new_z, 1.5*mld)

    if idx_1_5_MLD == len(z) or idx_1_5_MLD==idx_MLD or idx_MLD==0:
        return np.nan

    elif idx_1_5_MLD - idx_MLD < 3:
        return np.nan

    z_1_5 = new_z[:idx_1_5_MLD]
    y_1_5 = new_y[:idx_1_5_MLD]


    return 1 - np.std(y_ML)/np.std(y_1_5)
    

def quality_index_mp(data_chain, mld, output_path):
    temp = data_chain.temp.data
    depth = data_chain.depth.data

    depths = np.repeat(depth[None, :], len(mld), axis=0)

    pool_args = list(zip(temp, depths, mld))
    
    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        QI_ = pool.starmap(QI, tqdm.tqdm(pool_args, total=len(pool_args)), chunksize=1)

    df_result = pd.DataFrame(zip(data_chain.date.data, QI_), columns=['date', 'x']) 
    df_result.to_csv(output_path, index=False)
    print(f'Results saved to {output_path}')

# quality_index_mp(data_chain, df_s.D1, data_dir / 'SHDR_fit/aux' / 'QI_is_20181116.csv')


# G_alpha
def G_alpha(row):
    
    alpha = 0.05
    b2 = row['b2']
    c2 = row['c2']
    D1 = row['D1']
    
    if c2 < 1e-15 and b2 < 1e-15:
        G_alpha = 0
        return G_alpha

    if c2 < 1e-15:
        delta_alpha = -np.log(alpha) / b2
    
    elif b2 == 0:
        delta_alpha = np.sqrt(- np.log(alpha) / c2)
    
    else:
        lambda_ = 2 * c2 / b2**2
        delta_alpha = - b2 / 2 / c2 * (1 - np.sqrt(1 - 2*lambda_*np.log(alpha)))
        
    z_alpha = delta_alpha + D1
    f_z_alpha = fit_function_row(row, z_alpha)
    G_alpha = (fit_function_row(row, D1) - f_z_alpha) / delta_alpha
    
    return G_alpha

def delta_alpha(row):
    alpha = 0.05
    b2 = row['b2']
    c2 = row['c2']
    D1 = row['D1']
    
    # limite testeado experimentalmente siuu
    if c2 < 1e-15:
        delta_alpha = -np.log(alpha) / b2
    
    elif b2 == 0:
        delta_alpha = np.sqrt(- np.log(alpha) / c2)
    
    else:
        lambda_ = 2 * c2 / b2**2
        delta_alpha = - b2 / 2 / c2 * (1 - np.sqrt(1 - 2*lambda_*np.log(alpha)))

    return delta_alpha


ddf = dd.from_pandas(df_c, npartitions=10)
series = ddf.apply(G_alpha, axis=1, meta=('x', 'f8'))  
series = series.compute()
series.to_csv(data_dir / 'SHDR_fit' / 'aux' / 'G05_c.csv')




