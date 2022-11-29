#!/usr/bin/env python3
import time
import argparse
import h5py
import netCDF4
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import date
from pathlib import Path
from numba import njit, prange, int32
from tqdm import tqdm
from scipy.io import loadmat
from scipy.optimize import OptimizeResult
from scipy.signal.windows import boxcar
from scipy.ndimage import uniform_filter1d, convolve1d
from scipy.interpolate import interp1d


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('datafile', help='File containing data to be fitted')

    # Assist fit 
    parser.add_argument('--reference_fit', type=str, default=None, 
                        help='File containing results of SHDR fit \
                        for a reference time series')
    parser.add_argument('-c', '--continous_fit', action='store_true')
    parser.add_argument('-i', '--interpolate', action='store_true')


    # Genetics
    parser.add_argument('-CC', '--cross_probability', type=float, default=0.61)
    parser.add_argument('--mutation_factor', type=float, default=0.71)
    parser.add_argument('--num_generations', type=int, default=1200)
    parser.add_argument('--num_individuals', type=int, default=60)

    # Fit
    parser.add_argument('--max_b2_c2', type=float, default=0.2)
    parser.add_argument('--exp_limit', type=float, default=100)
    parser.add_argument('--bias_parametres', type=float, nargs= '+', default=None)

    # Physical
    parser.add_argument('--min_depth', type=float, default=100)
    parser.add_argument('--max_depth', type=float, default=450)
    parser.add_argument('--min_obs', type=int, default=10)
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='/home/manu/TFG_repo/data/SHDR_fit')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--tol', type=float, default=0.00025)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--debuging', type=float, default=None)
    
    return parser.parse_args()


def extract_data_from_file(args, sal=False):
    fn = args.datafile

    # for hdf5 files
    if fn.endswith('.h5'):
        with h5py.File(fn, 'r') as data:
            temp = np.array(data['temperature'])
            pres = np.array(data['pressure'])
            date = np.array(data['date'])
            lat = np.array(data['lat'])
            lon = np.array(data['lon'])
        return lat, lon, pres, temp, date
        

    # for .mat files
    if fn.endswith('.mat'):
        data = loadmat(fn)

        lat = data['lat'][0]
        lon = data['lon'][0]
        pres = data['pres']
        temp = data['tems']
        dates = data['dates']
        if sal:
            sal = data['sals']
       
        return lat, lon, pres, temp, dates
    

    # for netCDF4 files
    elif fn.endswith('.nc'):
        with netCDF4.Dataset(fn, 'r') as ds:

            # check if dataset is from argo float
            if 'source' in dir(ds) and ds.source == 'Argo float':
                lat = ds.variables['LATITUDE']
                lon = ds.variables['LONGITUDE']
                pres = ds.variables['PRES_ADJUSTED']
                temp = ds.variables['TEMP_ADJUSTED']
                juld = ds.variables['JULD']  # fecha desde ff.variables['REFERENCE_DATE_TIME']
                plat = ds.variables['PLATFORM_NUMBER']
                dateref = ds.variables['REFERENCE_DATE_TIME']

                year = '%s%s%s%s' % (dateref[0], dateref[1], dateref[2], dateref[3])
                month = '%s%s' % (dateref[4], dateref[5])
                day = '%s%s' % (dateref[6], dateref[7])

                origin = datetime.date(int(year), int(month), int(day))
                return None # not implemented hehe
    
            else:
                lat = ds.variables['lat'][:]
                lon = ds.variables['lon'][:]
                try:
                    pres = ds.variables['depth'][:]
                except:
                    pres = ds.variables['pres'][:]
                temp = ds.variables['temp'][:]
                date = ds.variables['date'][:]

                return lat, lon, pres, temp, date

    else: 
        raise ValueError('Data format not recognised.')


def get_fit_limits(z, y, args):
    '''Returns the limits for the parametres of the fit function given a certain
       profile with meassures y at heights z.'''
       
    z = np.abs(z) # in case heights are defined negative

    min_z, max_z = z.min(), z.max()
    min_y, max_y = y.min(), y.max()
    
    lims = np.array([[1.0, max_z],    # D1
            [0.0, args.max_b2_c2],    # b2
            [0.0, args.max_b2_c2],    # c2
            [0.0 if max_z < args.min_depth else - abs((max_y - min_y) / (max_z - min_z)), 0.0], # b3
            [0.0, max_y - min_y],     # a2
            [min_y, max_y]])          # a1

    lims_min = lims[:, 0]
    lims_max = lims[:, 1]
    
    return lims_min, lims_max


def limits_from_previous(z, y, previous_result, args):
    lims_min, lims_max = get_fit_limits(z, y, args)

    slice = np.s_[[0, 3, 4, 5]]
    updated_lims_min = previous_result[slice]*0.8
    updated_lims_max = previous_result[slice]*1.2

    # if previous_result[1] >= 1e-4:
    #     updated_lim_max[1] = previous_result[1]*
    # updated_lims_max = previous_result[]

    updated_lims_min = np.where(abs(updated_lims_min) > abs(lims_min[slice]), updated_lims_min, lims_min[slice])
    updated_lims_max = np.where(abs(updated_lims_max) < abs(lims_max[slice]), updated_lims_max, lims_max[slice])

    # lims_min[1] = previous_result[1] * 0.5
    
    lims_min[slice] = updated_lims_min
    lims_max[slice] = updated_lims_max

    # if previous_result[1] >=1e-5:
    #     lims_max[1] = previous_result[1]*1.5

    return lims_min, lims_max

def b3_lims_from_reference(date, args):
    '''Given dataframe of reference fit, calculate mean and std
    of b3 values in that dataframe and generate a list the same
    length as date containing (mean - std) and (mean + std) in every
    value. This might be subject to changes depending on what reference
    is used.
    '''
    
    df_reference = pd.read_csv(args.reference_fit)
    b3_mean = np.mean(df_reference['b3'])
    b3_std = np.std(df_reference['b3'])

    b3_min = b3_mean - b3_std * 0.5
    b3_max = b3_mean + b3_std * 0.5
    
    # b3_min = 0.0
    # b3_max = 0.0
    
    b3_lims = [[b3_min, b3_max] for _ in range(len(date))]
    print(b3_lims)
    return b3_lims


@njit
def fit_function(individuals, z, limit):
    '''Estimate the function a group of individuals at a height z'''
		
    D1, b2, c2, b3, a2, a1 = np.split(individuals, 6, axis=1)

    pos = np.where(z >= D1, 1.0, 0.0)
    exponent = - (z -D1) * (b2 + (z - D1) * c2)
    
    # chech if exponent is inside limits
    exponent = np.where(exponent > limit, limit, exponent)
    exponent = np.where(exponent < - limit, - limit, exponent)

    return a1 + pos * (b3 * (z - D1) + a2 * (np.exp(exponent) - 1.0))


def random_init_population(lims, args):
    ''' Returns a random population of solutions of size num_individuals 
    initialized randomly with values inside limits for a profile with meassures
    y at heights z '''
    
    n = args.num_individuals 
    lims_min, lims_max = lims
    n_var = np.size(lims_max)
    
    norm = lims_max - lims_min
    individuals = lims_min + norm * np.random.random((n, n_var))
    return individuals
    

@njit
def RMS_fitness(individuals, z, y, exp_limit):
    '''Estimate the fitting for a group of individuals via mean squared error'''
    fitness = np.sqrt(np.sum((y - fit_function(individuals, z, exp_limit))**2, axis=1) / len(y))
    return fitness
    

@njit
def penalty_f(z, MLD, a, c):

     pos = np.where(z < MLD, 1.0, 0.0)
     # return a * np.exp(- (z - MLD / 4)**2 / 2 / c**2)
     return a*pos

@njit
def modified_fitness(individuals, z, y, exp_limit, MLD, a, c,):
    ''' Modfied version to implement higher error weights to points in
    the MLD '''
    
    alpha = RMS_fitness(individuals, z, y, exp_limit) * np.sqrt(len(y)) \
            / np.sum(((y - fit_function(individuals, z, exp_limit))**2 
            * penalty_f(z, MLD, a, c)), axis=1)

    fitness = np.sqrt(np.sum(((y - fit_function(individuals, z, exp_limit))**2)
                             * (1 + np.expand_dims(alpha, axis=1) * penalty_f(z, MLD, a, c)), axis=1) / len(y))
    
    return fitness


@njit
def diferential_evolution(individuals, z, y, lims, exp_limit, num_individuals, 
                          num_generations, mutation_factor, cross_probability, tol,
                          penalty_args):
    n = num_individuals
    lims_min, lims_max = lims
    n_var = lims_max.size

    if penalty_args is not None:
        # parameters = (individuals, z, y, exp_limit)
        present_fitns = modified_fitness(individuals, z, y, exp_limit, penalty_args[0], penalty_args[1], penalty_args[2])

    else:
        # parameters = (individuals, z, y, exp_limit, penalty_args[0], penalty_args[1], penalty_args[2])
        present_fitns = RMS_fitness(individuals, z, y, exp_limit)
    # present_fitns = fitness_func(*parameters)

    best_fit_loc = present_fitns.argmin()
    best_fit = individuals[best_fit_loc]
    
    for generation in range(num_generations):
        #print(np.all(individuals[0] >= lims_min) and np.all(individuals[0] <= lims_max))
        # weight of best indivual is most important in later generations

        best_weight = 0.2 + 0.8 * (generation / num_generations)**2
        
        # generate random permutations 
        perm_1 = np.random.permutation(n)
        perm_2 = np.random.permutation(n)
        new_gen = (1 - best_weight) * individuals + best_weight * best_fit + (mutation_factor
                  * (individuals[perm_1] - individuals[perm_2]))
        
        new_gen = np.where(np.random.rand(n, n_var) < cross_probability, 
                  new_gen, individuals)
                             

        # seting limits
        lims_min_bcst = np.broadcast_to(lims_min, new_gen.shape)
        lims_max_bcst = np.broadcast_to(lims_max, new_gen.shape)
        new_gen = np.where(new_gen < lims_min_bcst, lims_min_bcst, new_gen)
        new_gen = np.where(new_gen > lims_max_bcst, lims_max_bcst, new_gen)

        if penalty_args is None:
            new_fitns = RMS_fitness(new_gen, z, y, exp_limit)

        else:
            new_fitns = modified_fitness(new_gen, z, y, exp_limit, penalty_args[0], penalty_args[1], penalty_args[2])

        new_fitns_bcst = np.broadcast_to(np.expand_dims(new_fitns, axis=1), individuals.shape)
        present_fitns_bcst = np.broadcast_to(np.expand_dims(present_fitns, axis=1), individuals.shape)

        # update individuals to new generation
        individuals = np.where(present_fitns_bcst < new_fitns_bcst, individuals, new_gen)
        present_fitns = np.where(present_fitns < new_fitns, present_fitns, new_fitns)

        best_fit_loc = present_fitns.argmin()
        best_fit = individuals[best_fit_loc]
        best_fitness = present_fitns[best_fit_loc]
        
        if present_fitns.mean() * tol / present_fitns.std() > 1:
            break

        # if best_fitness < tol:
        #     break
    
    # result = OptimizeResult(x = best_fit, fun = present_fitns[best_fit_loc])
    # result = best_fit, present_fitns[best_fit_loc]
    return best_fit, present_fitns[best_fit_loc]
    
def interpolate(z, y, z_values):
    
    if isinstance(z, np.ma.core.MaskedArray):
        z = np.asarray(z[z.mask==False])
        y = np.asarray(y[y.mask==False])

    if len(z) != len(np.unique(z)):
        idx = np.argmin((z[1:] - z[:-1])) + 1
        z = np.delete(z, idx)
        y = np.delete(y, idx)
         

    interp = interp1d(z, y, 'cubic')
    idx = np.searchsorted(z, z_values)
    z = np.insert(z, idx, z_values)
    y = np.insert(y, idx, interp(z_values))
    return z, y


def fit_profile(z, y, args, previous_result=None, b3_reference_lims=None): 
    '''Parse and fit data from a single profile'''

    # remove masks and work with normal arrays
    if isinstance(z, np.ma.core.MaskedArray):
        z = np.asarray(z[z.mask==False])
        y = np.asarray(y[y.mask==False])

    z = z[np.isfinite(y)]
    y = y[np.isfinite(y)]
    
    # only use depths until max_depth
    if (z > args.max_depth).any():
        max_z_idx = np.argmax(z > args.max_depth)
        z = z[:max_z_idx]
        y = y[:max_z_idx]
    
    # if the profile doesn't have enough observations, return an array of nans
    if len(z) < args.min_obs:
        return np.full(8, np.nan)

    if isinstance(previous_result, np.ndarray):        
        lims_min, lims_max = limits_from_previous(z, y, previous_result, args)
        add = 0

    else:
        lims_min, lims_max = get_fit_limits(z, y, args)
        if args.continous_fit:
            add = +1000
        else:
            add = 0

    if b3_reference_lims != None:
        lims_min[3], lims_max[3] = b3_reference_lims

    lims = (lims_min, lims_max)

    first_gen = random_init_population(lims, args)
    result_1, fitnss_1 = diferential_evolution(first_gen, z, y, lims, args.exp_limit, 
                                     args.num_individuals, args.num_generations + add, args.mutation_factor, 
                                     args.cross_probability, args.tol, None)  

    #### DELTA CODING ####
    # set new limits for fit in function of previous fit result
    # and have them meet the physical limits
    

    lims_min_delta, lims_max_delta = 0.85 * result_1, 1.15 * result_1
    for i in np.where(np.sign(result_1) < 0)[0]:
        lims_min_delta[i], lims_max_delta[i] = lims_max_delta[i], lims_min_delta[i]


    lims_min_delta = np.where(lims_min_delta >= lims_min, lims_min_delta,  lims_min)
    lims_max_delta = np.where(lims_max_delta <= lims_max, lims_max_delta, lims_max)

    # if args.continous_fit and b3_reference_lims != None:
    #     lims_min[1], lims_min[2] = 0.0, 0.0
    #     lims_max[1], lims_max[2] = args.max_b2_c2, args.max_b2_c2

    lims_delta = (lims_min_delta, lims_max_delta)

    first_gen = random_init_population(lims_delta, args)   # new first generation

    if args.bias_parametres == None:
        result_delta, fitnss_delta = diferential_evolution(first_gen, z, y, lims, args.exp_limit, 
                                     args.num_individuals, args.num_generations, args.mutation_factor, 
                                     args.cross_probability, args.tol, None)  
    else:
        MLD = result_1[0]
        penalty_args = np.array([MLD, args.bias_parametres[0], MLD/6])
        result_delta, fitnss_delta = diferential_evolution(first_gen, z, y, lims, args.exp_limit, 
                                     args.num_individuals, args.num_generations, args.mutation_factor, 
                                     args.cross_probability, args.tol, penalty_args)

    if fitnss_1 < fitnss_delta:
        result = result_1
        fitnss = fitnss_1
    
    else:
        result = result_delta
        fitnss = fitnss_delta

    D1, b2, c2, b3, a2, a1 = result
    em = fitnss
    a3 = a1 - a2 
    return np.array([D1, b2, c2, b3, a2, a1, a3, em])
    
        

def save_results(lat, lon, dates, results, args):
    '''Save to results to a .csv file in results_folder using pandas df'''

    print('Writing results to')

    columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1', 'a3', 'em']

    if len(lat) == 1:
        lat = np.array([lat for _ in range(len(dates))])
        lon = np.array([lon for _ in range(len(dates))])
    
    # Convert results list to pd.Dataframe and save as .csv
    results_df = pd.DataFrame(results, columns=columns)
    results_df.insert(0, 'date', dates)
    results_df.insert(1, 'lat', lat)
    results_df.insert(2, 'lon', lon)
    
    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = '{}_fit.csv'.format(Path(args.datafile).stem)
        

#     metadata_string = f'''Fit for time series {Path(args.datafile).name}
# Parametres of fit:
# continous_fit = {args.continous_fit}
# cross_probability = {args.cross_probability}
# mutation_factor = {args.mutation_factor}
# num_generations = {args.num_generations}
# max_b2_c2 = {args.max_b2_c2}
# exp_limit = {args.exp_limit}
# tol = {args.tol}'''

    output_path = Path(args.output_dir) / output_file
    with open(output_path, 'w+') as f:
        # f.write(metadata_string)
        
        results_df.to_csv(f, index=False, mode='w+')

def main():
    args = parse_args() 

    
    t_0 = time.time()
        
    print('loading data')
    lat, lon, pres, temp, date = extract_data_from_file(args)
    
    if args.interpolate:
        print('generating pool arguments')
        # z_values = np.array([13, 18, 25.5, 36, 39, 46, 56.3, 59.6, 68, 73, 82, 89])
        # z_vaues = [13, 18, 38, 58, 68, 73, 87]
        # z_values = np.array([13, 18, 25.5, 36, 39, 46, 56.3, 59.6, 68, 73, 82, 89, 102, 117, 139.5, 163.5])

        z_values = [13, 18, 38, 48, 58, 68, 73, 84, 90, 102, 114, 120, 131, 136, 141, 146, 156, 161, 166, 171]
        pool_arguments = [[pres[i], temp[i], z_values] for i in range(len(date))]
        
        # check for reference_fit and construct argument list for pool
        # that contains the reference (parametres of fit) that each fit should 
        # have depending on their date
        print('generating interpolated temperature')
        with mp.Pool(processes=mp.cpu_count()) as pool:
            interpolated_x_y = pool.starmap(interpolate, tqdm(pool_arguments,
                                                total=len(pool_arguments)), chunksize=1)

        if args.continous_fit and args.reference_fit != None:
            b3_lims = b3_lims_from_reference(date, args)
            results_fit = [fit_profile(interpolated_x_y[0][0],interpolated_x_y[0][1], args, b3_reference_lims=b3_lims[0])]
            print('Begining DE fit...')
            for k in tqdm(range(1, len(date))):
                results_fit.append(fit_profile(interpolated_x_y[k][0], interpolated_x_y[k][1], args, results_fit[k - 1], b3_lims[k]))  

        elif args.reference_fit != None:
            b3_lims = b3_lims_from_reference(date, args)
            print(b3_lims[0])
            results_fit = []
            for k in tqdm(range(len(date))):
                results_fit.append(fit_profile(pres[k], temp[k], args, b3_reference_lims=b3_lims[k]))


    

    # if args.debuging != None:
    #     # from analysis_functions import *
    #     ### code to perform debuging
    #     if args.continous_fit and args.reference_fit != None:
    #         slice = np.s_[10855:10858]
    #         pres = pres[slice]
    #         temp = temp[slice]
    #         date = date[slice]
    #         b3_lims = b3_lims_from_reference(date, args)
    #         C = np.array([0.502, 0.527, 0.545, 0.593, 0.634, 0.678, 0.703, 0.728])
    #         F = C.copy()
    #
    #         args.mutation_factor = F[0]
    #         args.cross_probability = C[0]
    #         results_fit = [fit_profile(pres[0], temp[0], args, b3_reference_lims=b3_lims[0])]
    #         for c in C[1:]:
    #             args.cross_probability = c
    #             for f in F[1:]:
    #                 args.mutation_factor = f
    #                 results_fit = [fit_profile(pres[0], temp[0], args, b3_reference_lims=b3_lims[0])]
    #                 for k in range(1, len(date)):
    #                     results_fit.append([fit_profile(pres[k], temp[k], args, results_fit[k - 1], b3_lims[k]),
    #                                    args.k, f, c])
    #             
    #         columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1', 'a3', 'em', 'k', 'f', 'c']
    #         results_df = pd.DataFrame(results, columns=columns)
    #         date = np.tile(date, len(c)**2)
    #         results_df.insert(0, 'date', date)
            
            
                
    
    elif args.continous_fit and args.reference_fit != None:
        b3_lims = b3_lims_from_reference(date, args)
        results_fit = [fit_profile(pres[0], temp[0], args, b3_reference_lims=b3_lims[0])]
        for k in tqdm(range(1, len(date))):
            results_fit.append(fit_profile(pres[k], temp[k], args, results_fit[k - 1], b3_lims[k]))

    elif args.reference_fit != None:
        b3_lims = b3_lims_from_reference(date, args)
        results_fit = []
        for k in tqdm(range(len(date))):
            results_fit.append(fit_profile(pres[k], temp[k], args, b3_reference_lims=b3_lims[k]))

    else:
        results_fit = []
        for k in tqdm(range(len(date))):
            results_fit.append(fit_profile(pres[k], temp[k], args))
         
    save_results(lat, lon, date, results_fit, args)

if __name__ == '__main__':
    main()
