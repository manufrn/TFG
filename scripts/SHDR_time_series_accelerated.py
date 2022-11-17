#!/usr/bin/env python3
import time
from tqdm import tqdm
import argparse
import multiprocessing as mp
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import netCDF4
import scipy.io
import jax
from numba import jit
from scipy.optimize import OptimizeResult

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('datafile', help='File containing data to be fitted')

    # Assist fit
    parser.add_argument('--reference_fit', type=str, default=None, 
                        help='File containing results of SHDR fit \
                        for a reference time series')

    # Genetics
    parser.add_argument('--cross_probability', type=float, default=0.5)
    parser.add_argument('--mutation_factor', type=float, default=0.5)
    parser.add_argument('--num_generations', type=int, default=1000)
    parser.add_argument('--num_individuals', type=int, default=60)

    # Fit
    parser.add_argument('--max_b2_c2', type=float, default=0.5)
    parser.add_argument('--exp_limit', type=float, default=100)
    parser.add_argument('--bias_parametres', type=float, nargs= '+', default=None)

    # Physical
    parser.add_argument('--min_depth', type=float, default=100)
    parser.add_argument('--max_depth', type=float, default=450)
    parser.add_argument('--min_obs', type=int, default=10)
    
    # Misc
    parser.add_argument('--results_folder', type=str, default='/home/manu/TFG_repo/data/SHDR_fit')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--tol', type=float, default=0.0025)
    parser.add_argument('--seed', type=int, default=111)
    
    return parser.parse_args()


def extract_data_from_file(args, sal=False):

    fn = args.datafile
    # for .mat files
    if fn.endswith('.h5'):
        with h5py.File(fn, 'r') as data:
            temp = np.array(data['temperature'])
            pres = np.array(data['pressure'])
            date = np.array(data['date'])
            lat = np.array(data['lat'])
            lon = np.array(data['lon'])
        return lat, lon, pres, temp, date
        

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
    

    # for netCDF4 files:
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
                return None
    
            else:
                lat = ds.variables['lat'][:]
                lon = ds.variables['lon'][:]
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
    previous_result = previous_result[:-2]
    negative_sign_locs = np.where(np.sign(previous_result) == -1)[0]

    lims_min = previous_result * 0.1
    lims_max = previous_result * 10

    for i in negative_sign_locs:
        lims_max[i], lims_min[i] = lims_min[i], lims_max[i]

    lims_min_standard, lims_max_standard = get_fit_limits(z, y, args)

    lims_min = np.where(abs(lims_min) > abs(lims_min_standard), lims_min, lims_min_standard)
    lims_max = np.where(abs(lims_max) < abs(lims_max_standard), lims_max, lims_max_standard)

    return lims_min, lims_max

def limits_from_reference(z, y, reference, args):
    '''Given SHDR fit parametres from reference, get limits
    for new fit that are consistent with previous 
    TODO: for the moment it just returns get_fit_limits()
    '''
    lims_min, lims_max = get_fit_limits(z, y, args)
    lower_b3 = reference[0] - reference[1]
    upper_b3 = reference[0] + reference[1]
    lims_min[3], lims_max[3] = lower_b3, upper_b3

    return lims_min, lims_max

@jit(nopython=True)
def fit_function(individuals, z, limit):
    '''Estimate the function a group of individuals at a height z'''
		
    D1, b2, c2, b3, a2, a1 = np.split(individuals, 6, axis=1)

    pos = np.where(z >= D1, 1.0, 0.0)
    exponent = - (z -D1) * (b2 + (z - D1) * c2)
    
    # chech if exponent is inside limits
    exponent = np.where(exponent > limit, limit, exponent)
    exponent = np.where(exponent < - limit, - limit, exponent)

    return a1 + pos * (b3 * (z - D1) + a2 * (np.exp(exponent) - 1.0))


def random_init_population(z, y, args):
    ''' Returns a random population of solutions of size num_individuals 
    initialized randomly with values inside limits for a profile with meassures
    y at heights z '''
    
    n = args.num_individuals 
    lims_min, lims_max = get_fit_limits(z, y, args)
    n_var = np.size(lims_max)
    
    norm = lims_max - lims_min
    individuals = lims_min + norm * np.random.random((n, n_var))
    return individuals
    

@jit(nopython=True)
def RMS_fitness(individuals, z, y, exp_limit):
    '''Estimate the fitting for a group of individuals via mean squared error'''
    
    fitness = np.sqrt(np.sum((y - fit_function(individuals, z, exp_limit))**2, axis=1) / len(y))
    return fitness
    

def penalty_f(z, MLD, a, c):

     pos = np.where(z > 200, 1.0, 0.0)
     # return a * np.exp(- (z - MLD / 4)**2 / 2 / c**2)
     return a*pos


def modified_fitness(individuals, z, y, args, MLD, a, c,):
    ''' Modfied version to implement higher error weights to points in
    the MLD '''
    
    alpha = RMS_fitness(individuals, z, y, args) * np.sqrt(len(y)) \
            / np.sum(((y - fit_function(individuals, z, args))**2 
            * penalty_f(z, MLD, a, c)), axis=1)

    fitness = np.sqrt(np.sum(((y - fit_function(individuals, z, args))**2)
                             * (1 + alpha[:, None] * penalty_f(z, MLD, a, c)), axis=1) / len(y))
    return fitness

@jit(nopython=True)
def diferential_evolution(individuals, z, y, lims, fitness_func, exp_limit, num_individuals, 
                          num_generations, mutation_factor, cross_probability, tol,
                          penalty_args = None):
    n = num_individuals
    lims_min, lims_max = lims
    n_var = lims_max.size
     
    if penalty_args == None:
        present_fitns = fitness_func(individuals, z, y, exp_limit)

    else:
        present_fitns = fitness_func(individuals, z, y, args, *penalty_args)

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

        if penalty_args == None:
            new_fitns = fitness_func(new_gen, z, y, exp_limit)

        else:
            new_fitns = fitness_func(new_gen, z, y, args, *penalty_args)

        new_fitns_bcst = np.broadcast_to(np.expand_dims(new_fitns, axis=1), individuals.shape)
        present_fitns_bcst = np.broadcast_to(np.expand_dims(present_fitns, axis=1), individuals.shape)
        # update individuals to new generation
        individuals = np.where(present_fitns_bcst < new_fitns_bcst, individuals, new_gen)
        present_fitns = np.where(present_fitns < new_fitns, present_fitns, new_fitns)

        best_fit_loc = present_fitns.argmin()
        best_fit = individuals[best_fit_loc]
        
        if present_fitns.mean() * tol / present_fitns.std() > 1:
            break
    
    # result = OptimizeResult(x = best_fit, fun = present_fitns[best_fit_loc])
    # result = best_fit, present_fitns[best_fit_loc]
    return best_fit, present_fitns[best_fit_loc]
    
def fit_profile(z, y, args, previous_result = None, reference=None): 
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
    
    if len(z) < args.min_obs:
        return 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.9
    
    if isinstance(previous_result, np.ndarray):        
        lims_min, lims_max = limits_from_previous(z, y, previous_result, args)

    else:
        lims_min, lims_max = get_fit_limits(z, y, args)
    
    # else:
    #     lims_min, lims_max = limits_from_reference(z, y, reference, args)

    lims = (lims_min, lims_max)

    first_gen = random_init_population(z, y, args)
    result_1, fitnss_1 = diferential_evolution(first_gen, z, y, lims, RMS_fitness, args.exp_limit, 
                                     args.num_individuals, args.num_generations, args.mutation_factor, 
                                     args.cross_probability, args.tol)  

    #### DELTA CODING ####
    
    # set new limits for fit in function of previous fit result
    # and have them meet the physical limits
    v_min, v_max = 0.85 * result_1, 1.15 * result_1
    for i in range(6):
        lim_min_d = min(v_min[i], v_max[i])
        lim_max_d = max(v_min[i], v_max[i])
        lims_min[i] = max(lims_min[i], lim_min_d)
        lims_max[i] = max(lims_max[i], lim_max_d)
    lims_delta = (lims_min, lims_max)

    first_gen = random_init_population(z, y, args)   # new first generation

    if args.bias_parametres == None:
        result_delta, fitnss_delta = diferential_evolution(first_gen, z, y, lims, RMS_fitness, args.exp_limit, 
                                     args.num_individuals, args.num_generations, args.mutation_factor, 
                                     args.cross_probability, args.tol)  
    else:
        MLD = result_1.x[0]
        penalty_args = [MLD, args.bias_parametres[0], MLD/6]
        result_delta, fitnss_delta = diferential_evolution(first_gen, z, y, lims, modified_fitness, args, \
                                             penalty_args)

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
    results_df.insert(0, 'Dates', dates)
    results_df.insert(1, 'lat', lat)
    results_df.insert(2, 'lon', lon)
    
    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = '{}_fit.csv'.format(Path(args.datafile).stem)
        
    output_path = Path(args.results_folder) / output_file

    results_df.to_csv(output_path, index=False, mode='w+')

def generate_reference_arguments(date, args):

    df_reference = pd.read_csv(args.reference_fit)

    idx = np.searchsorted(date, df_reference['Dates'])
    array_reference = df_reference.to_numpy()[:, 3:] # convert dataframe to array
                                                     # and remove date, lat, lon
    b3_mean = np.mean(df_reference['b3'])
    b3_std = np.std(df_reference['b3'])
    del df_reference

    reference_arguments = []
    for i in range(len(date)):
        l = i >= idx
        i = len(l) - np.argmax(l[::-1]) - 1
        # reference_arguments.append(array_reference[i])
        reference_arguments.append([b3_mean, b3_std])
    return reference_arguments


def main():
    args = parse_args() 
    
    t_0 = time.time()
    
    lat, lon, pres, temps, dates = extract_data_from_file(args)
    # pool_arguments = [[pres[i, :], temps[i, :], args] for i in range(len(dates))]
    
    # check for reference_fit and construct argument list for pool
    # that contains the reference (parametres of fit) that each fit should 
    # have depending on their date
    # if args.reference_fit != None:
    #     reference_arguments = generate_reference_arguments(dates, args)
    #     pool_arguments = [[pres[i, :], temps[i, :], args, reference_arguments[i]] for i in range(len(dates))]
    #
    # else:
    #     pool_arguments = [[pres[i, :], temps[i, :], args] for i in range(len(dates))]

        # pool_arguments.append(reference_arguments)

    # print('Time used to read data and generate arguments: ', time.time() - t_0)

    print('Begining DE fit...')

    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results_fit = pool.starmap(fit_profile, tqdm.tqdm(pool_arguments,
    #                                                       total=len(pool_arguments)), chunksize=1)
    results_fit = []
    results_fit.append(fit_profile(pres[0], temps[0], args))
    t_0_0 = time.time()
    for k in tqdm(range(1, len(dates))):
        results_fit.append(fit_profile(pres[k], temps[k], args, results_fit[k - 1]))
    
    print(time.time() - t_0_0)
    save_results(lat, lon, dates, results_fit, args)
    
    print('Elapsed time: {:.2f} seconds'.format(time.time() - t_0))

if __name__ == '__main__':
    main()
