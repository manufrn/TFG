#!/usr/bin/env python3
import sys
import time
import argparse
import h5py
import netCDF4
import textwrap
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import date, timedelta
from pathlib import Path
from numba import njit
from tqdm import tqdm, trange
from scipy.interpolate import interp1d
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('datafile', help='File containing data to be fitted')

    # Fit options 
    parser.add_argument('--b3_ref', type=str, default=None, 
                        help='File containing results of SHDR fit \
                        for a reference time series')

    parser.add_argument('-c', '--continous_fit', action='store_true')
    parser.add_argument('-i', '--interpolate', action='store_true')
    parser.add_argument('-d', '--delta_coding', action='store_true')
    parser.add_argument('--resume_fit', type=str, default=None)


    # Genetics
    parser.add_argument('-CC', '--cross_probability', type=float, default=0.61)
    parser.add_argument('--mutation_factor', type=float, default=0.71)
    parser.add_argument('--num_generations', type=int, default=1200, 
                        help='Number of generations')
    parser.add_argument('--num_individuals', type=int, default=60, 
                        help='Number of individuals per generation')

    # Numerical
    parser.add_argument('--max_b2_c2', type=float, default=0.2, 
                        help='''Maximun value for gaussian and exponential b2
                        and c2 parametres''')

    parser.add_argument('--exp_limit', type=float, default=100,
                        help="""Maximum decay combining gaussian and
                        exponential decays""")

    # Physical
    parser.add_argument('--min_depth', type=float, default=100, 
                        help="""Minimum max depth of profile""")
    parser.add_argument('--max_depth', type=float, default=450, 
                        help="""Maximum depth taken acount into the fit""")
    parser.add_argument('--min_obs', type=int, default=10, 
                        help="""Minimum datapoints""")

    # Misc
    parser.add_argument('--output_dir', type=str, default='./data/SHDR_fit', 
                        help="""Path to save results""")
    parser.add_argument('--output_file', type=str, default=None,
                        help="""Results filename""")
    parser.add_argument('--tol', type=float, default=0.00015, 
                        help='Tolerance for genetic evolution')
    parser.add_argument('--seed', type=int, default=111, help='Random seed')
    parser.add_argument('-v' , action='store_true', help='Ya veremos')
    
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
                if len(pres.shape) != 2:
                    pres = np.broadcast_to(pres, temp.shape)

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
    
    # a1
    if previous_result[5] != 0:
        a1_min = previous_result[5]*0.85
        a1_max = previous_result[5]*1.15
        lims_min[5] = a1_min if a1_min > lims_min[5] else lims_min[5]
        lims_max[5] = a1_max if a1_max < lims_max[5] else lims_max[5]

    # D1
    if previous_result[0] != 1:
        D1_min = previous_result[0]*0.8
        D1_max = previous_result[0]*1.2
        lims_min[0] = D1_min if D1_min > lims_min[0] else lims_min[0]
        lims_max[0] = D1_max if D1_max < lims_max[0] else lims_max[0]

    if args.b3_ref is None and previous_result[3] != 0:
        b3_max = previous_result[0]*0.8
        b3_min = previous_result[0]*1.2
        lims_min[3] = a1_min if a1_min > lims_min[3] else lims_min[3]
        lims_max[3] = a1_max if a1_max < lims_max[3] else lims_max[3]

    # if previous_result[4] !=0:
    #     a2_min = previous_result[4]*0.8
    #     a2_max = previous_result[4]*1.2
    #     lims_min[4] = a2_min # no need to check it is between limits, it will be
        # lims_max[4] = a2_max if a2_max < lims_max[4] else lims_max[4]

    # updated_lims_min = previous_result[slice]*0.8
    # updated_lims_max = previous_result[slice]*1.2
    #
    # updated_lims_min = np.where(abs(updated_lims_min) > abs(lims_min[slice]), 
    #                             updated_lims_min, lims_min[slice])
    # updated_lims_max = np.where(abs(updated_lims_max) < abs(lims_max[slice]), 
    #                             updated_lims_max, lims_max[slice])
    #
    # 
    # lims_min[slice] = updated_lims_min
    # lims_max[slice] = updated_lims_max

    return lims_min, lims_max


def b3_lims_from_reference(date, args):
    '''Given dataframe of reference fit, calculate mean and std
    of b3 values in that dataframe and generate a list the same
    length as date containing (mean - std) and (mean + std) in every
    value. This might be subject to changes depending on what reference
    is used.
    '''
    
    df_reference = pd.read_csv(args.b3_ref)
    b3_mean = np.mean(df_reference['b3'])
    b3_std = np.std(df_reference['b3']) 

    b3_min = b3_mean
    b3_max = b3_mean
    
    b3_lims = [b3_min, b3_max]
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
def diferential_evolution(individuals, z, y, lims, exp_limit, num_individuals, 
                          num_generations, mutation_factor, cross_probability, tol):
    n = num_individuals
    lims_min, lims_max = lims
    n_var = lims_max.size

    present_fitns = RMS_fitness(individuals, z, y, exp_limit)

    best_fit_loc = present_fitns.argmin()
    best_fit = individuals[best_fit_loc]
    
    for generation in range(num_generations):

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

        new_fitns = RMS_fitness(new_gen, z, y, exp_limit)


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

    return best_fit, present_fitns[best_fit_loc]
    

def interpolate(z, y, z_values):
    
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

    if args.interpolate:
        z_values = [13, 18, 38, 48, 58, 68, 73, 84, 90, 102, 114, 
                    120, 131, 136, 141, 146, 156, 161, 166, 171]
        z, y = interpolate(z, y, z_values)
        
    
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
                                     args.num_individuals, args.num_generations + add, 
                                     args.mutation_factor, args.cross_probability, args.tol)  

    #### DELTA CODING ####
    # set new limits for fit in function of previous fit result
    # and have them meet the physical limits
    if args.delta_coding:
        lims_min_delta, lims_max_delta = 0.85 * result_1, 1.15 * result_1
        for i in np.where(np.sign(result_1) < 0)[0]:
            lims_min_delta[i], lims_max_delta[i] = lims_max_delta[i], lims_min_delta[i]


        lims_min_delta = np.where(lims_min_delta >= lims_min, lims_min_delta,  lims_min)
        lims_max_delta = np.where(lims_max_delta <= lims_max, lims_max_delta, lims_max)

        if args.continous_fit and b3_reference_lims != None:
            lims_min[1], lims_min[2] = 0.0, 0.0
            lims_max[1], lims_max[2] = args.max_b2_c2, args.max_b2_c2

        lims_delta = (lims_min_delta, lims_max_delta)

        first_gen = random_init_population(lims_delta, args)   # new first generation

        result_delta, fitnss_delta = diferential_evolution(first_gen, z, y, lims, args.exp_limit, 
                                     args.num_individuals, args.num_generations, args.mutation_factor,
                                     args.cross_probability, args.tol)  

        if fitnss_1 < fitnss_delta:
            result = result_1
            fitnss = fitnss_1

        else:
            result = result_delta
            fitnss = fitnss_delta

    else:
        result = result_1
        fitnss = fitnss_1

    D1, b2, c2, b3, a2, a1 = result
    em = fitnss
    a3 = a1 - a2 
    return np.array([D1, b2, c2, b3, a2, a1, a3, em])
        

def save_results(lat, lon, dates, results, args):
    '''Save to results to a .csv file in results_folder using pandas df'''

    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = '{}_fit.csv'.format(Path(args.datafile).stem)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_file

    print(f'Writing results to {output_path}')

    columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1', 'a3', 'em']

    if len(lat) == 1:
        lat = np.array([lat for _ in range(len(dates))])
        lon = np.array([lon for _ in range(len(dates))])
    
    # Convert results list to pd.Dataframe and save as .csv
    results_df = pd.DataFrame(results, columns=columns)
    results_df.insert(0, 'date', dates)
    results_df.insert(1, 'lat', lat)
    results_df.insert(2, 'lon', lon)
        
    metadata_string = textwrap.dedent(f'''\
            SHDR fit for time_series = {Path(args.datafile).name}
            FIT PROPERTIES:
            continous = {args.continous_fit}
            interpolation = {args.interpolate}
            delta_coding = {args.delta_coding}
            b3_ref_fit = {args.b3_ref}
            FIT PARAMETRES:
            cross_probability = {args.cross_probability}
            mutation_factor = {args.mutation_factor}
            num_generations = {args.num_generations}
            max_b2_c2 = {args.max_b2_c2}
            exp_limit = {args.exp_limit}
            tol = {args.tol}
            seed = {args.seed}\n''')

    with open(output_path, 'w+') as f:
        f.write(metadata_string)
        results_df.to_csv(f, index=False, mode='w+')


def print_iteration_status(k, N, t_0):
    t_delta = round(time.time() - t_0)
    t_delta = timedelta(seconds=t_delta)
    print('{} profiles fitted out of {} ({:.2f} %) | Running time {}'.format(k, N, 100*k/N, t_delta))


def load_fit_to_resume(args):
    file_path = args.resume_fit

    skiprows = 0
    with open(file_path, 'r+') as f:
        first_line = f.readline()
        if first_line.startswith('SHDR'):
            skiprows = 14

    df_fit = pd.read_csv(file_path, skiprows=skiprows)
    results = []
    for index, row in df_fit.iterrows():
        result = np.array([row.D1, row.b2, row.c2, row.b3, row.a2, row.a1, row.a3, row.em])
        results.append(result)

    last_date = df_fit.iloc[-1]['date']
    return results, last_date


def main():
    args = parse_args() 

    t_0 = time.time()
        
    print('Loading data...')

    lat, lon, pres, temp, date = extract_data_from_file(args)

    if args.resume_fit:
        saved_results, last_date = load_fit_to_resume(args)
        i_0 = np.where(date == last_date)[0][0] + 1

    else:
        i_0 = 1

    N = len(date)

    if args.v:
        print('Begining fit...')
        trange_date = range(i_0, N)
        
    else:
        trange_date = trange(i_0, N, desc='Fitting profiles')

    if args.b3_ref != None:
        b3_lims = b3_lims_from_reference(date, args)

        if args.resume_fit is not None:
            results_fit = saved_results
        else:
            results_fit = [fit_profile(pres[0], temp[0], args, b3_reference_lims=b3_lims)]

        # b3 ref and continous
        if args.continous_fit:
            for k in trange_date:
                if args.v and k % int(N/200) == 0:
                    print_iteration_status(k, N, t_0)
                results_fit.append(fit_profile(pres[k], temp[k], args, results_fit[k - 1], b3_lims))

        # only b3_ref
        else: 
            for k in trange_date:
                if args.v and k % int(N/200)  == 0:
                    print_iteration_status(k, N, t_0)
                results_fit.append(fit_profile(pres[k], temp[k], args, b3_reference_lims=b3_lims))

    else:
        if args.resume_fit is not None:
            results_fit = saved_results
        else:
            print(pres[0], temp[0])
            results_fit = [fit_profile(pres[0], temp[0], args)]
        
        # only continous
        if args.continous_fit:
            for k in trange_date:
                if args.v and k % int(N/200) == 0:
                    print_iteration_status(k, N, t_0)
                results_fit.append(fit_profile(pres[k], temp[k], args, results_fit[k - 1]))

        # neither continous nor b3_ref
        else:
            for k in trange_date:
                if args.v and k % int(N/200) == 0:
                    print_iteration_status(k, N, t_0)
                results_fit.append(fit_profile(pres[k], temp[k], args))

    save_results(lat, lon, date, results_fit, args)
    print('Program exited succesfully')


if __name__ == '__main__':
    main()
