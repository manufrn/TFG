import time
import tqdm
import argparse
import multiprocessing as mp
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import scipy.io
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
    parser.add_argument('--bias_parametres', type=tuple, default=(1, 1))

    # Physical
    parser.add_argument('--min_depth', type=float, default=100)
    parser.add_argument('--max_depth', type=float, default=300)
    parser.add_argument('--min_obs', type=int, default=10)
    
    # Misc
    parser.add_argument('--results_folder', type=str, default='/home/manu/TFG_repo/data/SHDR_fit')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--tol', type=float, default=0.0025)
    
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

def fit_function(individuals, z, args):
    '''Estimate the function a group of individuals at a height z'''
		
    limit = args.exp_limit
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
    

def fitness(individuals, z, y, args):
    '''Estimate the fitting for a group of individuals via mean squared error'''
    
    fitness = np.sqrt(np.sum((y - fit_function(individuals, z, args))**2, axis=1) / len(y))
    return fitness
    

def penalty_f(z, a, c, MLD):
    return a * np.exp(- (z - MLD)**2 / 2 / c**2)


def modified_fitness(individuals, z, y, args, MLD, a, c,):
    ''' Modfied version to implement higher error weights to points in
    the MLD '''
    
    alpha = fitness(individuals, z, y, args) * np.sqrt(len(y)) \
            / np.sum(((y - fit_function(individuals, z, args))**2 
            * penalty_f(z, a, c, MLD)), axis=1)

    fitness = np.sqrt(np.sum(((y - fit_function(individuals, z, args))**2)
              * (1 + alpha)) / len(y))

    return fitness


def diferential_evolution(individuals, z, y, args, delta_args=None):
    n = args.num_individuals
    lims_min, lims_max = get_fit_limits(z, y, args)
    n_var = np.size(lims_max)
    
    if delta_args == None:
        present_fitns = fitness(individuals, z, y, args)

    else:
        present_fitns = modified_fitness(individuals, z, y, args, **delta_args)

    best_fit_loc = present_fitns.argmin()
    best_fit = individuals[best_fit_loc]
    
    for generation in range(args.num_generations):
        #print(np.all(individuals[0] >= lims_min) and np.all(individuals[0] <= lims_max))
        # weight of best indivual is most important in later generations

        best_weight = 0.2 + 0.8 * (generation / args.num_generations)**2
        
        # generate random permutations 
        perm_1 = np.random.permutation(n)
        perm_2 = np.random.permutation(n)
        new_gen = (1 - best_weight) * individuals + best_weight * best_fit + (args.mutation_factor
                  * (individuals[perm_1] - individuals[perm_2]))
        
        new_gen = np.where(np.random.rand(n, n_var) < args.cross_probability, 
                  new_gen, individuals)
                             

        # seting limits
        new_gen = np.where(new_gen < lims_min.reshape((1,6)), lims_min.reshape((1,6)), new_gen)
        new_gen = np.where(new_gen > lims_max.reshape((1,6)), lims_max.reshape((1,6)), new_gen)
            
        new_fitns = fitness(new_gen, z, y, args)
        
        # update individuals to new generation
        individuals = np.where(present_fitns[:, None] < new_fitns[:, None], individuals, new_gen)
        present_fitns = np.where(present_fitns < new_fitns, present_fitns, new_fitns)

        best_fit_loc = present_fitns.argmin()
        best_fit = individuals[best_fit_loc]
        
        if present_fitns.mean() * args.tol / present_fitns.std() > 1:
            break
    
    result = OptimizeResult(x = best_fit, fun = present_fitns[best_fit_loc])
    return result
    

def fit_profile(z, y, args):
    '''Parse and fit data from a single profile'''
    
    z = z[np.isfinite(y)]
    y = y[np.isfinite(y)]
    
    # only use depths until max_depth
    max_z = np.nonzero(z > args.max_depth)[0]
    z = z[:max_z] if len(max_z) != 0 else z
    
    if len(z) < args.min_obs:
        return 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.9
    
    first_gen = random_init_population(z, y, args)

    optimized_result = diferential_evolution(first_gen, z, y, args)
    
    #### DELTA CODING ####
    # TO-DO

    
    D1, b2, c2, b3, a2, a1 = optimized_result.x
    a3 = a1 - a2 
    return D1, b2, c2, b3, a2, a1, a3, optimized_result.fun
    
        
def save_results(lat, lon, dates, results, args):
    '''Save to results to a .csv file in results_folder using pandas df'''

    print('Writing results to')

    columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1', 'a3', 'em']

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


def fit_time_series(args):

    t_0 = time.time()

    lat, lon, pres, temps, dates = extract_data_from_file(args)
    pool_arguments = [[pres, temps[:, i], args] for i in range(len(dates))]

    print('Begining DE fit...')

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_fit = pool.starmap(fit_profile, tqdm.tqdm(pool_arguments,
                                                          total=len(pool_arguments)), chunksize=1)
    save_results(lat, lon, dates, results_fit, args)
    
    print('Elapsed time: {:.2f} seconds'.format(time.time() - t_0))

def fit_with_reference(args):

    lat, lon, pres, temps, dates = extract_data_from_file(args)

    df_reference = pd.read_csv(args.reference_fit)

    # check where referece == dates
    idx = dates.searchsorted(df_reference['Dates'])


def main():
    args = parse_args() 

    if args.reference_fit == None:
        fit_time_series()

    else:
        fit_with_reference()


if __name__ == '__main__':
    main()
    
