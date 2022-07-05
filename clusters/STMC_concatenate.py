'''
##=========================##
## CONCATENATE STMC OUTPUT ##
##=========================##
- Efficient to output a single array for each root file. This script is
designed to efficiently concatenate individual outputs into a single mem-map

author: Russell Bate
russellbate@phas.ubc.ca
russell.bate@cern.ch
rbate@triumf.ca
'''

datapath_prefix = "/data/atlas/rbate/"
module_path = '/home/russbate/MLPionCollaboration/LCStudies/'

print()
print('starting concatenation for STMC..');print()

## General Imports
#===============w=======================
import numpy as np
import time, argparse
from time import process_time as cput
import os
cwd = os.getcwd()

## Local ML Packages
import sys
sys.path.append(module_path)
from util import deep_set_util as dsu

## Read in Parameters
#=============================================================================
parser = argparse.ArgumentParser(description='Inputs for concatenating STMC.')

parser.add_argument('--file_loc', action="store", dest='fl', default=\
                    '/fast_scratch_1/atlas/staged/STMC/',
                   type=str)
parser.add_argument('--output_loc', action="store", dest='ol', default=\
                    '/fast_scratch_1/atlas/staged/',
                   type=str)
parser.add_argument('--train', action='store_true',
                    dest='train')
parser.add_argument('--test', action='store_true',
                    dest='test')
parser.add_argument('--files', action='store', dest='fi',
                    default=None, type=int, nargs="+")
parser.add_argument('--x_size', action='store', dest='xs', default=None,
                   type=int, nargs="+")
parser.add_argument('--event_start', action='store', dest='es', default=None,
                   type=int)
args = parser.parse_args()


file_loc = args.fl
output_loc = args.ol
x_shape = args.xs
evt_start = args.es
test_bool = args.test
train_bool = args.train

# Checks
if not train_bool and not test_bool:
    sys.exit('Need to specify --train or --test. Exiting.')
    
elif train_bool and test_bool:
    sys.exit('Cannot specify --train and --test. Exiting.')
        
if len(x_shape) != 3:
    raise ValueError('--x_size must be dimension three.')

    
# Training
if train_bool:
    tr_tst_str = 'train'
    if args.fi is not None:
        start_file, end_file = args.fi

        if end_file - start_file >= 400:
            raise ValueError('Too many files requested by --files')

        if end_file - start_file == 400:
            outstr = '400_train.npy'

        else:
            outstr = '{}_to_{}_train.npy'.format(start_file, end_file)

    else:
        start_file, end_file = (0,400)
        outstr = '400_train.npy'

# Test
elif test_bool:
    tr_tst_str = 'test'
    if args.fi is not None:
        start_file, end_file = args.fi

        if end_file - start_file >= 100:
            raise ValueError('Too many files requested by --files')

        if end_file - start_file == 100:
            outstr = '100_test.npy'

        else:
            outstr = '{}_to_{}_test.npy'.format(start_file, end_file)

    else:
        start_file, end_file = (0,100)
        outstr = '100_test.npy'


if evt_start is not None:
    Xfin = np.lib.format.open_memmap(output_loc+'X_STMC_v2_'+outstr, mode='w+',
                                    dtype=np.float64,
                                    shape=(x_shape[0], x_shape[1], 5))
    Yfin = np.lib.format.open_memmap(output_loc+'Y_STMC_v2_'+outstr, mode='w+',
                                    dtype=np.float64,
                                    shape=(x_shape[0],))
    Etafin = np.lib.format.open_memmap(output_loc+'Eta_STMC_v2_'+outstr, mode='w+',
                                    dtype=np.float64,
                                    shape=(x_shape[0],))
else:
    Xfin = np.lib.format.open_memmap(output_loc+'X_STMC_v2_'+outstr, mode='w+',
                                    dtype=np.float64,
                                    shape=(x_shape[0], x_shape[1], 5)
                                    )[evt_start:,:,:]
    Yfin = np.lib.format.open_memmap(output_loc+'Y_STMC_v2_'+outstr, mode='w+',
                                    dtype=np.float64,
                                    shape=(x_shape[0],))[evt_start:]
    Etafin = np.lib.format.open_memmap(output_loc+'Eta_STMC_v2_'+outstr, mode='w+',
                                    dtype=np.float64,
                                    shape=(x_shape[0],))[evt_start:]


## Main execution loop ##
nEvts = 0
t_tot = 0
print()
for i in range(start_file, end_file, 1):
    print('Working on file {}'.format(i))
    x_str = 'X_STMC_v2_{}_{}.npy'.format(tr_tst_str, i)
    y_str = 'Y_STMC_v2_{}_{}.npy'.format(tr_tst_str, i)
    e_str = 'Eta_STMC_v2_{}_{}.npy'.format(tr_tst_str, i)
    
    print('loading files..')
    t0 = cput()
    X = np.load(file_loc+x_str)
    Y = np.load(file_loc+y_str)
    Eta = np.load(file_loc+e_str)
    t1 = cput()
    load_time = t1 - t0
    print('load time: {} (s)'.format(load_time))
    
    old_evts = nEvts
    nEvts += X.shape[0]
    print('slices: {}:{}'.format(old_evts, nEvts))
    
    ## X ##
    t0 = cput()
    Xfin[old_evts:nEvts,:X.shape[1],:5] = X[:,:,:5]
    t1 = cput()
    x_copy = t1 - t0
    print('x copy time: {} (s)'.format(x_copy))
    
    ## Y ##
    t0 = cput()
    Yfin[old_evts:nEvts] = Y[:,0]

    ## Eta ##
    Etafin[old_evts:nEvts] = Eta
    t1 = cput()
    the_rest = t1 - t0
    print('the rest time: {} (s)'.format(the_rest))
    
    
    ## flooooosh
    print('flooshing..')
    t0 = cput()
    Xfin.flush()
    Yfin.flush()
    Etafin.flush()
    t1 = cput()
    print('floosh time: {} (s)'.format(t1 - t0))
    print()
    
    t_file_tot = load_time + x_copy + the_rest
    t_tot += t_file_tot
    print('Total time for this file: {}'.format(t_file_tot))
    print('Total time: {}'.format(t_tot))
    print()