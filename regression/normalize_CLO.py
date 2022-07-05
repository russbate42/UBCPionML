'''
##===================##
## CLO NORMALIZATION ##
##===================##
- Normalize the Cluster Only dataset

author: Russell Bate
russellbate@phas.ubc.ca
russell.bate@cern.ch
'''

datapath_prefix = "/data/atlas/rbate/"
module_path = '/home/russbate/MLPionCollaboration/LCStudies/'

print()
print('starting data normalization for CLO..');print()

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


## Declaration of size!
''' This is determined by rounding to the nearest 100
of the largest number of points either training or evaluation arrays '''
MAX_NPTS = 1300


## Read in Parameters
#=============================================================================
parser = argparse.ArgumentParser(description='Inputs for normalizing CLO.')

parser.add_argument('--file_loc', action="store", dest='fl', default=\
                    '/fast_scratch_1/atlas/staged/',
                   type=str)
parser.add_argument('--output_loc', action="store", dest='ol', default=\
                    '/fast_scratch_1/atlas/normalized/',
                   type=str)
parser.add_argument('--train', action='store_true',
                    dest='train')
parser.add_argument('--test', action='store_true',
                    dest='test')
parser.add_argument('--pipm', action='store_true',
                    dest='pipm')
parser.add_argument('--pi0', action='store_true',
                    dest='pi0')
parser.add_argument('--find_values', action='store_true',
                    dest='fv')
parser.add_argument('--nClust', action='store', dest='nc',
                    default=None, type=int)
args = parser.parse_args()

file_loc = args.fl
output_loc = args.ol
train_bool = args.train
test_bool = args.test
pipm_bool = args.pipm
pi0_bool = args.pi0
find_values = args.fv
nClust = args.nc


if not train_bool and not test_bool:
    sys.exit('Need to specify --train or --test. Exiting.')
    
elif train_bool and test_bool:
    sys.exit('Cannot specify --train and --test. Exiting.')

if not pi0_bool and not pipm_bool:
    sys.exit('Need to specify --pipm or pi0. Exiting')

elif pipm_bool and pi0_bool:
    sys.exit('Cannor specify --pipm and --pi0. Exiting')

if train_bool:
    if pi0_bool:
        print('Normalizing training data for pi0')
        x_string = file_loc + 'X_CLO_PI0_400_train.npy'
        y_string = file_loc + 'Y_CLO_PI0_400_train.npy'
        e_string = file_loc + 'Eta_CLO_PI0_400_train.npy'
        
        if nClust is not None:
            x_norm_str = 'X_CLO_PI0_train_norm_{}.npy'.format(nClust)
            y_norm_str = 'Y_CLO_PI0_train_norm_{}'.format(nClust)
            e_norm_str = 'Eta_CLO_PI0_train_norm_{}'.format(nClust)
        elif nClust is None:
            x_norm_str = 'X_CLO_PI0_400_train_norm.npy'
            y_norm_str = 'Y_CLO_PI0_400_train_norm'
            e_norm_str = 'Eta_CLO_PI0_400_train_norm'

    if pipm_bool:
        print('Normalizing training data for pipm')
        x_string = file_loc + 'X_CLO_PIPM_400_train.npy'
        y_string = file_loc + 'Y_CLO_PIPM_400_train.npy'
        e_string = file_loc + 'Eta_CLO_PIPM_400_train.npy'
        
        if nClust is not None:
            x_norm_str = 'X_CLO_PIPM_train_norm_{}.npy'.format(nClust)
            y_norm_str = 'Y_CLO_PIPM_train_norm_{}'.format(nClust)
            e_norm_str = 'Eta_CLO_PIPM_train_norm_{}'.format(nClust)
        elif nClust is None:
            x_norm_str = 'X_CLO_PIPM_400_train_norm.npy'
            y_norm_str = 'Y_CLO_PIPM_400_train_norm'
            e_norm_str = 'Eta_CLO_PIPM_400_train_norm'

if test_bool:
    if pi0_bool:
        print('Normalizing test data for pi0')
        x_string = file_loc + 'X_CLO_PI0_100_test.npy'
        y_string = file_loc + 'Y_CLO_PI0_100_test.npy'
        e_string = file_loc + 'Eta_CLO_PI0_100_test.npy'
        
        if nClust is not None:
            x_norm_str = 'X_CLO_PI0_test_norm_{}.npy'.format(nClust)
            y_norm_str = 'Y_CLO_PI0_test_norm_{}'.format(nClust)
            e_norm_str = 'Eta_CLO_PI0_test_norm_{}'.format(nClust)
        elif nClust is None:
            x_norm_str = 'X_CLO_PI0_100_test_norm.npy'
            y_norm_str = 'Y_CLO_PI0_100_test_norm'
            e_norm_str = 'Eta_CLO_PI0_100_test_norm'
            
    if pipm_bool:
        print('Normalizing test data for pipm')
        x_string = file_loc + 'X_CLO_PIPM_100_test.npy'
        y_string = file_loc + 'Y_CLO_PIPM_100_test.npy'
        e_string = file_loc + 'Eta_CLO_PIPM_100_test.npy'
        
        if nClust is not None:
            x_norm_str = 'X_CLO_PIPM_test_norm_{}.npy'.format(nClust)
            y_norm_str = 'Y_CLO_PIPM_test_norm_{}'.format(nClust)
            e_norm_str = 'Eta_CLO_PIPM_test_norm_{}'.format(nClust)
        elif nClust is None:
            x_norm_str = 'X_CLO_PIPM_100_test_norm.npy'
            y_norm_str = 'Y_CLO_PIPM_100_test_norm'
            e_norm_str = 'Eta_CLO_PIPM_100_test_norm'


############################################
## Load Raw Data - Make Copy Destinations ##
############################################
t0 = cput()
if nClust is None:
    Xraw = np.load(x_string, mmap_mode='r')
    Yraw = np.load(y_string, mmap_mode='r')
    Etaraw = np.load(e_string, mmap_mode='r')
    max_clust = True
    nClust = Xraw.shape[0]
    array_pts = Xraw.shape[1]
elif nClust is not None:
    Xraw = np.load(x_string, mmap_mode='r')[:nClust,:,:]
    Yraw = np.load(y_string, mmap_mode='r')[:nClust]
    Etaraw = np.load(e_string, mmap_mode='r')[:nClust]
    max_clust = False
    array_pts = Xraw.shape[1]

print('X shape: {}'.format(Xraw.shape))
print('Y shape: {}'.format(Yraw.shape))

X = np.lib.format.open_memmap(output_loc+x_norm_str,
                             mode='w+', dtype=np.float64, shape=(nClust,
                                                                 MAX_NPTS, 4))

Y = np.empty((Yraw.shape[0],))
t1 = cput()
load_time = t1 - t0

print('Time to load memory mapped data and new copy: {:8.6f} (s)'.format(t1-t0))
print()


# place this after loading and assume that the X and Y arrays take minimal
# time to load
if find_values:
    print('Determining values for normalization!')
    if test_bool:
        raise ValueError('Not intended to find values from test data.'\
                        +' Use --train flag instead.')
        
    t0 = cput()
    nz_mask = Xraw[:,:,3] != 0
    t1 = cput()
    mask_time = t1 - t0
    print('Time to solve for mask: {} (s)'.format(mask_time));print()
    
    t0 = cput()
    logX = np.log(Xraw[nz_mask,0])
    cellE_mean = np.mean(logX)
    cellE_std = np.std(logX)
    t1 = cput()
    cellE_time = t1 - t0
    print('Cell energy mean: {}'.format(cellE_mean))
    print('Cell energy std: {}'.format(cellE_std))
    print('Time: {} (s)'.format(cellE_time))
    
    t0 = cput()
    Eta_mean = np.mean(Xraw[nz_mask,1])
    Eta_std = np.std(Xraw[nz_mask,1])
    t1 = cput()
    Eta_time = t1 - t0
    print('Eta mean: {}'.format(Eta_mean))
    print('Eta std: {}'.format(Eta_std))
    print('Time: {} (s)'.format(Eta_time))
    
    t0 = cput()
    Phi_mean = np.mean(Xraw[nz_mask,2])
    Phi_std = np.std(Xraw[nz_mask,2])
    t1 = cput()
    Phi_time = t1 - t0
    print('Phi mean: {}'.format(Phi_mean))
    print('Phi std: {}'.format(Phi_std))
    print('Time: {} (s)'.format(Phi_time))
    
    t0 = cput()
    target_mean = np.mean(np.log(Yraw))
    t1 = cput()
    target_time = t1 - t0    
    print('Target mean: {}'.format(target_mean))
    print('Time: {} (s)'.format(target_time))
    
    if pi0_bool:
        if max_clust:
            filestr = 'pi0_CLO_values_full.txt'
        else:
            filestr = 'pi0_CLO_values_{}.txt'.format(nClust)
    elif pipm_bool:
        if max_clust:
            filestr = 'pipm_CLO_values_full.txt'
        else:
            filestr = 'pipm_CLO_values_{}.txt'.format(nClust)
        
    with open(filestr, 'w+') as f:
        if max_clust:
            f.write('##\n')
            f.write('# Number of clusters: {} (max)\n'.format(nClust))
        else:
            f.write('##\n')
            f.write('# Number of clusters: {}\n'.format(nClust))
        f.write('##\n')
        f.write('# Cell energy mean: {}\n'.format(cellE_mean))
        f.write('# Cell energy std: {}\n'.format(cellE_std))
        f.write('# Eta mean: {}\n'.format(Eta_mean))
        f.write('# Eta std: {}\n'.format(Eta_std))
        f.write('# Phi mean: {}\n'.format(Phi_mean))
        f.write('# Phi std: {}\n'.format(Phi_std))
        f.write('# Target mean: {}\n'.format(target_mean))
        f.write(str(cellE_mean)+'\n')
        f.write(str(cellE_std)+'\n')
        f.write(str(Eta_mean)+'\n')
        f.write(str(Eta_std)+'\n')
        f.write(str(Phi_mean)+'\n')
        f.write(str(Phi_std)+'\n')
        f.write(str(target_mean)+'\n')
    
    print()
    print('Finished determining values. Exiting code.');print()
    sys.exit()


#########################
## Hard Coded Averages ##
#########################
'''
Note: in order to balance the target for both charged and neutral pions
which will have different means of the log of the target, we determine this
value for each and average the two. This is determined to be
2.519134455627526. Thus we hard code in the value of 2.52 :-)

---------------------PIPM--------------------PI0-----------
cellE_mean  = (-2.234687302600965 + -2.1826265122783526) /2
cellE_std   = (1.9115307566123543 + 1.932786980244513)   /2
Eta_std     = (0.09803379949581109 + 0.05483370415405759)/2
Phi_std     = (0.6571351152071913 + 0.588562927404193)   /2
target_mean = (2.046657588606947 + 2.9916113226481054)   /2

Average cellE_mean: -2.208656907439659
Average cellE_std: 1.9221588684284336
Average Eta_std: 0.07643375182493434
Average Phi_std: 0.6228490213056921
Average target_mean: 2.519134455627526
'''
cellE_mean = -2.208656907439659
cellE_std = 1.9221588684284336
Eta_std = 0.07643375182493434
Phi_std = 0.6228490213056921
target_mean = 2.519134455627526

##############################
## NORMALIZE TARGET and Eta ##
##############################
print('normalizing target..')
t0 = cput()
Y = np.log(Yraw) - target_mean
t1 = cput()
target_time = t1 - t0

######################
## NORMALIZE INPUTS ##
######################
nDivide = 10 ## This is an arbitrary number to deal with memory in chunks
print('assigning zero elements')
t0 = cput()
nz_mask = np.zeros((Xraw.shape[0], Xraw.shape[1]), dtype=bool)
nz_mask[:,:] = Xraw[:,:,3] != 0.0

multiplier = np.floor(nClust/nDivide).astype(int)

for i in range(nDivide):
    l_idx = i*multiplier
    h_idx = (i+1)*multiplier
    if i == 9:
        h_idx = nClust
        
    nz_mask_sl = np.invert(nz_mask[l_idx:h_idx,:])
    X[l_idx:h_idx,:array_pts,:][nz_mask_sl,:] = 0.0
    X[l_idx:h_idx,array_pts:,:] = 0.0
    X.flush()

t1 = cput()
zero_elem_time = t1 - t0
print('{:6.2f} (m)'.format((zero_elem_time)/60));print()

print('normalizing inputs..')
t0 = cput()
## Normalize rPerp to 1/3630
X[:,:array_pts,:][nz_mask,3] = Xraw[nz_mask,3]/3630.

## Energy Values that are not zero
X[:,:array_pts,:][nz_mask,0] = np.log(Xraw[nz_mask,0])
X[:,:array_pts,:][nz_mask,0] = (X[:,:array_pts,:][nz_mask,0]
                                - cellE_mean)/cellE_std

## Eta and Phi
X[:,:array_pts,:][nz_mask,1] = Xraw[nz_mask,1]/Eta_std
X[:,:array_pts,:][nz_mask,2] = Xraw[nz_mask,2]/Phi_std

t1 = cput()
input_time = t1 - t0
print('Time to Normalize: {:6.2f} (m)'.format((input_time)/60))
print()


################################
## FLUSH MEMORY + SAVE ARRAYS ##
################################
t0 = cput()
print('flushing memory for X...')
X.flush()
t1 = cput()
flush_time = t1 - t0
print('Time to flush:    {:8.2f} (s)'.format(flush_time))
print('                  {:8.2f} (m)'.format(flush_time/60))
print()

t0 = cput()
print('Saving target and Eta')
np.save(output_loc + y_norm_str, Y)
np.save(output_loc + e_norm_str, Etaraw)
t1 = cput()
save_time = t1 - t0
print('Time to save: {:8.2f} (m)'.format(save_time/60))
print()

t_tot = load_time + target_time + zero_elem_time + input_time + flush_time\
        + save_time
print('Total time: {:8.2f} (m)'.format(t_tot/60))
print('Total time: {:8.2f} (h)'.format(t_tot/3600))
print()
print('..finished normalizing the CLO dataset..')
print()
