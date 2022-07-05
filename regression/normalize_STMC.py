'''
##====================##
## STMC NORMALIZATION ##
##====================##
- Normalize the STMC dataset

author: Russell Bate
russellbate@phas.ubc.ca
'''

datapath_prefix = "/data/atlas/rbate/"
module_path = '/home/russbate/MLPionCollaboration/LCStudies/'

print()
print('starting data normalization for STMC..v2..');print()

## General Imports
#===============w=======================
import numpy as np
import os, sys, time, argparse
from time import process_time as cput
cwd = os.getcwd()

## Local ML Packages
sys.path.append(module_path)
from util import deep_set_util as dsu


## Declaration of size!
''' This is determined by rounding to the nearest 100
of the largest number of points either training or evaluation arrays '''
MAX_NPTS = 1600


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
parser.add_argument('--find_values', action='store_true',
                    dest='fv')
parser.add_argument('--nEvent', action='store', dest='ne',
                    default=None, type=int)
args = parser.parse_args()

file_loc = args.fl
output_loc = args.ol
train_bool = args.train
test_bool = args.test
find_values = args.fv
nEvent = args.ne


if not train_bool and not test_bool:
    sys.exit('Need to specify --train or --test. Exiting.')
    
elif train_bool and test_bool:
    sys.exit('Cannot specify --train and --test. Exiting.')


if train_bool:
    print('Normalizing training data for STMC..')
    x_string = file_loc + 'X_STMC_v2_400_train.npy'
    y_string = file_loc + 'Y_STMC_v2_400_train.npy'
    e_string = file_loc + 'Eta_STMC_v2_400_train.npy'
    
    if nEvent is not None:
        x_norm_str = 'X_STMC_v2_train_norm_{}.npy'.format(nEvent)
        y_norm_str = 'Y_STMC_v2_train_norm_{}.npy'.format(nEvent)
        e_norm_str = 'Eta_STMC_v2_train_norm_{}.npy'.format(nEvent)
    elif nEvent is None:
        x_norm_str = 'X_STMC_v2_400_train_norm.npy'
        y_norm_str = 'Y_STMC_v2_400_train_norm.npy'
        e_norm_str = 'Eta_STMC_v2_400_train_norm.npy'

if test_bool:
    print('Normalizing test data for STMC..')
    x_string = file_loc + 'X_STMC_v2_100_test.npy'
    y_string = file_loc + 'Y_STMC_v2_100_test.npy'
    e_string = file_loc + 'Eta_STMC_v2_100_test.npy'
    
    if nEvent is not None:
        x_norm_str = 'X_STMC_v2_test_norm_{}.npy'.format(nEvent)
        y_norm_str = 'Y_STMC_v2_test_norm_{}.npy'.format(nEvent)
        e_norm_str = 'Eta_STMC_v2_test_norm_{}.npy'.format(nEvent)
    elif nEvent is None:
        x_norm_str = 'X_STMC_v2_100_test_norm.npy'
        y_norm_str = 'Y_STMC_v2_100_test_norm.npy'
        e_norm_str = 'Eta_STMC_v2_100_test_norm.npy'
    

############################################
## Load Raw Data - Make Copy Destinations ##
############################################
t0 = cput()
if nEvent is None:
    Xraw = np.load(x_string, mmap_mode='r')
    Yraw = np.load(y_string, mmap_mode='r')
    Etaraw = np.load(e_string, mmap_mode='r')
    max_event = True
    nEvent = Xraw.shape[0]
    array_pts = Xraw.shape[1]
elif nEvent is not None:
    Xraw = np.load(x_string, mmap_mode='r')[:nEvent,:,:]
    Yraw = np.load(y_string, mmap_mode='r')[:nEvent,0]
    Etaraw = np.load(e_string, mmap_mode='r')[:nEvent]
    max_event = False
    array_pts = Xraw.shape[1]

print('X shape: {}'.format(Xraw.shape))
print('Y shape: {}'.format(Yraw.shape))

X = np.lib.format.open_memmap(output_loc+x_norm_str,
                              mode='w+', dtype=np.float64,
                              shape=(nEvent, MAX_NPTS, 5)
                             )
Y = np.empty((Yraw.shape[0],))
t1 = cput()
load_time = t1 - t0

print('Time to load memory mapped data and new copy: {:8.6f} (s)'.format(t1-t0))
print()

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
    
    if max_event:
        filestr = 'STMC_values_full.txt'
    else:
        filestr = 'STMC_values_{}.txt'.format(nEvent)
    
    with open(filestr, 'w+') as f:
        if max_event:
            f.write('##\n')
            f.write('# Number of events: {} (max)\n'.format(nEvent))
        else:
            f.write('##\n')
            f.write('# Number of events: {}\n'.format(nEvent))
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
Note: This is considerably easier for STMC :-)

Number of events: 4106533 (max)
Cell energy mean: -2.1388742753256067
Cell energy std: 2.1083446472270846
Eta mean: 0.00021436696834480691
Eta std: 0.2436809732714367
Phi mean: 0.0008612312370200947
Phi std: 0.7352875085115939
Target mean: 4.427552853750981
'''

cellE_mean = -2.1388742753256067
cellE_std = 2.1083446472270846
Eta_std = 0.2436809732714367
Phi_std = 0.7352875085115939
target_mean = 4.427552853750981

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
nz_mask = np.zeros((nEvent, array_pts), dtype=bool)
nz_mask[:,:] = Xraw[:,:,3] != 0.0

multiplier = np.floor(nEvent/nDivide).astype(int)

for i in range(nDivide):
    l_idx = i*multiplier
    h_idx = (i+1)*multiplier
    if i == 9:
        h_idx = nEvent
        
    nz_mask_sl = np.invert(nz_mask[l_idx:h_idx,:])
    X[l_idx:h_idx,:array_pts,:][nz_mask_sl,:] = 0.0
    X[l_idx:h_idx,array_pts:,:] = 0.0 # zero padding
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

## Track flag
X[:,:array_pts,:][nz_mask,4] = Xraw[nz_mask,4]

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
print('..finished normalizing the STMC dataset..')
print()
    
        
        
'''
## NORMALIZE TARGET
print()
print('normalizing target..')
t0 = cput()
Y_log_mean = np.mean(np.log(Yraw[:,0]))
print('Y_log_mean: ')
print(Y_log_mean); print()
with open(cwd+'/STMC_v2_25_files_Y_logmean.txt', 'w+') as f:
    f.write('{}'.format(Y_log_mean))
Y = np.log(Yraw[:,0]) - Y_log_mean
t1 = cput()
print('{:6.2f} (m)'.format((t1-t0)/60));print()
print('normalized target..');print()
target_time = t1 - t0

## NORMALIZE INPUTS
print('assigning zero elements')
t0 = cput()
nz_mask = Xraw[:,:,3] != 0
X[np.invert(nz_mask),:] = 0
t1 = cput()
print('{:6.2f} (m)'.format((t1-t0)/60));print()
zero_elem_time = t1 - t0

print('normalizing inputs..')
t0 = cput()
## Normalize rPerp to 1/3630
# rPerp_mask = X[nz_mask,3] != 0
X[nz_mask,3] = np.ndarray.copy(Xraw[nz_mask,3]/3630.)

## Energy Values that are not zero! This should coincide with the EM vals...
X[nz_mask,0] = np.log(Xraw[nz_mask,0])
cellE_mean = np.mean(X[nz_mask,0])
cellE_std = np.std(X[nz_mask,0])
X[nz_mask,0] = (X[nz_mask,0] - cellE_mean)/cellE_std

## Eta and Phi
# eta_mask = X[:,:,1] != 0
'''

''' not sure why divide by zero errors are encountered here,
debug later? Not concerned for now... '''

'''
eta_std = np.std(Xraw[nz_mask,1])
print('eta standard deviation: {}'.format(eta_std))
X[nz_mask,1] = np.ndarray.copy(Xraw[nz_mask,1]/eta_std)

# phi_mask = X[:,:,2] != 0
cellPhi_std = np.std(Xraw[nz_mask,2])
X[nz_mask,2] = np.ndarray.copy(Xraw[nz_mask,2]/cellPhi_std)

# Track Flag!
X[nz_mask,4] = np.ndarray.copy(Xraw[nz_mask,4])

t1 = cput()
print('Time to Normalize: {:6.2f} (m)'.format((t1-t0)/60))
print()
input_time = t1 - t0

## Shuffle Data
print('shuffling indices..')
t0 = cput()
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
t1 = cput()
print('time shuffling indices: {:6.2f} (m)'.format((t1-t0)/60))
print()
shuffle_time = t1 - t0

print('Saving shuffled data..')
t0 = cput()
Xfin = np.lib.format.open_memmap('/fast_scratch_1/atlas/X_STMC_v2_25_norm2.npy',
                             mode='w+', dtype=np.float64, shape=(Xraw.shape[0],
                                                         Xraw.shape[1], 5))

Yfin = np.lib.format.open_memmap('/fast_scratch_1/atlas/Y_STMC_v2_25_norm2.npy',
                             mode='w+', dtype=np.float64, shape=(Yraw.shape[0],))

np.copyto(src=X[indices,:,:], dst=Xfin, casting='same_kind', where=True)
np.copyto(src=Y[indices], dst=Yfin, casting='same_kind', where=True)

EtaFin = np.ndarray.copy(EtaRaw[indices])
np.save('/fast_scratch_1/atlas/Eta_STMC_v2_25_norm2', EtaFin)

del X
del Y
os.system('rm '+datapath_prefix+'X_STMC_tmp.npy')
os.system('rm '+datapath_prefix+'Y_STMC_tmp.npy')
t1 = cput()
save_time = t1 - t0

print()
print('time to copy shuffled files: {:8.2f} (m)'.format((t1-t0)/60))
print()
print('Total time: {:8.2f} (m)'.format((load_time+target_time+zero_elem_time\
                         +input_time+shuffle_time+save_time)/60))
print()
print('finished normalizing the STMC v2 dataset')
print()
'''


