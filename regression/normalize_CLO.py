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
import time
from time import process_time as cput
import os
cwd = os.getcwd()

## Local ML Packages
import sys
sys.path.append(module_path)
from util import deep_set_util as dsu


## Load Raw Data
#======================================
t0 = cput()
Xraw = np.load(datapath_prefix+'X_CLO_50_files.npy', mmap_mode='r')
Yraw = np.load(datapath_prefix+'Y_CLO_50_files.npy', mmap_mode='r')
Etaraw = np.load(datapath_prefix+'Eta_CLO_50_files.npy', mmap_mode='r')
# EtaRaw = np.load(datapath_prefix+'Eta_STMC_v2_25_files.npy')
print('X shape: {}'.format(Xraw.shape))
print('Y shape: {}'.format(Yraw.shape))

X = np.lib.format.open_memmap(datapath_prefix+'X_CLO_tmp.npy',
                             mode='w+', dtype=np.float64, shape=(Xraw.shape[0], Xraw.shape[1], 4))

Y = np.lib.format.open_memmap(datapath_prefix+'Y_CLO_tmp.npy',
                             mode='w+', dtype=np.float64, shape=(Yraw.shape[0],))
Eta = np.empty((Yraw.shape[0],))
t1 = cput()
load_time = t1 - t0

print('Time to load memory mapped data and copy: {:8.6f} (s)'.format(t1-t0))

## NORMALIZE TARGET
print()
print('normalizing target..')
t0 = cput()
Y_log_mean = np.mean(np.log(Yraw))
print('Y_log_mean: ')
print(Y_log_mean); print()
with open(cwd+'/CLO_50_files_Y_logmean.txt', 'w+') as f:
    f.write('{}'.format(Y_log_mean))
Y = np.log(Yraw) - Y_log_mean
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
''' not sure why divide by zero errors are encountered here,
debug later? Not concerned for now... '''
eta_std = np.std(Xraw[nz_mask,1])
print('eta standard deviation: {}'.format(eta_std))
X[nz_mask,1] = np.ndarray.copy(Xraw[nz_mask,1]/eta_std)


# phi_mask = X[:,:,2] != 0
cellPhi_std = np.std(Xraw[nz_mask,2])
X[nz_mask,2] = np.ndarray.copy(Xraw[nz_mask,2]/cellPhi_std)

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
Xfin = np.lib.format.open_memmap('/fast_scratch_1/atlas/X_CLO_50_norm.npy',
                             mode='w+', dtype=np.float64, shape=(Xraw.shape[0],
                                                         Xraw.shape[1], 4))

Yfin = np.lib.format.open_memmap('/fast_scratch_1/atlas/Y_CLO_50_norm.npy',
                             mode='w+', dtype=np.float64, shape=(Yraw.shape[0],))

np.copyto(src=X[indices,:,:], dst=Xfin, casting='same_kind', where=True)
np.copyto(src=Y[indices], dst=Yfin, casting='same_kind', where=True)
np.copyto(src=Etaraw[indices], dst=Eta, casting='same_kind', where=True)

np.save('/fast_scratch_1/atlas/Eta_CLO_50_norm', Eta)

del X
del Y
os.system('rm '+datapath_prefix+'X_CLO_tmp.npy')
os.system('rm '+datapath_prefix+'Y_CLO_tmp.npy')
t1 = cput()
save_time = t1 - t0

print()
print('time to copy shuffled files: {:8.2f} (m)'.format((t1-t0)/60))
print()
print('Total time: {:8.2f} (m)'.format((load_time+target_time+zero_elem_time\
                         +input_time+shuffle_time+save_time)/60))
print()
print('finished normalizing the CLO dataset for 50 files')
print()
