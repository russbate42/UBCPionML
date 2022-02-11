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
print('starting data normalization for STMC..');print()

## General Imports
#======================================
import numpy as np
import time
from time import process_time as cput

## Local ML Packages
import sys
sys.path.append(module_path)
from util import deep_set_util as dsu


## Load Raw Data
#======================================
t0 = cput()
Xraw = np.load(datapath_prefix+'X_STMC_502_files.npy', mmap_mode='r')[:,:,:]
Yraw = np.load(datapath_prefix+'Y_STMC_502_files.npy', mmap_mode='r')[:,:]
print('X shape: {}'.format(Xraw.shape))
print('Y shape: {}'.format(Yraw.shape))

X = np.lib.format.open_memmap('/data/atlas/rbate/X_STMC_full_norm.npy',
                             mode='w+', dtype=np.float64, shape=(Xraw.shape[0], Xraw.shape[1], 5))

Y = np.lib.format.open_memmap('/data/atlas/rbate/Y_STMC_full_norm.npy',
                             mode='w+', dtype=np.float64, shape=(Yraw.shape[0],))
t1 = cput()

print('Time to load memory mapped data and copy: {} (s)'.format(t1-t0))

## NORMALIZE TARGET
print()
print('normalizing target..')
Y[:] = np.ndarray.copy(np.log(Yraw[:,0]))
print('normalized target..');print()

## NORMALIZE INPUTS
nz_mask = Xraw[:,:,3] != 0

print('normalizing inputs..')
t0 = cput()
## Normalize rPerp to 1/3630
# rPerp_mask = X[nz_mask,3] != 0
X[nz_mask,3] = np.ndarray.copy(Xraw[nz_mask,3]/3630.)

## Energy Values that are not zero! This should coincide with the EM vals...
X[nz_mask,0] = np.ndarray.copy(np.log(Xraw[nz_mask,0]))
cellE_mean = np.mean(X[nz_mask,0])
cellE_std = np.std(X[nz_mask,0])
X[nz_mask,0] = (X[nz_mask,0] - cellE_mean)/cellE_std

## Eta and Phi
# eta_mask = X[:,:,1] != 0
X[nz_mask,1] = np.ndarray.copy(Xraw[nz_mask,1]/.7)

# phi_mask = X[:,:,2] != 0
cellPhi_std = np.std(Xraw[nz_mask,2])
X[nz_mask,2] = np.ndarray.copy(Xraw[nz_mask,2]/cellPhi_std)
t1 = cput()

print('Time to Normalize: {} (m)'.format((t1-t0)/60))

