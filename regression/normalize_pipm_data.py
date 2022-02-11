'''
##====================##
## PIPM NORMALIZATION ##
##====================##
- Scripts for normalizing the pipm dataset

author: Russell Bate
russellbate@phas.ubc.ca
'''

datapath_prefix = "/data/atlas/rbate/"
module_path = '/home/russbate/MLPionCollaboration/LCStudies/'

print()
print('starting data normalization for pipm..');print()

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
Xraw = np.load(datapath_prefix+'PIPM_X_35_files.npy', mmap_mode='r')[:,:,:]
Yseg = np.load(datapath_prefix+'PIPM_Y_segm_35_files.npy', mmap_mode='r')[:,:,:]
Yreg = np.load(datapath_prefix+'PIPM_Y_regr_35_files.npy', mmap_mode='r')[:,:]
print(Xraw.shape)
print(Yseg.shape)
print(Yreg.shape)
t1 = cput()

X = np.lib.format.open_memmap('/data/atlas/rbate/X_PPM_EM_notebook.npy',
                             mode='w+', dtype=np.float64, shape=(Xraw.shape[0], Xraw.shape[1], 5))

Yem = np.lib.format.open_memmap('/data/atlas/rbate/Y_PPM_EM_notebook.npy',
                             mode='w+', dtype=np.float64, shape=Yseg.shape)
np.copyto(dst=Yem, src=Yseg, casting='same_kind', where=True)

# Energy_EM = np.ndarray.copy(Yraw[:,:,0])
# Energy_nonEM = np.ndarray.copy(Yraw[:,:,1])
nz_mask = Yseg[:,:,0] + Yseg[:,:,1] != 0 
print(np.shape(nz_mask))
print(np.count_nonzero(nz_mask.flatten()))

ratio = np.zeros((X.shape[0], X.shape[1]))
ratio[nz_mask] = Yem[nz_mask,0]/(Yem[nz_mask,0] + Yem[nz_mask,1])
ratio = ratio*2 - 1
print(ratio.shape)

X[nz_mask,:4] = np.ndarray.copy(Xraw[nz_mask,:4])
X[nz_mask,4] = np.ndarray.copy(ratio[nz_mask])
# np.copyto(dst=X[:,:,:4], src=Xraw[:,:,:4], casting='same_kind', where=nz_mask)
# np.copyto(dst=X[:,:,4], src=ratio, casting='same_kind', where=nz_mask)
# Make sure that non-zero elements are copied as zeros due to mis-match
X[np.invert(nz_mask),:] = 0

t1 = cput()

print('Time to load memory mapped data: {} (s)'.format(t1-t0))


## NORMALIZE TARGET
Y = np.ndarray.copy(np.log(Yreg[:,0]))


## NORMALIZE INPUTS
t0 = cput()
## Normalize rPerp to 1/3630
# rPerp_mask = X[nz_mask,3] != 0
X[nz_mask,3] = X[nz_mask,3]/3630.

## Energy Values that are not zero! This should coincide with the EM vals...
X[nz_mask,0] = np.log(X[nz_mask,0])
cellE_mean = np.mean(X[nz_mask,0])
cellE_std = np.std(X[nz_mask,0])
X[nz_mask,0] = (X[nz_mask,0] - cellE_mean)/cellE_std

## Eta and Phi
# eta_mask = X[:,:,1] != 0
X[nz_mask,1] = X[nz_mask,1]/.7

# phi_mask = X[:,:,2] != 0
cellPhi_std = np.std(X[nz_mask,2])
X[nz_mask,2] = X[nz_mask,2]/cellPhi_std
t1 = cput()

print('Time to Normalize: {} (m)'.format((t1-t0)/60))
