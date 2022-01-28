'''
##============================##
### Rho Dataset Normalization ##
##============================##

author: Russell Bate
russellbate@phas.ubc.ca
'''

datapath_prefix = "/data/atlas/rbate/"
module_path = '/home/russbate/MLPionCollaboration/LCStudies/'
log_file = ""


print()
print('starting data normalization script')
print()

## General imports
#======================================
import numpy as np
import matplotlib.pyplot as plt
import time as t
from time import perf_counter as cput
import logging

## local ML Packages
import sys
sys.path.append(module_path)
from util import deep_set_util as dsu

## Load Data
#======================================
t0 = cput()
Xraw = np.load(datapath_prefix+'Rho_X_20_files.npy', mmap_mode='r')[:,:,:]
Yraw = np.load(datapath_prefix+'Rho_Y_segm_20_files.npy', mmap_mode='r')[:,:,:]

print('segmentation X data shape: {}'.format(Xraw.shape))
print('segmentation Y data shape: {}'.format(Yraw.shape))
print()

X = np.lib.format.open_memmap(datapath_prefix+'XR_norm.npy',
                             mode='w+', dtype=np.float32, shape=(Xraw.shape[0], Xraw.shape[1], 4))

nz_mask = (Yraw[:,:,0] + Yraw[:,:,1]) != 0

# Make sure that non-zero elements are copied as zeros due to mis-match
X[np.invert(nz_mask),:] = 0

t1 = cput()
print()
print('Time to load memory mapped data: {} (s)'.format(t1-t0))
print()

## Create Target
#======================================
t0 = cput()
target_ratio = np.zeros(nz_mask.shape)

target_ratio[nz_mask] = Yraw[nz_mask,0] / (Yraw[nz_mask,0] + Yraw[nz_mask,1])

Y = np.atleast_3d(target_ratio)
t1 = cput()
print('Time to create targets: {} (s)'.format(t1-t0))
print()

## Point Normalization
#======================================
t0 = cput()
X[:,:,1:4] = dsu.to_xyz(np.ndarray.copy(Xraw[:,:,1:4]), nz_mask)
t1 = cput()

## ENERGY ##
log_E_mask = Xraw[:,:,0] > 0
X[log_E_mask,0] = np.log(np.ndarray.copy(Xraw[log_E_mask,0]))

## X ##
X[:,:,1] = np.ndarray.copy(Xraw[:,:,1])/3000

## Y ##
X[:,:,2] = np.ndarray.copy(Xraw[:,:,2])/1000

## Z ##
X[:,:,3] = np.ndarray.copy(Xraw[:,:,3])/1000
t2 = cput()

# Save target
np.save(datapath_prefix+'YR_segm_norm', Y)
t3 = cput()

print()
print('Time to convert to xyz: {} (s)'.format(t1-t0))
print('Time to normalize: {} (s)'.format(t2-t1))
print('Time to save target: {} (s)'.format(t3-t2))
print('Total time: {} (s)'.format(t3-t0))
print('Target shape: {}'.format(Y.shape))
print('Input shape: {}'.format(X.shape))
print()
print('Finished normalizing rho dataset!');print()

