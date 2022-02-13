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
import os

## Local ML Packages
import sys
sys.path.append(module_path)
from util import deep_set_util as dsu


## Load Raw Data
#======================================
t0 = cput()
Xraw = np.load(datapath_prefix+'PIPM_X_35_files.npy', mmap_mode='r')[:,:,:]
Yseg_raw = np.load(datapath_prefix+'PIPM_Y_segm_35_files.npy', mmap_mode='r')[:,:,:]
Yreg_raw = np.load(datapath_prefix+'PIPM_Y_regr_35_files.npy', mmap_mode='r')[:,:]
print('X shape: {}'.format(Xraw.shape))
print('Y segmentation shape: {}'.format(Yseg_raw.shape))
print('Y regression shape: {}'.format(Yreg_raw.shape))

X = np.lib.format.open_memmap(datapath_prefix+'X_PIPM_tmp.npy',
                             mode='w+', dtype=np.float64, shape=(Xraw.shape[0], Xraw.shape[1], 5))

Yem = np.lib.format.open_memmap(datapath_prefix+'Y_PPM_em_tmp.npy',
                                mode='w+', dtype=np.float64,
                                shape=(Yseg_raw.shape[0], Yseg_raw.shape[1]))

Yrg = np.lib.format.open_memmap(datapath_prefix+'Y_PPM_rg_tmp.npy',
                                mode='w+', dtype=np.float64,
                                shape=(Yreg_raw.shape[0],))

t1 = cput()
print('Time to load memory mapped data: {} (s)'.format(t1-t0));print()


print('normalizing target..');print()
## NORMALIZE TARGET
Yrg = np.log(Yreg_raw[:,0])


## Masking and EM variable
t0 = cput()
print('masking, setting zeros, handling EM variable..')
nz_mask = Yseg_raw[:,:,0] + Yseg_raw[:,:,1] != 0

Yem[np.invert(nz_mask)] = 0.
X[np.invert(nz_mask),:] = 0.

Yem[nz_mask] = 2*(Yseg_raw[nz_mask,0]/(Yseg_raw[nz_mask,0] + Yseg_raw[nz_mask,1])) - 1

X[nz_mask,4] = Yem[nz_mask]
t1 = cput()
print('{:8.2f} (m)'.format((t1-t0)/60))
print()


## NORMALIZE INPUTS
t0 = cput()
print('normalizing inputs..')
## Normalize rPerp to 1/3630
# rPerp_mask = X[nz_mask,3] != 0
X[nz_mask,3] = Xraw[nz_mask,3]/3630.

## Energy Values that are not zero! This should coincide with the EM vals...
X[nz_mask,0] = np.log(Xraw[nz_mask,0])
cellE_mean = np.mean(X[nz_mask,0])
cellE_std = np.std(X[nz_mask,0])
X[nz_mask,0] = (X[nz_mask,0] - cellE_mean)/cellE_std

## Eta and Phi
# eta_mask = X[:,:,1] != 0
X[nz_mask,1] = Xraw[nz_mask,1]/.7

# phi_mask = X[:,:,2] != 0
cellPhi_std = np.std(Xraw[nz_mask,2])
X[nz_mask,2] = Xraw[nz_mask,2]/cellPhi_std
t1 = cput()
print('Time to Normalize: {:8.2f} (m)'.format((t1-t0)/60))

## Shuffle Data
print('shuffling indices..')
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
print()

''' Notes: this may be left over from troubleshooting empty mem-map files.
Try to see if this step can be eliminated by writing to the new file directly.
Although now the saving is done shuffled so it might be worth it...
'''
print('Saving shuffled data..')

## Copy to final destination
t0 = cput()
Xfin = np.lib.format.open_memmap('/fast_scratch/atlas/X_PIPM_full_norm.npy',
                             mode='w+', dtype=np.float64, shape=(Xraw.shape[0], Xraw.shape[1], 5))

Yfin = np.lib.format.open_memmap('/fast_scratch/atlas/Y_PIPM_full_norm.npy',
                             mode='w+', dtype=np.float64, shape=(Yreg_raw.shape[0],))

np.copyto(src=X[indices,:,:], dst=Xfin, casting='same_kind', where=True)
np.copyto(src=Yrg[indices], dst=Yfin, casting='same_kind', where=True)

del X
del Yrg
del Yem
os.system('rm '+datapath_prefix+'X_PIPM_tmp.npy')
os.system('rm '+datapath_prefix+'Y_PPM_rg_tmp.npy')
os.system('rm '+datapath_prefix+'Y_PPM_em_tmp.npy')
t1 = cput()
print()
print('time to copy files: {:8.2f} (m)'.format((t1-t0)/60))
print()
print('finished normalizing the PIPM dataset')
print()

