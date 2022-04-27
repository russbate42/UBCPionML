'''
##======================================##
## SPLIT STMC INTO TRACKS/CLUSTERS ONLY ##
##======================================##
- Split this dataset into two separate datasets post
normalization

author: Russell Bate
russellbate@phas.ubc.ca
'''

filepath_prefix = '/fast_scratch_1/atlas/'

print()
print('Starting Splitting STMC into Clusters and Tracks Separately')
print()

## General Imports
#======================================
import numpy as np
import time
from time import process_time as cput
import os
cwd = os.getcwd()

X = np.load(filepath_prefix+'X_STMC_v2_25_norm2.npy', mmap_mode='r')

# loop through each event
nEvts = X.shape[0]

## Clusters ##
# - 10 for tracks, -1 for eliminating track flag
xClustDims = (X.shape[0], X.shape[1]-10, 4)
X_clust = np.lib.format.open_memmap(filepath_prefix+'X_STMC_v2_Cl_25.npy',
                                mode='w+', dtype=np.float64, shape=xClustDims)

## Tracks ##
xTrackDims = (X.shape[0], 10, 4)
X_track = np.lib.format.open_memmap(filepath_prefix+'X_STMC_v2_Tr_25.npy',
                                mode='w+', dtype=np.float64, shape=xTrackDims)

print('Loaded files and mem-maps, now copying ..');print()

_25 = False
_50 = False
_75 = False

t0 = cput()
for i in range(nEvts):
    
    # Progress!
    perc_compl = float(i/nEvts)
    if perc_compl > .25 and _25 == False:
        _25 = True
        print('25% complete!');print()
    elif perc_compl > .5 and _50 == False:
        _50 = True
        print('50% complete!');print()
    elif perc_compl > .75 and _75 == False:
        _75 = True
        print('75% complete!');print()
        
    # tracks
    tmask = X[i,:,4] == 1
    nTrack = np.count_nonzero(tmask)
    X_track[i,:nTrack,:4] = np.ndarray.copy(X[i,tmask,:4])
    X_track[i,nTrack:,:4] = 0.0
    
    t_idx = np.argmax(tmask)
    
    # clusters
    # we can get away with this for now because tracks are saved after clusters
    X_clust[i,:t_idx,:4] = np.ndarray.copy(X[i,:t_idx,:4])
    X_clust[i,t_idx:,:4] = 0.0

t1 = cput()

print('Time to split: {:8.4f} (m)'.format((t1-t0)/60));print()
