
#====================
# Load Utils ========
#====================

import numpy as np
import uproot as ur
import awkward as ak
import time as t
import os
print("Awkward version: "+str(ak.__version__))
print("Uproot version: "+str(ur.__version__))

import sys
sys.path.append('')


#====================
# Import functions ==
#====================

import sys
sys.path.append('/home/russbate/MLPionCollaboration/LCStudies/util/')
import deep_set_util as dsu

# Import relevant branches
from deep_set_util import track_branches, event_branches, ak_event_branches, np_event_branches, geo_branches


#====================
# File setup ========
#====================
# user.angerami.24559744.OutputStream._000001.root
# Number of files
Nfile = 40
fileNames = []
file_prefix = 'user.angerami.24409109.OutputStream._000'
for i in range(1,Nfile+1):
    endstring = f'{i:03}'
    fileNames.append(file_prefix + endstring + '.root')

print(fileNames)


#====================
# Load Data Files ===
#====================

## GEOMETRY DICTIONARY ##
geo_file = ur.open('/fast_scratch/atlas_images/v01-45/cell_geo.root')
CellGeo_tree = geo_file["CellGeo"]
geo_dict = dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

# cell geometry data
cell_geo_ID = geo_dict['cell_geo_ID']
cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

# additional geometry data
layer_rPerp = np.array([1540., 1733., 1930., 2450., 3010., 3630.])
track_sample_layer = np.array([1,2,3,12,13,14])

# for event dictionary
events_prefix = '/fast_scratch/atlas_images/v01-45/rho/'

# Use this to compare with the dimensionality of new events
firstArray = True

## MEMORY MAPPED ARRAY ALLOCATION ##
# XR_large = np.lib.format.open_memmap('/data/rbate/XR_large.npy', mode='w+', dtype=np.float64,
#                        shape=(1700000,1500,6), fortran_order=False, version=None)
# YR_large = np.lib.format.open_memmap('/data/rbate/YR_large.npy', mode='w+', dtype=np.float64,
#                        shape=(1700000,3), fortran_order=False, version=None)
