'''
ML4P Point Cloud Data Script
Author: Russell Bate
russell.bate@cern.ch
russellbate@phas.ubc.ca

Notes: Cluster only Data Script.
- Only cut is that the clusterE > 0'''


''' Check google document to align cuts with
other groups. Any cuts at all on clusters? '''


#====================
# Load Utils ========
#====================

import numpy as np
import uproot as ur
import awkward as ak
import time as t
from time import perf_counter as cput
import os
import argparse
from copy import deepcopy

Nfile=1

print()
print('='*37)
print('== Larger Data Cluster Only Script ==')
print('='*37)
print()
print("Awkward version: "+str(ak.__version__))
print("Uproot version: "+str(ur.__version__))
print("Numpy version: {}".format(np.__version__))


## Read in Parameters
#=============================================================================
parser = argparse.ArgumentParser(description='Inputs for CLO track script.')

parser.add_argument('--nFile', action="store", dest='nf', default=1,
                   type=int)

args = parser.parse_args()

Nfile = args.nf

print('Working on {} files'.format(Nfile))


#====================
# Functions =========
#====================

def DeltaR(coords, ref):
    ''' Straight forward function, expects Nx2 inputs for coords, 1x2 input for ref '''
    ref = np.tile(ref, (len(coords[:,0]), 1))
    DeltaCoords = np.subtract(coords, ref)
    ## Mirroring ##
    gt_pi_mask = DeltaCoords > np.pi
    lt_pi_mask = DeltaCoords < - np.pi
    DeltaCoords[lt_pi_mask] = DeltaCoords[lt_pi_mask] + 2*np.pi
    DeltaCoords[gt_pi_mask] = DeltaCoords[gt_pi_mask] - 2*np.pi
    return np.sqrt(DeltaCoords[:,0]**2 + DeltaCoords[:,1]**2)


def find_max_clust(indices, event_dict):
    ''' Designed to find the largest cluster size given
    event selection.
    Inputs: indices - list of (event, [cluster_indices]) '''
    max_clust = 0
    
    for evt, cl_idx in indices:

        cluster_nCells = event_dict['cluster_nCells'][evt].to_numpy()[cl_idx]
        
        max_evt_nCells = np.max(cluster_nCells)
        
        if max_evt_nCells > max_clust:
            max_clust = max_evt_nCells
    
    # return the shape of the maximum array size, (events, cluster)
    return max_clust


def dict_from_tree(tree, branches=None, np_branches=None):
    ''' Loads branches as default awkward arrays and np_branches as numpy arrays. '''
    dictionary = dict()
    if branches is not None:
        for key in branches:
            branch = tree.arrays()[key]
            dictionary[key] = branch
            
    if np_branches is not None:
        for np_key in np_branches:
            np_branch = np.ndarray.flatten(tree.arrays()[np_key].to_numpy())
            dictionary[np_key] = np_branch
    
    if branches is None and np_branches is None:
        raise ValueError("No branches passed to function.")
        
    return dictionary


def find_index_1D(values, dictionary):
    ''' Use a for loop and a dictionary. values are the IDs to search for. dict must be in format 
    (cell IDs: index) '''
    idx_vec = np.zeros(len(values), dtype=np.int32)
    for i in range(len(values)):
        idx_vec[i] = dictionary[values[i]]
    return idx_vec


#====================
# Metadata ==========
#====================
event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", 'cluster_nCells', "nCluster", "eventNumber",
                  "nTrack", "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", 'trackPt', 'trackP',
                  'trackMass', 'trackEta', 'trackPhi', 'truthPartE', 'cluster_ENG_CALIB_TOT', "cluster_E", 'truthPartPt']

ak_event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", "cluster_nCells",
                  "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", "trackPt", "trackP",
                  "trackMass", "trackEta", "trackPhi", "truthPartE", "cluster_ENG_CALIB_TOT", "cluster_E", "truthPartPt"]

np_event_branches = ["nCluster", "eventNumber", "nTrack", "nTruthPart"]

geo_branches = ["cell_geo_ID", "cell_geo_eta", "cell_geo_phi", "cell_geo_rPerp", "cell_geo_sampling"]


#====================
# File setup ========
#====================
# user.angerami.24559744.OutputStream._000001.root
fileNames = []
file_prefix = 'user.angerami.24559744.OutputStream._000'
for i in range(1,Nfile+1):
    endstring = f'{i:03}'
    fileNames.append(file_prefix + endstring + '.root')

    
#====================
# Load Data Files ===
#====================

## GEOMETRY DICTIONARY ##
geo_file = ur.open('/fast_scratch_1/atlas_images/v01-45/cell_geo.root')
CellGeo_tree = geo_file["CellGeo"]
geo_dict = dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

# cell geometry data
cell_geo_ID = geo_dict['cell_geo_ID']
cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

## MEMORY MAPPED ARRAY ALLOCATION ##
X_large = np.lib.format.open_memmap('/data/atlas/rbate/X_CLO_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,1500,6), fortran_order=False, version=None)
Y_large = np.lib.format.open_memmap('/data/atlas/rbate/Y_CLO_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,), fortran_order=False, version=None)
Eta_large = np.empty(2500000)


# Pre-Loop Definitions ##
#======================================
k = 1 # tally used to keep track of file number
ClN = 0 # used for keeping track of total number of events
max_nPoints = 0 # used for keeping track of the largest 'point cloud'
t_tot = 0 # total time
# for event dictionary
events_prefix = '/fast_scratch_1/atlas_images/v01-45/pipm/'


## Main File Loop ##
#======================================
for currFile in fileNames:
    
    # Check for file, a few are missing
    if not os.path.isfile(events_prefix+currFile):
        print()
        print('File '+events_prefix+currFile+' not found..')
        print()
        k += 1
        continue
    
    else:
        print()
        print('Working on File: '+str(currFile)+' - '+str(k)+'/'+str(Nfile))
        k += 1
        

    #===============#
    ## LOAD EVENTS ##
    #===============#
    t0 = cput()
    event = ur.open(events_prefix+currFile)
    event_tree = event["EventTree"]
    event_dict = dict_from_tree(tree=event_tree, branches=ak_event_branches, np_branches=np_event_branches)
    
    #==============#
    ## APPLY CUTS ##
    #==============#
    # create ordered list of events to use for index slicing
    nEvents = len(event_dict['eventNumber'])
    all_events = np.arange(0,nEvents,1,dtype=np.int32)
    
    # Cluster Cuts
    nCluster = event_dict['nCluster']
    clust_mask = nCluster > 0
    filtered_event = all_events[clust_mask]
    
    event_indices = []
    for evt in filtered_event:
        nClst = event_dict['nCluster'][evt]
        clst_idx = np.arange(nClst)
        clst_E = event_dict['cluster_E'][evt][clst_idx].to_numpy()
        energy_cut = clst_E > 0
        
        cluster_ENG_CALIB_TOT = event_dict['cluster_ENG_CALIB_TOT'][evt]\
            [clst_idx].to_numpy()
        truth_energy_cut = cluster_ENG_CALIB_TOT > 0
        
        energy_cuts = np.logical_and(energy_cut, truth_energy_cut)
        
        if np.count_nonzero(energy_cuts) > 0:
            cut_idx = clst_idx[energy_cuts]
            event_indices.append((evt, np.ndarray.copy(cut_idx)))
        
    t1 = cput()
    events_cuts_time = t1 - t0
    
    #=========================#
    ## MAX DIMS FOR CLUSTERS ##
    #=========================#
    t0 = cput()
    max_clust_curr = find_max_clust(event_indices, event_dict)

    # keep track of the largest point cloud to use for saving later
    if max_clust_curr > max_nPoints:
        max_nPoints = max_clust_curr
    t1 = cput()
        
    find_create_max_dims_time = t1 - t0    
    
    ##=================##
    ## Fill in Entries ##
    ##=================##
    t0 = cput()
    for evt, cluster_idc in event_indices:
        
        nClust = cluster_idc.shape[0]
        
        for c in cluster_idc:
            
            ## X array ##
            cluster_cell_ID = event_dict['cluster_cell_ID'][evt][c].to_numpy()
            nCellCurr = len(cluster_cell_ID)
            cluster_cell_E = event_dict['cluster_cell_E'][evt][c].to_numpy()

            cell_indices = find_index_1D(cluster_cell_ID, cell_ID_dict)
            
            eta_ctr = event_dict['cluster_Eta'][evt][c]
            phi_ctr = event_dict['cluster_Phi'][evt][c]
            
            cluster_cell_Eta = geo_dict['cell_geo_eta'][cell_indices] - eta_ctr
            cluster_cell_Phi = geo_dict['cell_geo_phi'][cell_indices] - phi_ctr
            cluster_cell_rPerp = geo_dict['cell_geo_rPerp'][cell_indices]
            
            X_large[ClN,:nCellCurr,0] = np.ndarray.copy(cluster_cell_E)
            X_large[ClN,:nCellCurr,1] = np.ndarray.copy(cluster_cell_Eta)
            X_large[ClN,:nCellCurr,2] = np.ndarray.copy(cluster_cell_Phi)
            X_large[ClN,:nCellCurr,3] = np.ndarray.copy(cluster_cell_rPerp)
            X_large[ClN,nCellCurr:,:] = 0.
            
            ## Y array ##
            cluster_ENG_CALIB_TOT = event_dict['cluster_ENG_CALIB_TOT'][evt][c]
            Y_large[ClN] = cluster_ENG_CALIB_TOT
            
            ## Eta ##
            Eta_large[ClN] = eta_ctr
            
            ## Cluster Counter ##
            ClN += 1
            
    t1 = cput()
    fill_entries_time = t1 - t0
    time_for_this_file = fill_entries_time + events_cuts_time +\
                         find_create_max_dims_time
    print()
    print('time for this file: {} (m)'.format((time_for_this_file)/60))
    print()
    t_tot += time_for_this_file

##===============================##
## COPY ARRAYS TO SMALLER FORMAT ##
##===============================##

print()
print('nClusters: {}'.format(ClN))
print('Point cloud size: {}'.format(max_nPoints))
print('total time: {} (h)'.format((t_tot)/3600));print()

print('Copying elements ..')
t0 = cput()

## X ##
X = np.lib.format.open_memmap('/data/atlas/rbate/X_CLO_'+str(Nfile)+'_files.npy',
                       mode='w+', dtype=np.float64, shape=(ClN,max_nPoints,4),
                       fortran_order=False, version=None)
np.copyto(src=X_large[:ClN,:max_nPoints,:4], dst=X, casting='same_kind',
         where=True)
del X_large
os.system('rm /data/atlas/rbate/X_CLO_large.npy')

## Y ##
Y = np.lib.format.open_memmap('/data/atlas/rbate/Y_CLO_'+str(Nfile)+'_files.npy',
                       mode='w+', dtype=np.float64, shape=(ClN,),
                       fortran_order=False, version=None)
np.copyto(src=Y_large[:ClN], dst=Y, casting='same_kind', where=True)
del Y_large
os.system('rm /data/atlas/rbate/Y_CLO_large.npy')

## Eta ##
Eta = np.save('/data/atlas/rbate/Eta_CLO_'+str(Nfile)+'_files', Eta_large[:ClN])
t1 = cput()

print('time to copy: {} (m)'.format((t1-t0)/60))

        
