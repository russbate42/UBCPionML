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
import os, sys, argparse, json
from copy import deepcopy

sys.path.append('/home/russbate/MLPionCollaboration/LCStudies')
from util import deep_set_util as dsu
from util.deep_set_util import dict_from_tree, DeltaR, find_max_dim_tuple
from util.deep_set_util import find_index_1D, find_max_clust

Nfile=1

print()
print('='*37)
print('== Larger Data Cluster Only Script ==')
print('='*37)
print()
print('== Generating data for PubNote 2022 ==')
print()
print("Awkward version: "+str(ak.__version__))
print("Uproot version: "+str(ur.__version__))
print("Numpy version: {}".format(np.__version__))


## Read in Parameters
#=============================================================================
parser = argparse.ArgumentParser(description='Inputs for CLO track script.')

parser.add_argument('--nFile', action="store", dest='nf', default=1,
                   type=int)
parser.add_argument('--file_loc', action="store", dest='fl', default=\
                    '/fast_scratch_1/atlas_images/v01-45/',
                   type=str)
parser.add_argument('--output_loc', action="store", dest='ol', default=\
                    '/fast_scratch_1/atlas/staged',
                   type=str)
parser.add_argument('--geo_loc', action="store", dest='gl', default=\
                    '/fast_scratch_1/atlas_images/v01-45/cell_geo.root',
                    type=str)
parser.add_argument('--train', action='store_true',
                    dest='train')
parser.add_argument('--test', action='store_true',
                    dest='test')
parser.add_argument('--pipm', action='store_true',
                    dest='pipm')
parser.add_argument('--pi0', action='store_true',
                    dest='pi0')
args = parser.parse_args()

Nfile = args.nf
file_loc = args.fl
output_loc = args.ol
geo_loc = args.gl
train_bool = args.train
test_bool = args.test
pipm_bool = args.pipm
pi0_bool = args.pi0

if not train_bool and not test_bool:
    sys.exit('Need to specify --train or --test. Exiting.')
    
elif train_bool and test_bool:
    sys.exit('Cannot specify --train and --test. Exiting.')

if not pi0_bool and not pipm_bool:
    sys.exit('Need to specify --pipm or pi0. Exiting')

elif pipm_bool and pi0_bool:
    sys.exit('Cannor specify --pipm and --pi0. Exiting')

print('Working on {} files'.format(Nfile))
if train_bool:
    if pi0_bool:
        print('Creating training data for pi0')
    if pipm_bool:
        print('Creating training data for pipm')
if test_bool:
    if pi0_bool:
        print('Creating test data for pi0')
    if pipm_bool:
        print('Creating test data for pipm')

# file location stuff
if pi0_bool:
    atlas_image_subfolder = 'pi0/'
elif pipm_bool:
    atlas_image_subfolder = 'pipm/'

#====================
# Metadata ==========
#====================
event_branches = dsu.event_branches
ak_event_branches = dsu.ak_event_branches
np_event_branches = dsu.np_event_branches
geo_branches = dsu.event_branches


#======================================
# Track related meta-data
#======================================
geo_branches = dsu.geo_branches
eta_trk_dict = dsu.eta_trk_dict
calo_dict = dsu.calo_dict
z_calo_dict = dsu.z_calo_dict
r_calo_dict = dsu.r_calo_dict
trk_proj_eta = dsu.trk_proj_eta
trk_proj_phi = dsu.trk_proj_phi
trk_em_eta = dsu.trk_em_eta
trk_em_phi = dsu.trk_em_phi
calo_numbers = dsu.calo_numbers
calo_layers = dsu.calo_layers
fixed_z_numbers = dsu.fixed_z_numbers
fixed_r_numbers = dsu.fixed_r_numbers


#====================
# JSON Files ========
#====================
root_files = []
if train_bool:
    if pipm_bool:
        with open('/home/russbate/MLPionCollaboration/LCStudies/clusters/pion_train.json', 'r') as f:
            files = json.load(f)

            for file in files:
                file_end = os.path.basename(file)
                root_files.append(file_end)

        with open('/home/russbate/MLPionCollaboration/LCStudies/clusters/pion_val.json', 'r') as f:
            files = json.load(f)

            for file in files:
                file_end = os.path.basename(file)
                root_files.append(file_end)
                
    if pi0_bool:
        with open('/home/russbate/MLPionCollaboration/LCStudies/clusters/pi0_training_data.json', 'r') as f:
            files = json.load(f)

            for file in files:
                file_end = os.path.basename(file)
                root_files.append(file_end)

if test_bool:
    if pipm_bool:
        with open('/home/russbate/MLPionCollaboration/LCStudies/clusters/pion_test.json', 'r') as f:
            files = json.load(f)

            for file in files:
                file_end = os.path.basename(file)
                root_files.append(file_end)
                
        with open('/home/russbate/MLPionCollaboration/LCStudies/clusters/pion_val.json', 'r') as f:
            files = json.load(f)

            for file in files:
                file_end = os.path.basename(file)
                root_files.append(file_end)
                
    if pi0_bool:
        with open('/home/russbate/MLPionCollaboration/LCStudies/clusters/pi0_validation_data.json', 'r') as f:
            files = json.load(f)

            for file in files:
                file_end = os.path.basename(file)
                root_files.append(file_end)


for i in range(len(root_files)):
    root_files[i] = file_loc + atlas_image_subfolder + root_files[i]

if Nfile > len(root_files):
    sys.exit('More files requested than in json. Exiting early.')


#====================
# Load Data Files ===
#====================
MAX_EVENTS = int(2.0e7)
MAX_CELLS = 1500

## GEOMETRY DICTIONARY ##
geo_file = ur.open(geo_loc)
CellGeo_tree = geo_file["CellGeo"]
geo_dict = dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

# cell geometry data
cell_geo_ID = geo_dict['cell_geo_ID']
cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

## MEMORY MAPPED ARRAY ALLOCATION ##
if train_bool: #these are named del string because they will be deleted later :'-)
    if pipm_bool:
        x_del_string = output_loc+'/X_CLO_PIPM_large.npy'
        y_del_string = output_loc+'/Y_CLO_PIPM_large.npy'
    elif pi0_bool:
        x_del_string = output_loc+'/X_CLO_PI0_large.npy'
        y_del_string = output_loc+'/Y_CLO_PI0_large.npy'
        
elif test_bool: #this is so we can run both at once
    if pipm_bool:
        x_del_string = output_loc+'/X_CLO_PIPM_large2.npy'
        y_del_string = output_loc+'/Y_CLO_PIPM_large2.npy'
    elif pi0_bool:
        x_del_string = output_loc+'/X_CLO_PI0_large2.npy'
        y_del_string = output_loc+'/Y_CLO_PI0_large2.npy'

X_large = np.lib.format.open_memmap(x_del_string, mode='w+', dtype=np.float64,
                       shape=(MAX_EVENTS,MAX_CELLS,4), fortran_order=False, version=None)
Y_large = np.lib.format.open_memmap(y_del_string, mode='w+', dtype=np.float64,
                       shape=(MAX_EVENTS,), fortran_order=False, version=None)
Eta_large = np.empty(MAX_EVENTS)


# Pre-Loop Definitions ##
#======================================
k = 1 # tally used to keep track of file number
ClN = 0 # used for keeping track of total number of events
max_nPoints = 0 # used for keeping track of the largest 'point cloud'
t_tot = 0 # total time


## Main File Loop ##
#======================================
for currFile in root_files[:Nfile]:
    
    # Check for file, a few are missing
    if not os.path.isfile(currFile):
        print()
        print('File '+currFile+' not found..')
        print()
        continue
    
    else:
        print()
        print('Working on File: '+str(currFile)+' - '+str(k)+'/'+str(Nfile))
        k += 1
        

    #===============#
    ## LOAD EVENTS ##
    #===============#
    t0 = cput()
    event = ur.open(currFile)
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
#==================================================================================================
if train_bool:
    if pipm_bool:
        x_string = output_loc+'/X_CLO_PIPM_'+str(Nfile)+'_train.npy'
        y_string = output_loc+'/Y_CLO_PIPM_'+str(Nfile)+'_train.npy'
        eta_string = output_loc+'/Eta_CLO_PIPM_'+str(Nfile)+'_train'
    elif pi0_bool:
        x_string = output_loc+'/X_CLO_PI0_'+str(Nfile)+'_train.npy'
        y_string = output_loc+'/Y_CLO_PI0_'+str(Nfile)+'_train.npy'
        eta_string = output_loc+'/Eta_CLO_PI0_'+str(Nfile)+'_train'
        
elif test_bool: #this is so we can run both at once
    if pipm_bool:
        x_string = output_loc+'/X_CLO_PIPM_'+str(Nfile)+'_test.npy'
        y_string = output_loc+'/Y_CLO_PIPM_'+str(Nfile)+'_test.npy'
        eta_string = output_loc+'/Eta_CLO_PIPM_'+str(Nfile)+'_test'
    elif pi0_bool:
        x_string = output_loc+'/X_CLO_PI0_'+str(Nfile)+'_test.npy'
        y_string = output_loc+'/Y_CLO_PI0_'+str(Nfile)+'_test.npy'
        eta_string = output_loc+'/Eta_CLO_PI0_'+str(Nfile)+'_test'
#==================================================================================================

X = np.lib.format.open_memmap(x_string, mode='w+', dtype=np.float64,
                              shape=(ClN, max_nPoints, 4))
np.copyto(dst=X, src=X_large[:ClN,:max_nPoints,:], casting='same_kind',
          where=True)
del X_large
os.system('rm '+x_del_string)

Y = np.lib.format.open_memmap(y_string, mode='w+', dtype=np.float64,
                              shape=(ClN,))
np.copyto(dst=Y, src=Y_large[:ClN,], casting='same_kind', where=True)
del Y_large
os.system('rm '+y_del_string)

np.save(eta_string, Eta_large[:ClN])

t1 = cput()
print('time to copy: {} (m)'.format((t1-t0)/60));print()

        
