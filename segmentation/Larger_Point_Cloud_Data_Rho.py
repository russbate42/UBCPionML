#====================
# Load Utils ========
#====================

import numpy as np
import uproot as ur
import awkward as ak
import time as t
from time import perf_counter as cput
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
from deep_set_util import DeltaR

# Import relevant branches
from deep_set_util import track_branches, event_branches, ak_event_branches, np_event_branches, geo_branches


#====================
# File setup ========
#====================
# user.angerami.24559744.OutputStream._000001.root
# Number of files
Nfile = 42
fileNames = []
file_prefix = 'user.akong.26471083.OutputStream._000'
for i in range(1,Nfile+1):
    endstring = f'{i:03}'
    fileNames.append(file_prefix + endstring + '.root')

#====================
# Load Data Files ===
#====================

## GEOMETRY DICTIONARY ##
geo_file = ur.open('/fast_scratch/atlas_images/v01-45/cell_geo.root')
CellGeo_tree = geo_file["CellGeo"]
geo_dict = dsu.dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

# cell geometry data
cell_geo_ID = geo_dict['cell_geo_ID']
cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

# additional geometry data
layer_rPerp = np.array([1540., 1733., 1930., 2450., 3010., 3630.])
track_sample_layer = np.array([1,2,3,12,13,14])

# for event dictionary
events_prefix = '/data/atlas/akong/singlerho-percell/'

# Use this to compare with the dimensionality of new events
firstArray = True


#==================================
# MEMORY MAPPED ARRAY ALLOCATION ##
#==================================
X_large = np.lib.format.open_memmap('/data/atlas/rbate/XR_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,1700,5), fortran_order=False, version=None)
Y_large = np.lib.format.open_memmap('/data/atlas/rbate/YR_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,1700,2), fortran_order=False, version=None)
Y2_large = np.zeros((2500000,2)) # I don't think we need a mem-map here

k = 1 # tally used to keep track of file number
tot_nEvts = 0 # used for keeping track of total number of events
max_nPoints = 0 # Keep track of the largest point cloud overall
t_tot = 0 # total time

print('starting run...')
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

    t0 = cput()
    ## EVENT DICTIONARY ##
    event = ur.open(events_prefix+currFile)
    event_tree = event["EventTree"]
    event_dict = dsu.dict_from_tree(tree=event_tree, branches=ak_event_branches, np_branches=np_event_branches)
    
    ## TRACK DICTIONARY ##
    # track_dict = dsu.dict_from_tree(tree=event_tree, branches=track_branches)
    t1 = cput()
    dict_time = t1 - t0
    
    
    #=======================================================
    # New Cuts
    #=======================================================
    t0 = cput()
    nEvents = len(event_dict['eventNumber'])
    all_events = np.arange(0,nEvents,1,dtype=np.int32)
    
    # TRACK CUT
    #/ no track cut
    
    # CLUSTER MASK
    nz_cluster_mask = event_dict['nCluster'] != 0
    print('Number of events total: '+str(nEvents))
    print('Number after nonzero_mask: '+str(np.count_nonzero(nz_cluster_mask)))
    filtered_events = all_events[nz_cluster_mask]
    
    ## CUTS WITH A FOR LOOP  ## :'^(
    event_indices = []

    biggest_pCloud = 0
    for evt in filtered_events:
        nPts = 0       
        # pull cluster number, don't need zero as it's loaded as a np array
        nClust = event_dict["nCluster"][evt]
        cluster_idx = np.arange(nClust)
        
        ## FIND MEAN AND DELTA_R OF clusters
        clust_Eta = event_dict['cluster_Eta'][evt].to_numpy()
        clust_Phi = event_dict['cluster_Phi'][evt].to_numpy()
        clust_E = event_dict['cluster_E'][evt].to_numpy()
        tot_E = np.sum(clust_E)
        # Use energy weighting
        av_Eta = np.sum(clust_Eta * clust_E)/tot_E
        av_Phi = np.sum(clust_Phi * clust_E)/tot_E
        clust_av = np.array([av_Eta, av_Phi])
        
        clusterCoords = np.stack((clust_Eta, clust_Phi), axis=1)

        _DeltaR = DeltaR(clusterCoords, clust_av)
        DeltaR_mask = _DeltaR < .2
        matched_clusters = cluster_idx[DeltaR_mask]
        
        ## CHECK ENERGY
        evt_energy = np.sum( event_dict['cluster_E'][evt][matched_clusters].to_numpy() )
        energy_cut = evt_energy > 100
        
        ## CHECK N POINTS
        nPts = len(ak.flatten(event_dict['cluster_cell_ID'][evt][matched_clusters], axis=None))
        nPts_cut = nPts > 15
        
        ## CENTRAL EVENT CUTS
        eta_cut = np.abs(av_Eta) <= .7
        
        ## ALL CUTS
        cuts = np.array([energy_cut, nPts_cut, eta_cut])
        if np.all(cuts):
            event_indices.append((evt, matched_clusters))

            # Note this is done in the nested loop because we only care if it
            # is appended to the event_indices list
            if nPts > biggest_pCloud:
                biggest_pCloud = nPts

    nEvts = len(event_indices)
    print('Number of events after filter: '+str(nEvts))
    event_indices = np.array(event_indices, dtype=np.object_)
    t1 = cput()
    cuts_time = t1 - t0
    
    
    #=======================#
    ## CREATE X, Y ARRAYS  ##
    #=======================#
    # [Energy, Eta, Phi, rPerp, calo_sample_layer]
    max_dims = (nEvts, biggest_pCloud, 5)
    
    if biggest_pCloud > max_nPoints:
        max_nPoints = biggest_pCloud
        
    
    t0 = cput()
    # [EM, nonEM]
    Y_new = np.zeros((nEvts,biggest_pCloud,2))
    Y2_new = np.zeros((nEvts,2))
    X_new = np.zeros((nEvts,biggest_pCloud,5)) 

    #===================#
    ## FILL IN ENTRIES ##==============================================================
    #===================#
    for i in range(len(event_indices)):
        # pull all relevant indices
        evt = event_indices[i,0]
        # recall this now returns an array
        cluster_nums = event_indices[i,1]

        ##############
        ## CLUSTERS ##
        ##############
        # set up to have no clusters, further this with setting up the same thing for tracks
        # find averaged center of clusters

        clust_Eta = event_dict['cluster_Eta'][evt].to_numpy()
        clust_Phi = event_dict['cluster_Phi'][evt].to_numpy()
        clust_E = event_dict['cluster_E'][evt].to_numpy()
        tot_E = np.sum(clust_E)
        # Use energy weighting
        av_Eta = np.sum(clust_Eta * clust_E)/tot_E
        av_Phi = np.sum(clust_Phi * clust_E)/tot_E
        clust_av = np.array([av_Eta, av_Phi])
        
        nClust_current_total = 0
        for c in cluster_nums:            
            # cluster data
            cluster_cell_ID = event_dict['cluster_cell_ID'][evt][c].to_numpy()
            nInClust = len(cluster_cell_ID)
            cluster_cell_E = event_dict['cluster_cell_E'][evt][c].to_numpy()    
            cell_indices = dsu.find_index_1D(cluster_cell_ID, cell_ID_dict)
            
            cluster_cell_Eta = geo_dict['cell_geo_eta'][cell_indices]
            cluster_cell_Phi = geo_dict['cell_geo_phi'][cell_indices]
            cluster_cell_rPerp = geo_dict['cell_geo_rPerp'][cell_indices]
            cluster_cell_sampling = geo_dict['cell_geo_sampling'][cell_indices]

            # input all the data
            # note here we leave the fourth entry zeros (zero for flag!!!)
            low = nClust_current_total
            high = low + nInClust
            X_new[i,low:high,0] = cluster_cell_E
            # Normalize to average cluster centers
            ''' This might be the wrong thing to do because the scaling is nonlinear '''
            X_new[i,low:high,1] = cluster_cell_Eta - av_Eta
            X_new[i,low:high,2] = cluster_cell_Phi - av_Phi
            X_new[i,low:high,3] = cluster_cell_rPerp
            X_new[i,low:high,4] = cluster_cell_sampling

            nClust_current_total += nInClust

            #####################
            ## TARGET ENERGIES ##
            #####################
            # this should be flattened or loaded as np array instead of zeroth index in future
            Y_new[i,low:high,0] = event_dict['cluster_cell_hitsE_EM'][evt][c]
            Y_new[i,low:high,1] = event_dict['cluster_cell_hitsE_nonEM'][evt][c]
            
        Y2_new[i,0] = event_dict['truthPartE'][evt][0]
        all_CECT = ak.flatten(event_dict['cluster_ENG_CALIB_TOT'][evt][cluster_nums],
                              axis=None).to_numpy()
        Y2_new[i,1] = np.sum(all_CECT)

    #####################################################
    t1 = cput()
    array_construction_time = t1 - t0
    
    
    #=======================#
    ## ARRAY CONCATENATION ##
    #=======================#
    t0 = cput()
    ## Write to X ##
    X_large[tot_nEvts:tot_nEvts+nEvts, :max_dims[1], :6] = np.ndarray.copy(X_new)
    # pad the remainder with zeros (just to be sure)
    fill_shape_X = (nEvts, 1700 - max_dims[1], 5)
    X_large[tot_nEvts:tot_nEvts+nEvts, max_dims[1]:1700, :6] = np.zeros(fill_shape_X)
    
    ## Write to Y ##
    Y_large[tot_nEvts:tot_nEvts+nEvts, :max_dims[1], :2] = np.ndarray.copy(Y_new)
    # pad the remainder with zeros (just to be sure)
    fill_shape_Y = (nEvts, 1700 - max_dims[1], 2)
    Y_large[tot_nEvts:tot_nEvts+nEvts, max_dims[1]:1700, :2] = np.zeros(fill_shape_Y)
    
    ## Regression Targets ##
    Y2_large[tot_nEvts:tot_nEvts+nEvts, :] = np.ndarray.copy(Y2_new)
    
    # Book keeping for total number of events
    tot_nEvts += nEvts
        
    t1 = cput()
    time_to_memmap = t1-t0
    
    thisfile_t_tot = dict_time+cuts_time+array_construction_time+time_to_memmap
    t_tot += thisfile_t_tot
    
    print('Array dimension: '+str(max_dims))
    print('Time to load events and create dicts: '+str(dict_time))
    print('Time to make cuts: '+str(cuts_time))
    print('Time to populate elements: '+str(array_construction_time))
    print('Time to copy to memory map: '+str(time_to_memmap))
    print('Time for this file: '+str(thisfile_t_tot))
    print('Total events: '+str(tot_nEvts))
    print('Current size: '+str((tot_nEvts,max_nPoints,5)))
    print('Total time: '+str(t_tot))
    print()

t0 = cput()
# regression targets
np.save(file='/data/atlas/rbate/Rho_Y_regr_'+str(Nfile)+'_files.npy', arr=Y2_large[:tot_nEvts],
        allow_pickle=False)

# deep sets
X = np.lib.format.open_memmap('/data/atlas/rbate/Rho_X_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 5))
np.copyto(dst=X, src=X_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
del X_large
os.system('rm /data/atlas/rbate/XR_large.npy')

# segmentation targets
Y = np.lib.format.open_memmap('/data/atlas/rbate/Rho_Y_segm_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 2))
np.copyto(dst=Y, src=Y_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
del Y_large
os.system('rm /data/atlas/rbate/YR_large.npy')

t1 = cput()
print()
print('Time to copy new and delete old: '+str(t1-t0)+' (s)')
print()