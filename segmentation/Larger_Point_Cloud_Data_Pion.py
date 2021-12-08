''' This script is essentially a copy/paste of LargerData_Rho but attempting to make some universal
code that will work for multiple different data files with a config file. More on this later. '''

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

# Import relevant branches
from deep_set_util import track_branches, event_branches, ak_event_branches, np_event_branches, geo_branches


#====================
# File setup ========
#====================
# user.angerami.24559744.OutputStream._000001.root
# Number of files
Nfile = 35
fileNames = []
file_prefix = 'user.mswiatlo.27153452.OutputStream._000'
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
events_prefix = '/data/atlas/data/allCellTruthv1/pipm/'

# Use this to compare with the dimensionality of new events
firstArray = True


#==================================
# MEMORY MAPPED ARRAY ALLOCATION ##
#==================================
X_large = np.lib.format.open_memmap('/data/atlas/rbate/XPPM_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,1700,5), fortran_order=False, version=None)
Y_large = np.lib.format.open_memmap('/data/atlas/rbate/YPPM_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,1700,2), fortran_order=False, version=None)
Y2_large = np.zeros((2500000,2)) # I don't think we need a mem-map here

k = 1 # tally used to keep track of file number
tot_nEvts = 0 # used for keeping track of total number of events
max_nPoints = 0 # Keep track of the largest point cloud overall
t_tot = 0 # total time

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
    track_dict = dsu.dict_from_tree(tree=event_tree, branches=track_branches)
    
    #===================
    # APPLY CUTS =======
    #===================
    ''' This section combines the previous find_max_dim_tuple function seeing as we are already searching
    for the size of the point cloud anyways. Simultaneous cluster index creation and mem map size. '''
    event_indices = []
    # create ordered list of events to use for index slicing
    nEvts_in_file = len(event_dict['eventNumber'])
    
    # Need to refresh this number every file
    biggest_pCloud = 0 # used for keeping track of the largest 'point cloud' for THIS root file
    
    # loop through all clusters
    for i in range(nEvts_in_file):
        
        nClust = event_dict['nCluster'][i]
        truth_E = event_dict['truthPartE'][i]
        deltaR_cut = np.zeros(nClust,dtype=np.bool_)
        this_pCloud = 0
        cluster_E_tot = 0
        
        # find cluster average coords of [Eta, Phi]
        ''' This could be improved with an energy weighted average in the future. '''
        av_Eta = np.mean( event_dict['cluster_Eta'][i].to_numpy() )
        av_Phi = np.mean( event_dict['cluster_Phi'][i].to_numpy() )
        clust_av = np.array([av_Eta, av_Phi])
        
        # If the cluster average is not central then skip the for loop for clusters
        if np.abs(av_Eta) < .7:
            
            for j in range(nClust):

                ## DELTA_R CUTS ##
                current_coords = np.array([event_dict['cluster_Eta'][i][j], event_dict['cluster_Phi'][i][j]])
                deltaR = dsu.DeltaR(current_coords, clust_av)

                if deltaR <= .2:
                    
                    deltaR_cut[j] = True

                    ## ENERGY CUTS ##
                    cluster_E_tot += event_dict['cluster_E'][i][j]

                    ## POINT CLOUD SIZE CUTS ##
                    this_pCloud += len(event_dict['cluster_cell_ID'][i][j])

            ## CHECK CUTS AND ADD TO LIST
            # short circuit logic for cuts
            if cluster_E_tot >= .50*truth_E:
                if this_pCloud > 15:
                    clusters_idx = np.nonzero(deltaR_cut)[0]
                    event_indices.append((i, clusters_idx))

                    # Note this is done in the nested loop because we only care if it
                    # is appended to the event_indices list
                    if this_pCloud > biggest_pCloud:
                        biggest_pCloud = this_pCloud


    nEvts = len(event_indices)
    print('Number of events after filter: '+str(nEvts))
    event_indices = np.array(event_indices, dtype=np.object_)
    t1 = cput()
    event_cuts_time = t1 - t0
    
    
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

        cluster_Eta = event_dict['cluster_Eta'][evt].to_numpy()
        cluster_Phi = event_dict['cluster_Phi'][evt].to_numpy()
        all_cluster_ENG_CALIB_TOT = 0
        av_Eta = np.mean(cluster_Eta)
        av_Phi = np.mean(cluster_Phi)

        nClust_current_total = 0
        for c in cluster_nums:            
            # cluster data
            cluster_cell_ID = event_dict['cluster_cell_ID'][evt][c].to_numpy()
            nInClust = len(cluster_cell_ID)
            cluster_cell_E = event_dict['cluster_cell_E'][evt][c].to_numpy()    
            cell_indices = dsu.find_index_1D(cluster_cell_ID, cell_ID_dict)
            
            # tally energy target
            all_cluster_ENG_CALIB_TOT += event_dict['cluster_ENG_CALIB_TOT'][evt][c]

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
        Y2_new[i,1] = all_cluster_ENG_CALIB_TOT

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
    
    thisfile_t_tot = event_cuts_time+array_construction_time+time_to_memmap
    t_tot += thisfile_t_tot
    
    print('Array dimension: '+str(max_dims))
    print('Time to create dicts and select events: '+str(event_cuts_time))
    print('Time to populate elements: '+str(array_construction_time))
    print('Time to copy to memory map: '+str(time_to_memmap))
    print('Time for this file: '+str(thisfile_t_tot))
    print('Total events: '+str(tot_nEvts))
    print('Current size: '+str((tot_nEvts,max_nPoints,5)))
    print('Total time: '+str(t_tot))
    print()

t0 = cput()
# regression targets
np.save(file='/data/atlas/rbate/PIPM_Y_regr_'+str(Nfile)+'_files.npy', arr=Y2_large[:tot_nEvts],
        allow_pickle=False)

# deep sets
X = np.lib.format.open_memmap('/data/atlas/rbate/PIPM_X_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 5))
np.copyto(dst=X, src=X_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
del X_large
os.system('rm /data/atlas/rbate/XPPM_large.npy')

# segmentation targets
Y = np.lib.format.open_memmap('/data/atlas/rbate/PIPM_Y_segm_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 2))
np.copyto(dst=Y, src=Y_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
del Y_large
os.system('rm /data/atlas/rbate/YPPM_large.npy')

t1 = cput()
print()
print('Time to copy new and delete old: '+str(t1-t0)+' (s)')
print()