
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

# MEMORY MAPPED ARRAY ALLOCATION ##
XR_large = np.lib.format.open_memmap('/data/atlas/rbate/XR_large.npy', mode='w+', dtype=np.float64,
                       shape=(200000,1700,4), fortran_order=False, version=None)
YR_large = np.lib.format.open_memmap('/data/atlas/rbate/YR_large.npy', mode='w+', dtype=np.float64,
                       shape=(200000,1700,2), fortran_order=False, version=None)

k = 1 # tally used to keep track of file number
tot_nEvts = 0 # used for keeping track of total number of events
max_nPoints = 0 # Keep track of the largest point cloud overall
biggest_pCloud = 0 # used for keeping track of the largest 'point cloud' per root file
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

    t0 = t.time()
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
    t0 = t.time()
    # create ordered list of events to use for index slicing
    nEvts_in_file = len(event_dict['eventNumber'])
    
    # loop through all clusters
    for i in range(nEvts_in_file):
        
        nClust = event_dict['nCluster'][i]
        cluster_eta_cut = np.zeros(nClust,dtype=np.bool_)
        cluster_ENG_cut = np.zeros(nClust,dtype=np.bool_)
        min_points_cut = np.zeros(nClust,dtype=np.bool_)
        this_pCloud = 0
        
        for j in range(nClust):
            
            cluster_ENG = event_dict['cluster_E'][i][j]
            if cluster_ENG > 50:
                cluster_ENG_cut[j] = True
            
            if event_dict['cluster_Eta'][i][j] < .7:
                cluster_eta_cut[j] = True
                this_pCloud += len(event_dict['cluster_cell_ID'][i][j])
        
        if this_pCloud > 10:
            min_points_cut.fill(True)
        
        ## CHECK CUTS AND ADD TO LIST
        # note numpy needs 0 index for where because it returns a tuple
        clusters_post_cut = np.logical_and(cluster_eta_cut, cluster_ENG_cut, min_points_cut)
        if np.any(clusters_post_cut):
            clusters_idx = np.where(clusters_post_cut)[0]
            event_indices.append((i, clusters_idx))
        
        if this_pCloud > biggest_pCloud:
            biggest_pCloud = this_pCloud

    nEvts = len(event_indices)
    print('Number of events after filter: '+str(nEvts))
    event_indices = np.array(event_indices, dtype=np.object_)
    t1 = t.time()
    event_cuts_time = t1 - t0
    
    
    #=======================#
    ## CREATE X, Y ARRAYS  ##
    #=======================#
    max_dims = (nEvts, biggest_pCloud, 4)
    tot_nEvts += nEvts
    if biggest_pCloud > max_nPoints:
        max_nPoints = biggest_pCloud
        
    
    t0 = t.time()
    Y_new = np.zeros((nEvts,biggest_pCloud,2))
    X_new = np.zeros((nEvts,biggest_pCloud,4)) 

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
        av_Eta = np.mean(cluster_Eta)
        av_Phi = np.mean(cluster_Phi)

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

            # input all the data
            # note here we leave the fourth entry zeros (zero for flag!!!)
            low = nClust_current_total
            high = low + nInClust
            X_new[i,low:high,0] = cluster_cell_E
            # Normalize to average cluster centers
            X_new[i,low:high,1] = cluster_cell_Eta - av_Eta
            X_new[i,low:high,2] = cluster_cell_Phi - av_Phi
            X_new[i,low:high,3] = cluster_cell_rPerp

            nClust_current_total += nInClust

            #####################
            ## TARGET ENERGIES ##
            #####################
            # this should be flattened or loaded as np array instead of zeroth index in future
            Y_new[i,low:high,0] = event_dict['cluster_cell_hitsE_EM'][evt][c]
            Y_new[i,low:high,1] = event_dict['cluster_cell_hitsE_nonEM'][evt][c]

    #####################################################
    t1 = t.time()
    array_construction_time = t1 - t0
    
    
    #=======================#
    ## ARRAY CONCATENATION ##
    #=======================#
    t0 = t.time()
    # Write to X
    old_tot = tot_nEvts - nEvts
    XR_large[old_tot:tot_nEvts, :max_dims[1], :4] = np.ndarray.copy(X_new)
    # pad the remainder with zeros (just to be sure)
    fill_shape_X = (nEvts, 1700 - max_dims[1], 4)
    XR_large[old_tot:tot_nEvts, max_dims[1]:1700, :4] = np.zeros(fill_shape_X)
    
    # Write to Y
    YR_large[old_tot:tot_nEvts, :max_dims[1], :2] = np.ndarray.copy(Y_new)
    # pad the remainder with zeros (just to be sure)
    fill_shape_Y = (nEvts, 1700 - max_dims[1], 2)
    YR_large[old_tot:tot_nEvts, max_dims[1]:1700, :2] = np.zeros(fill_shape_Y)

        
    t1 = t.time()
    time_to_memmap = t1-t0
    
    thisfile_t_tot = event_cuts_time+array_construction_time+time_to_memmap
    t_tot += thisfile_t_tot
    
    print('Array dimension: '+str(max_dims))
    print('Time to create dicts and select events: '+str(event_cuts_time))
    print('Time to populate elements: '+str(array_construction_time))
    print('Time to copy to memory map: '+str(time_to_memmap))
    print('Time for this file: '+str(thisfile_t_tot))
    print('Total events: '+str(tot_nEvts))
    print('Current size: '+str((tot_nEvts,max_nPoints,4)))
    print('Total time: '+str(t_tot))
    print()

t0 = t.time()
X = np.lib.format.open_memmap('/data/atlas/rbate/Rho_X_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 4))
np.copyto(dst=X, src=XR_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
del XR_large
os.system('rm /data/atlas/rbate/XR_large.npy')

Y = np.lib.format.open_memmap('/data/atlas/rbate/Rho_Y_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 2))
np.copyto(dst=Y, src=YR_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
del YR_large
os.system('rm /data/atlas/rbate/YR_large.npy')

t1 = t.time()
print()
print('Time to copy new and delete old: '+str(t1-t0)+' (s)')
print()       
        