
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

def find_max_dim_tuple(events, event_dict):
    nEvents = len(events)
    max_clust = 0
    
    for i in range(nEvents):
        event = events[i,0]
        track_nums = events[i,1]
        clust_nums = events[i,2]
        
        clust_num_total = 0
        # set this to six for now to handle single track events, change later
        track_num_total = 6
        
        # Check if there are clusters, None type object may be associated with it
        if clust_nums is not None:
            # Search through cluster indices
            for clst_idx in clust_nums:
                nInClust = len(event_dict['cluster_cell_ID'][event][clst_idx])
                # add the number in each cluster to the total
                clust_num_total += nInClust

        total_size = clust_num_total + track_num_total
        if total_size > max_clust:
            max_clust = total_size
    
    # 6 for energy, eta, phi, rperp, track flag, sample layer
    return (nEvents, max_clust, 6)

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
track_branches = ['trackEta_EMB1', 'trackPhi_EMB1', 'trackEta_EMB2', 'trackPhi_EMB2', 'trackEta_EMB3', 'trackPhi_EMB3',
                  'trackEta_TileBar0', 'trackPhi_TileBar0', 'trackEta_TileBar1', 'trackPhi_TileBar1',
                  'trackEta_TileBar2', 'trackPhi_TileBar2']

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
# Number of files
Nfile = 502
fileNames = []
file_prefix = 'user.angerami.24559744.OutputStream._000'
for i in range(1,Nfile+1):
    endstring = f'{i:03}'
    fileNames.append(file_prefix + endstring + '.root')

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
events_prefix = '/fast_scratch/atlas_images/v01-45/pipm/'

# Use this to compare with the dimensionality of new events
firstArray = True

## MEMORY MAPPED ARRAY ALLOCATION ##
X_large = np.lib.format.open_memmap('/data/atlas/rbate/X_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,1500,6), fortran_order=False, version=None)
Y_large = np.lib.format.open_memmap('/data/atlas/rbate/Y_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,3), fortran_order=False, version=None)

k = 1 # tally used to keep track of file number
tot_nEvts = 0 # used for keeping track of total number of events
max_nPoints = 0 # used for keeping track of the largest 'point cloud'
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
    event_dict = dict_from_tree(tree=event_tree, branches=ak_event_branches, np_branches=np_event_branches)
    
    ## TRACK DICTIONARY ##
    track_dict = dict_from_tree(tree=event_tree, branches=track_branches)
    
    #===================
    # APPLY CUTS =======
    #===================
    # create ordered list of events to use for index slicing
    nEvents = len(event_dict['eventNumber'])
    all_events = np.arange(0,nEvents,1,dtype=np.int32)

    # SINGLE TRACK CUT
    single_track_mask = event_dict['nTrack'] == np.full(nEvents, 1)
    filtered_event = all_events[single_track_mask]
    
    # CENTRAL TRACKS
    # Warning: we are safe for now with single tracks but not for multiples using this
    trackEta_EMB1 = ak.flatten(track_dict['trackEta_EMB1'][filtered_event]).to_numpy()
    central_track_mask = np.abs(trackEta_EMB1) < 4.9
    filtered_event = filtered_event[central_track_mask]
    
    # TRACKS WITH CLUSTERS
    nCluster = event_dict['nCluster'][filtered_event]
    filtered_event_mask = nCluster != 0
    filtered_event = filtered_event[filtered_event_mask]
    t1 = t.time()
    events_cuts_time = t1 - t0
    
    #============================================#
    ## CREATE INDEX ARRAY FOR TRACKS + CLUSTERS ##
    #============================================#
    event_indices = []
    t0 = t.time()

    for evt in filtered_event:

        # pull cluster number, don't need zero as it's loaded as a np array
        nClust = event_dict["nCluster"][evt]
        cluster_idx = np.arange(nClust)

        # Notes: this will need to handle more complex scenarios in the future for tracks with
        # no clusters

        ## DELTA R ##
        # pull coordinates of tracks and clusters from event
        # we can get away with the zeroth index because we are working with single track events
        trackCoords = np.array([event_dict["trackEta"][evt][0],
                                 event_dict["trackPhi"][evt][0]])
        clusterCoords = np.stack((event_dict["cluster_Eta"][evt].to_numpy(),
                                   event_dict["cluster_Phi"][evt].to_numpy()), axis=1)

        _DeltaR = DeltaR(clusterCoords, trackCoords)
        DeltaR_mask = _DeltaR < .2
        matched_clusters = cluster_idx[DeltaR_mask]

        ## CREATE LIST ##
        # Note: currently do not have track only events. Do this in the future    
        if np.count_nonzero(DeltaR_mask) > 0:
            event_indices.append((evt, 0, matched_clusters))
    
    event_indices = np.array(event_indices, dtype=np.object_)
    t1 = t.time()
    indices_time = t1 - t0
    
    #=========================#
    ## DIMENSIONS OF X ARRAY ##
    #=========================#
    t0 = t.time()
    max_dims = find_max_dim_tuple(event_indices, event_dict)
    evt_tot = max_dims[0]
    tot_nEvts += max_dims[0]
    # keep track of the largest point cloud to use for saving later
    if max_dims[1] > max_nPoints:
        max_nPoints = max_dims[1]
    
    # Create arrays
    Y_new = np.zeros((max_dims[0],3))
    X_new = np.zeros(max_dims)
    t1 = t.time()
    find_create_max_dims_time = t1 - t0    
    
    #===================#
    ## FILL IN ENTRIES ##==============================================================
    #===================#
    t0 = t.time()
    for i in range(max_dims[0]):
        # pull all relevant indices
        evt = event_indices[i,0]
        track_idx = event_indices[i,1]
        # recall this now returns an array
        cluster_nums = event_indices[i,2]

        ##############
        ## CLUSTERS ##
        ##############
        # set up to have no clusters, further this with setting up the same thing for tracks
        target_ENG_CALIB_TOT = -1
        if cluster_nums is not None:

            # find averaged center of clusters
            cluster_Eta = event_dict['cluster_Eta'][evt].to_numpy()
            cluster_Phi = event_dict['cluster_Phi'][evt].to_numpy()
            av_Eta = np.mean(cluster_Eta)
            av_Phi = np.mean(cluster_Phi)

            nClust_current_total = 0
            target_ENG_CALIB_TOT = 0
            for c in cluster_nums:            
                # cluster data
                target_ENG_CALIB_TOT += event_dict['cluster_ENG_CALIB_TOT'][evt][c]
                cluster_cell_ID = event_dict['cluster_cell_ID'][evt][c].to_numpy()
                nInClust = len(cluster_cell_ID)
                cluster_cell_E = event_dict['cluster_cell_E'][evt][c].to_numpy()            
                cell_indices = find_index_1D(cluster_cell_ID, cell_ID_dict)

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
                X_new[i,low:high,1] = cluster_cell_Eta - av_Eta
                X_new[i,low:high,2] = cluster_cell_Phi - av_Phi
                X_new[i,low:high,3] = cluster_cell_rPerp
                X_new[i,low:high,5] = cluster_cell_sampling

                nClust_current_total += nInClust

        #####################
        ## TARGET ENERGIES ##
        #####################
        # this should be flattened or loaded as np array instead of zeroth index in future
        Y_new[i,0] = event_dict['truthPartE'][evt][0]
        Y_new[i,1] = event_dict['truthPartPt'][evt][track_idx]
        Y_new[i,2] = target_ENG_CALIB_TOT

        ############
        ## TRACKS ##
        ############

        trackP = event_dict['trackP'][evt][track_idx]

        track_arr = np.zeros((6,6))
        track_arr[:,5] = track_sample_layer
        # track flag
        track_arr[:,4] = np.ones((6,))
        track_arr[:,3] = layer_rPerp

        # Fill in eta and phi values
        # this is complicated - simplify?
        p, q = 0, 1
        for j in range(12):
            # This gives the key for the track dict
            track_arr[p,q] = track_dict[track_branches[j]][evt][track_idx]
            if j%2 != 0:
                p += 1
                q = 1
            else:
                q = 2

        # search for NULL track flags
        track_eta_null_mask = np.abs(track_arr[:,1]) > 4.9
        track_phi_null_mask = np.abs(track_arr[:,2]) >= np.pi
        track_flag_null = np.logical_or(track_eta_null_mask, track_phi_null_mask)
        
        # Normalize track information!
        track_arr[:,1] = track_arr[:,1] - av_Eta
        track_arr[:,2] = track_arr[:,2] - av_Phi

        # where the flag is set to null, set values of energy and calo layer to zero
        if np.any(track_flag_null):
            # number for which to spread the energy out over
            p_nums = 6 - np.count_nonzero(track_flag_null)
            track_arr[track_flag_null,1:6] = 0
            # get where the track exists (not null)
            track_arr[np.invert(track_flag_null),0] = trackP/p_nums
        # otherwise fill in pt/6 for all
        else:
            track_arr[:,0] = trackP/6      

        # Save track information
        X_new[i,high:high+6,0:6] = track_arr

    #####################################################
    t1 = t.time()
    array_construction_time = t1 - t0
    
    #=======================#
    ## ARRAY CONCATENATION ##
    #=======================#
    t0 = t.time()
    # Write to X
    old_tot = tot_nEvts - max_dims[0]
    X_large[old_tot:tot_nEvts, :max_dims[1], :6] = np.ndarray.copy(X_new)
    # pad the remainder with zeros (just to be sure)
    fill_shape = (tot_nEvts - old_tot, 1500 - max_dims[1], 6)
    X_large[old_tot:tot_nEvts, max_dims[1]:1500, :6] = np.zeros(fill_shape)
    
    # Write to Y
    Y_large[old_tot:tot_nEvts,:] = np.ndarray.copy(Y_new)
        
    t1 = t.time()
    time_to_memmap = t1-t0
    thisfile_t_tot = events_cuts_time+find_create_max_dims_time+indices_time\
          +array_construction_time+time_to_memmap
    t_tot += thisfile_t_tot
    
    print('Array dimension: '+str(max_dims))
    print('Time to create dicts and select events: '+str(events_cuts_time))
    print('Time to find dimensions and make new array: '+str(find_create_max_dims_time))
    print('Time to construct index array: '+str(indices_time))
    print('Time to populate elements: '+str(array_construction_time))
    print('Time to copy to memory map: '+str(time_to_memmap))
    print('Time for this file: '+str(thisfile_t_tot))
    print('Total events: '+str(tot_nEvts))
    print('Current size: '+str((tot_nEvts,max_nPoints,6)))
    print('Total time: '+str(t_tot))
    print()

t0 = t.time()
X = np.lib.format.open_memmap('/data/atlas/rbate/X_STMC_'+str(Nfile)+'_files2.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 6))
np.copyto(dst=X, src=X_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
del X_large
os.system('rm /data/atlas/rbate/X_large.npy')

Y = np.lib.format.open_memmap('/data/atlas/rbate/Y_STMC_'+str(Nfile)+'_files2.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, 3))
np.copyto(dst=Y, src=Y_large[:tot_nEvts,:], casting='same_kind', where=True)
del Y_large
os.system('rm /data/atlas/rbate/Y_large.npy')

t1 = t.time()
print()
print('Time to copy new and delete old: '+str(t1-t0)+' (s)')
print()

# t0 = t.time()
# np.savez('/data/rbate/XY_STMC_allFiles', X, Y)
# t1 = t.time()
# print()
# print('Time to save file: '+str(t1-t0)+' (s)')
# print()
        
    


