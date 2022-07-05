'''
ML4P Point Cloud Data Script
Author: Russell Bate
russell.bate@cern.ch
russellbate@phas.ubc.ca

Notes: Version 2 of the STMC data script.
- single tracks
- clusters within DeltaR of 1.2 of track
- energy weighted cluster average '''
#====================
# Load Utils ========
#====================

import numpy as np
import uproot as ur
import awkward as ak
import time as t
import os, sys, argparse, json
from copy import deepcopy

sys.path.append('/home/russbate/MLPionCollaboration/LCStudies')
from util import deep_set_util as dsu
from util.deep_set_util import dict_from_tree, DeltaR, find_max_dim_tuple
from util.deep_set_util import find_index_1D

Nfile=1

print()
print('='*43)
print('== Single Track Multiple Cluster Script ==')
print('='*43)
print()
print('== Generating data for PubNote 2022 ==')
print()
print("Awkward version: "+str(ak.__version__))
print("Uproot version: "+str(ur.__version__))
print("Numpy version: {}".format(np.__version__))


## Read in Parameters
#=============================================================================
parser = argparse.ArgumentParser(description='Inputs for STMC track script.')

parser.add_argument('--nFile', action="store", dest='nf', default=1,
                   type=int)
parser.add_argument('--file_loc', action="store", dest='fl', default=\
                    '/fast_scratch_1/atlas_images/v01-45/pipm/',
                   type=str)
parser.add_argument('--output_loc', action="store", dest='ol', default=\
                    '/fast_scratch_1/atlas/staged/STMC',
                   type=str)
parser.add_argument('--geo_loc', action="store", dest='gl', default=\
                    '/fast_scratch_1/atlas_images/v01-45/cell_geo.root',
                    type=str)
parser.add_argument('--train', action='store_true',
                    dest='train')
parser.add_argument('--test', action='store_true',
                    dest='test')
parser.add_argument('--start', action='store', default=0, type=int)
args = parser.parse_args()

Nfile = args.nf
file_loc = args.fl
output_loc = args.ol
geo_loc = args.gl
train_bool = args.train
test_bool = args.test
file_start = args.start

if not train_bool and not test_bool:
    sys.exit('Need to specify --train or --test. Exiting.')
    
elif train_bool and test_bool:
    sys.exit('Cannot specify --train and --test. Exiting.')

print('Working on {} files'.format(Nfile))
if train_bool:
    print('Creating training data.')
if test_bool:
    print('Creating test data.')
    

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

if test_bool:
    with open('/home/russbate/MLPionCollaboration/LCStudies/clusters/pion_test.json', 'r') as f:
        files = json.load(f)

        for file in files:
            file_end = os.path.basename(file)
            root_files.append(file_end)
    
for i in range(len(root_files)):
    root_files[i] = file_loc + root_files[i]

if Nfile > len(root_files):
    sys.exit('More files requested than in json. Exiting early.')

#====================
# Load Data Files ===
#====================
MAX_EVENTS = int(5e6)
MAX_CELLS = 1600

## GEOMETRY DICTIONARY ##
geo_file = ur.open(geo_loc)
CellGeo_tree = geo_file["CellGeo"]
geo_dict = dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

# cell geometry data
cell_geo_ID = geo_dict['cell_geo_ID']
cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

## MEMORY MAPPED ARRAY ALLOCATION ##
# if train_bool:
#     X_large = np.lib.format.open_memmap(output_loc+'/X_large.npy', mode='w+', dtype=np.float64,
#                            shape=(MAX_EVENTS,MAX_CELLS,6), fortran_order=False, version=None)
#     Y_large = np.lib.format.open_memmap(output_loc+'/Y_large.npy', mode='w+', dtype=np.float64,
#                            shape=(MAX_EVENTS,3), fortran_order=False, version=None)

# if test_bool: #this is so we can run both at once
#     X_large = np.lib.format.open_memmap(output_loc+'/X_large2.npy', mode='w+', dtype=np.float64,
#                            shape=(MAX_EVENTS,MAX_CELLS,6), fortran_order=False, version=None)
#     Y_large = np.lib.format.open_memmap(output_loc+'/Y_large2.npy', mode='w+', dtype=np.float64,
#                            shape=(MAX_EVENTS,3), fortran_order=False, version=None)

# Eta_large = np.empty(MAX_EVENTS)


# Pre-Loop Definitions ##
#======================================
k = 1 # tally used to keep track of file number
tot_nEvts = 0 # used for keeping track of total number of events
max_nPoints = 0 # used for keeping track of the largest 'point cloud'
t_tot = 0 # total time
num_zero_tracks = 0


## Main File Loop ##
#======================================
for currFile in root_files[file_start:Nfile]:
    
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

    t0 = t.time()
    ## EVENT DICTIONARY ##
    event = ur.open(currFile)
    event_tree = event["EventTree"]
    event_dict = dict_from_tree(tree=event_tree, branches=ak_event_branches, np_branches=np_event_branches)
    
    ## TRACK DICTIONARY ##
    track_dict = dict_from_tree(tree=event_tree,
                branches=deepcopy(trk_proj_eta+trk_proj_phi))
    
    #===================
    # APPLY CUTS =======
    #===================
    # create ordered list of events to use for index slicing
    nEvents = len(event_dict['eventNumber'])
    all_events = np.arange(0,nEvents,1,dtype=np.int32)

    # SINGLE TRACK CUT
    single_track_mask = event_dict['nTrack'] == np.full(nEvents, 1)
    single_track_filter = all_events[single_track_mask]
    
    # TRACKS WITH CLUSTERS
    nCluster = event_dict['nCluster'][single_track_filter]
    nz_clust_mask = nCluster != 0
    filtered_event = single_track_filter[nz_clust_mask]
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
        '''We can get away with the zeroth index because we are working with
        single track events. Technically we cut on trackEta and then select
        from track EMx2 which is inconsistend. Could fix in the future.'''
        trackCoords = np.array([event_dict["trackEta"][evt][0],
                                 event_dict["trackPhi"][evt][0]])
        clusterCoords = np.stack((event_dict["cluster_Eta"][evt].to_numpy(),
                                   event_dict["cluster_Phi"][evt].to_numpy()), axis=1)

        _DeltaR = DeltaR(clusterCoords, trackCoords)
        DeltaR_mask = _DeltaR < 1.2
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
    Y_new = np.zeros((evt_tot,3))
    X_new = np.zeros(max_dims)
    Eta_new = np.zeros(max_dims[0])
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

        ## Centering ##
        trk_bool_em = np.zeros(2, dtype=bool)
        trk_full_em = np.empty((2,2))
    
        for l, (eta_key, phi_key) in enumerate(zip(trk_em_eta, trk_em_phi)):

            eta_em = track_dict[eta_key][evt][track_idx]
            phi_em = track_dict[phi_key][evt][track_idx]

            if np.abs(eta_em) < 2.5 and np.abs(phi_em) <= np.pi:
                trk_bool_em[l] = True
                trk_full_em[l,0] = eta_em
                trk_full_em[l,1] = phi_em
                
        nProj_em = np.count_nonzero(trk_bool_em)
        if nProj_em == 1:
            eta_ctr = trk_full_em[trk_bool_em, 0]
            phi_ctr = trk_full_em[trk_bool_em, 1]
            
        elif nProj_em == 2:
            trk_av_em = np.mean(trk_full_em, axis=1)
            eta_ctr = trk_av_em[0]
            phi_ctr = trk_av_em[1]
            
        elif nProj_em == 0:
            eta_ctr = event_dict['trackEta'][evt][track_idx]
            phi_ctr = event_dict['trackPhi'][evt][track_idx]      
        
        ##############
        ## CLUSTERS ##
        ##############
        # set up to have no clusters, further this with setting up the same thing for tracks
        target_ENG_CALIB_TOT = -1
        if cluster_nums is not None:

            # find averaged center of clusters
            cluster_Eta = event_dict['cluster_Eta'][evt].to_numpy()
            cluster_Phi = event_dict['cluster_Phi'][evt].to_numpy()
            cluster_E = event_dict['cluster_E'][evt].to_numpy()
            cl_E_tot = np.sum(cluster_E)

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
                X_new[i,low:high,1] = cluster_cell_Eta - eta_ctr
                X_new[i,low:high,2] = cluster_cell_Phi - phi_ctr
                X_new[i,low:high,3] = cluster_cell_rPerp
                X_new[i,low:high,5] = cluster_cell_sampling

                nClust_current_total += nInClust

        #####################
        ## TARGET ENERGIES ##
        #####################
        # this should be flattened or loaded as np array instead of zeroth index in future
        truthParticleE = event_dict['truthPartE'][evt][0]
        if truthParticleE <= 0:
            raise ValueError('Truth particle energy found to be zero post cuts!')
        else:
            Y_new[i,0] = truthParticleE
        Y_new[i,1] = event_dict['truthPartPt'][evt][track_idx]
        Y_new[i,2] = target_ENG_CALIB_TOT
        
        #########
        ## ETA ##
        #########
        # again only get away with this because we have a single track
        Eta_new[i] = event_dict["trackEta"][evt][track_idx]

        ############
        ## TRACKS ##
        ############
        
        trk_bool = np.zeros(len(calo_numbers), dtype=bool)
        trk_full = np.empty((len(calo_numbers), 4))
        
        for j, (eta_key, phi_key) in enumerate(zip(trk_proj_eta, trk_proj_phi)):
            
            cnum = eta_trk_dict[eta_key]
            layer = calo_dict[cnum]
            
            eta = track_dict[eta_key][evt][track_idx]
            phi = track_dict[phi_key][evt][track_idx]
            
            if np.abs(eta) < 2.5 and np.abs(phi) <= np.pi:
                trk_bool[j] = True
                trk_full[j,0] = eta
                trk_full[j,1] = phi
                trk_full[j,3] = cnum
                
                if cnum in fixed_r_numbers:
                    rPerp = r_calo_dict[cnum]
                    
                elif cnum in fixed_z_numbers:
                    z = z_calo_dict[cnum]
                    aeta = np.abs(eta)
                    rPerp = z*2*np.exp(aeta)/(np.exp(2*aeta) - 1)
                    
                else:
                    raise ValueError('Calo sample num not found in dicts..')
                
                if rPerp < 0:
                    print()
                    print('Found negative rPerp'); print()
                    print('Event number: {}'.format(evt))
                    print('Eta: {}'.format(eta))
                    print('Phi: {}'.format(phi))
                    print('rPerp: {}'.format(rPerp))
                    raise ValueError('Found negative rPerp')
                    
                trk_full[j,2] = rPerp
                
        # Fill in track array
        trk_proj_num = np.count_nonzero(trk_bool)
        
        if trk_proj_num == 0:
            trk_proj_num = 1
            trk_arr = np.empty((1, 6))
            num_zero_tracks += 1
            trk_arr[:,0] = event_dict['trackP'][evt][track_idx]
            trk_arr[:,1] = event_dict['trackEta'][evt][track_idx] - eta_ctr
            trk_arr[:,2] = event_dict['trackPhi'][evt][track_idx] - phi_ctr
            trk_arr[:,3] = 1532.18 # just place it in EMB1
            trk_arr[:,4] = 1 # track flag
            trk_arr[:,5] = 1 # place layer in EMB1
        else:
            trk_arr = np.empty((trk_proj_num, 6))
            trackP = event_dict['trackP'][evt][track_idx]
            trk_arr[:,1:4] = np.ndarray.copy(trk_full[trk_bool,:3])
            trk_arr[:,4] = np.ones(trk_proj_num)
            trk_arr[:,5] = np.ndarray.copy(trk_full[trk_bool,3])
            trk_arr[:,0] = trackP/trk_proj_num

            trk_arr[:,1] = trk_arr[:,1] - eta_ctr
            trk_arr[:,2] = trk_arr[:,2] - phi_ctr

        X_new[i,high:high+trk_proj_num,:] = np.ndarray.copy(trk_arr)
    
    #=========================================================================#
    t1 = t.time()
    array_construction_time = t1 - t0

    #==========================#
    ## SAVE INDIVIDUAL ARRAYS ##
    #==========================#
    t0 = t.time()
    if train_bool:
        np.save(output_loc+'/Eta_STMC_v2_train_'+str(k-2), Eta_new)
        np.save(output_loc+'/X_STMC_v2_train_'+str(k-2), X_new)
        np.save(output_loc+'/Y_STMC_v2_train_'+str(k-2), Y_new) 
    elif test_bool:
        np.save(output_loc+'/Eta_STMC_v2_test_'+str(k-2), Eta_new)
        np.save(output_loc+'/X_STMC_v2_test_'+str(k-2), X_new)
        np.save(output_loc+'/Y_STMC_v2_test_'+str(k-2), Y_new)
    t1 = t.time()
    
    time_to_save = t1-t0
    thisfile_t_tot = events_cuts_time+find_create_max_dims_time+indices_time\
          +array_construction_time+time_to_save
    t_tot += thisfile_t_tot
       
    
    print('Array dimension: '+str(max_dims))
    print('Number of null track projection: '+str(num_zero_tracks))
    print('Time to create dicts and select events: '+str(events_cuts_time))
    print('Time to find dimensions and make new array: '+str(find_create_max_dims_time))
    print('Time to construct index array: '+str(indices_time))
    print('Time to populate elements: '+str(array_construction_time))
    print('Time to copy to save numpy files: '+str(time_to_save))
    print('Time for this file: '+str(thisfile_t_tot))
    print('Total events: '+str(tot_nEvts))
    print('Current size: '+str((tot_nEvts,max_nPoints,6)))
    print('Total time: '+str(t_tot))
    print()


    '''
    #=======================#
    ## ARRAY CONCATENATION ##
    #=======================#
    t0 = t.time()
    # Write to X
    old_tot = tot_nEvts - max_dims[0]
    X_large[old_tot:tot_nEvts, :max_dims[1], :6] = np.ndarray.copy(X_new)
    # pad the remainder with zeros (just to be sure)
    fill_shape = (tot_nEvts - old_tot, MAX_CELLS - max_dims[1], 6)
    X_large[old_tot:tot_nEvts, max_dims[1]:MAX_CELLS, :6] = np.zeros(fill_shape)
    
    # Write to Y
    Y_large[old_tot:tot_nEvts,:] = np.ndarray.copy(Y_new)
    
    # Eta
    Eta_large[old_tot:tot_nEvts] = np.ndarray.copy(Eta_new)
        
    t1 = t.time()
    time_to_memmap = t1-t0
    
t0 = t.time()
if train_bool:
    X = np.lib.format.open_memmap(output_loc+'/X_STMC_v2_'+str(Nfile)+'_train.npy',
                                 mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 6))
    np.copyto(dst=X, src=X_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
    del X_large
    os.system('rm '+output_loc+'/X_large.npy')

    Y = np.lib.format.open_memmap(output_loc+'/Y_STMC_v2_'+str(Nfile)+'_train.npy',
                                 mode='w+', dtype=np.float64, shape=(tot_nEvts, 3))
    np.copyto(dst=Y, src=Y_large[:tot_nEvts,:], casting='same_kind', where=True)
    del Y_large
    os.system('rm '+output_loc+'/Y_large.npy')

    np.save(output_loc+'/Eta_STMC_v2_'+str(Nfile)+'_train', Eta_large[:tot_nEvts])

if test_bool:
    X = np.lib.format.open_memmap(output_loc+'/X_STMC_v2_'+str(Nfile)+'_test.npy',
                                 mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 6))
    np.copyto(dst=X, src=X_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
    del X_large
    os.system('rm '+output_loc+'/X_large2.npy')

    Y = np.lib.format.open_memmap(output_loc+'/Y_STMC_v2_'+str(Nfile)+'_test.npy',
                                 mode='w+', dtype=np.float64, shape=(tot_nEvts, 3))
    np.copyto(dst=Y, src=Y_large[:tot_nEvts,:], casting='same_kind', where=True)
    del Y_large
    os.system('rm '+output_loc+'/Y_large2.npy')

    np.save(output_loc+'/Eta_STMC_v2_'+str(Nfile)+'_test', Eta_large[:tot_nEvts])

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
'''        
    


