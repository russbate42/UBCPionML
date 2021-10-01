#==========================================================================================
# A collection of useful things that I've been using for creating deep set arrays as well as
# some tools for training the networks
# Author: R. Bate - russellbate@phas.ubc.ca
#===========================================================================================

import numpy as np

track_branches = ['trackEta_EMB1', 'trackPhi_EMB1', 'trackEta_EMB2', 'trackPhi_EMB2', 'trackEta_EMB3', 'trackPhi_EMB3',
                  'trackEta_TileBar0', 'trackPhi_TileBar0', 'trackEta_TileBar1', 'trackPhi_TileBar1',
                  'trackEta_TileBar2', 'trackPhi_TileBar2']

event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", 'cluster_nCells', "nCluster", "eventNumber",
                  "nTrack", "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", 'trackPt', 'trackP',
                  'trackMass', 'trackEta', 'trackPhi', 'truthPartE', 'cluster_ENG_CALIB_TOT', "cluster_E", 'truthPartPt']

ak_event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", "cluster_nCells",
                  "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", "trackPt", "trackP",
                  "trackMass", "trackEta", "trackPhi", "truthPartE", "cluster_ENG_CALIB_TOT", "cluster_E", "truthPartPt",
                    "cluster_cell_hitsE_EM", "cluster_cell_hitsE_nonEM"]

np_event_branches = ["nCluster", "eventNumber", "nTrack", "nTruthPart"]

geo_branches = ["cell_geo_ID", "cell_geo_eta", "cell_geo_phi", "cell_geo_rPerp", "cell_geo_sampling"]

def tvt_num(data, tvt=(75, 10, 15)):
    ''' Function designed to output appropriate numbers for traning validation and testing given
    a variable length input. TVT expressed as ratios and do not need to add to 100. '''
    tot = len(data)
    train, val, test = tvt
    tvt_sum = train + val + test
    
    train_rtrn = round(train*tot/tvt_sum)
    val_rtrn = round(val*tot/tvt_sum)
    test_rtrn = tot - train_rtrn - val_rtrn
    
    return train_rtrn, val_rtrn, test_rtrn

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

def find_max_calo_cells(events, event_dict):
    ''' This function is designed to work as the find_max_dim_tuple except we are negating tracks here '''
    return None
    
def find_max_dim_tuple(events, event_dict):
    ''' This function is designed to return the sizes of a numpy array such that we are efficient
    with zero padding. Please feel free to write this faster, it can be done. Notes: we add six
    to the maximum cluster number such that we have room for track info.
    Inputs:
        _events: filtered list of events to choose from in an Nx3 format for event, track, cluster index 
        _event_tree: the event tree dictionary
    Returns:
        3-tuple consisting of (number of events, maximum cluster_size, 5), 5 because of how we have structured
        the X data format in energyFlow to be Energy, Eta, Phi, rPerp, track flag, sampling layer
        _cluster_ENG_CALIB_TOT, turthPartE '''    
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

def find_index_1D(values, dictionary):
    ''' Use a for loop and a dictionary. values are the IDs to search for. dict must be in format 
    (cell IDs: index) '''
    idx_vec = np.zeros(len(values), dtype=np.int32)
    for i in range(len(values)):
        idx_vec[i] = dictionary[values[i]]
    return idx_vec
