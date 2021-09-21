#==========================================================================================
# A collection of useful things that I've been using for creating deep set arrays as well as
# some tools for training the networks
# Author: R. Bate - russellbate@phas.ubc.ca
#===========================================================================================

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
