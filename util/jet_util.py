# Utilities for jet clustering, plotting jet kinematics et cetera. These are functions used in our jet-clustering workflow,
# so some many not have to do *explicitly* with jets (e.g. there might be some stuff for plotting topo-cluster predicted energies).
# Note that functions that are purely for convenience will be placed in qol_util.py.
import sys
import ROOT as rt
import uproot as ur
import numpy as np
from numba import jit

# Perform jet clustering. The "energy" parameter determines
# the name of the branch used for topo-cluster energies. If
# it is set to None (default), we use our ML energy regression.
def ClusterJets(ur_trees, jet_name, R, pt_min, eta_max, fj_dir, classification_threshold = 0.5, energy_tree_key=None, energy_branch=None, tree_name = 'JetTree', input_GeV = True):
    
    energy_scaling = 1.
    if(input_GeV): energy_scaling = 1.0e3 # jet info saved in MeV by convention
    
    # import fastjet!
    sys.path.append(fj_dir)
    import fastjet as fj
    
    # Create our jet definition for clustering -- we use anti-kt.
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    
    # branch buffer for our jet tree, where we will save the jets we cluster
    branch_buffer = {
        jet_name + 'Pt':rt.std.vector('float')(),
        jet_name + 'Eta':rt.std.vector('float')(),
        jet_name + 'Phi':rt.std.vector('float')(),
        jet_name + 'E':rt.std.vector('float')()
    }
    
    # loop through the uproot trees (corresponding to some set of input files)
    for dfile, trees in ur_trees.items():
        
        # event info - which clusters belong to a given event
        cluster_min = trees['event'].array('clusterCount')
        cluster_max = cluster_min + trees['event'].array('nCluster') - 1
    
        # Reco cluster info. Excluding the energy, which will be packaged separately below.
        cluster_vec = np.column_stack(tuple(trees['cluster'].arrays(['clusterPt','clusterEta','clusterPhi','clusterE']).values()))
    
        if(energy_branch == None or energy_tree_key == None):
            # Topo-cluster classifications for all of the clusters in this file.
            cluster_classification = trees['score'].array('charged_likelihood_combo')
            # Topo-cluster regressed energies for all clusters in this file (regressions assuming cluster comes from charged/neutral pion)
            cluster_energies = np.column_stack(tuple(trees['score'].arrays(['clusterE_charged','clusterE_neutral']).values()))
        
        else: cluster_energies = trees[energy_tree_key].array(energy_branch)

        # ROOT access to the file -- we are making a new tree to save the jet information.
        f = rt.TFile(dfile, 'UPDATE')
        t = rt.TTree(tree_name, tree_name)
        branches = {}
        for key,val in branch_buffer.items(): branches[key] = t.Branch(key, val)
    
        vec_polar = rt.Math.PtEtaPhiEVector() # for performing polar/Cartesian conversions for fastjet
        
        # loop over events
        nevents = trees['event'].numentries
        for i in range(nevents):

            # Explicit list of cluster indices we're working with -- these are indices in ClusterTree, corresponding to event i.
            cluster_idxs = np.linspace(cluster_min[i], cluster_max[i], cluster_max[i] - cluster_min[i] + 1, dtype=np.dtype('i8'))        
                
            pseudojets = []
            for idx in cluster_idxs:
                
                # Energy assignment for each topo-cluster. By default, we use the cluster classifications and energy regressions
                # to assign energies, but the "energy" parameter to this function can be used to explictly pick a branch to access.
                
                if(energy_branch == None or energy_tree_key == None):
                    # Swap in the regressed energy, start off assuming this cluster corresponds with a charged pion.
                    energy = cluster_energies[idx,0]
                    # Switch to neutral energy regression if dictated by classification.
                    if cluster_classification[idx] < classification_threshold: energy = cluster_energies[idx,1]
                else: energy = cluster_energies[idx]
            
                # Get the ratio of the regressed energy to the original reco energy.
                energy_ratio = energy / cluster_vec[idx,3]
            
                # Rescale the pT according to how we've changed the topo-cluster energy (don't modify eta, phi).
                pt = cluster_vec[idx,0] * energy_ratio
            
                # Create 4-vector representing the topo-cluster.
                vec_polar.SetCoordinates(pt,cluster_vec[idx,1],cluster_vec[idx,2],energy)
            
                # Make a fastjet PseudoJet object from this 4-vector, add it to the PseudoJet list that will be handed off to jet clustering.
                pseudojets.append(fj.PseudoJet(vec_polar.Px(), vec_polar.Py(), vec_polar.Pz(), vec_polar.E())) # fastjet uses Cartesian input
        
            # Perform jet clustering.
            jets = jet_def(pseudojets)
        
            # Apply optional minimum jet pT cut.
            jet_pt = np.array([jet.pt() for jet in jets])
            jet_indices = np.linspace(0,len(jets)-1,len(jets),dtype=np.dtype('i8'))[jet_pt >= pt_min]
            jets = [jets[i] for i in jet_indices]
        
            # Apply optional maximum |eta| cut.
            jet_eta = np.array([jet.eta() for jet in jets])
            jet_indices = np.linspace(0,len(jets)-1,len(jets),dtype=np.dtype('i8'))[np.abs(jet_eta) <= eta_max]
            jets = [jets[i] for i in jet_indices]
        
            # Save jet info to a TTree.
            njets = len(jets)
            for key in branch_buffer.keys(): branch_buffer[key].clear() 
            for j in range(njets):    
                branch_buffer[jet_name + 'Pt'].push_back(jets[j].pt() * energy_scaling)
                branch_buffer[jet_name + 'Eta'].push_back(jets[j].eta())
                branch_buffer[jet_name + 'Phi'].push_back(jets[j].phi())
                branch_buffer[jet_name + 'E'].push_back(jets[j].e() * energy_scaling)
        
            t.Fill()
        t.Write(tree_name, rt.TObject.kOverwrite)
        f.Close()
    return

# Returns pairs & lists of matched and unmatched indices, for a single event. These indices are w.r.t.
# whatever is given as input -- if some jets have been dropped from lists (for not passing cuts)
# this will *not* be known to jet_matching(). (i.e. it will always work 
# internally with a set of sequential indices, with which it reports results.)
def JetMatching(reco_jets, truth_jets, max_distance = 0.3):
    ntruth = len(truth_jets['eta'])
    nreco = len(reco_jets['eta'])
    reco_indices = np.linspace(0, nreco, nreco + 1, dtype = np.dtype('i2'))
    
    #TLorentzVectors for computing deltaR
    vec1 = rt.Math.PtEtaPhiEVector()
    vec2 = rt.Math.PtEtaPhiEVector()

    matched_indices = []
    unmatched_truth = []
    unmatched_reco = []
    
    for i in range(ntruth):
        truth_eta = truth_jets['eta'][i]
        truth_phi = truth_jets['phi'][i]
        vec1.SetCoordinates(0.,truth_eta,truth_phi,0.)
        
        # get distances between this truth jet and all unmatched reco jets
        distances = np.zeros(nreco)
        for j in range(nreco):
            reco_idx = reco_indices[j]
            if(reco_idx < 0):
                distances[j] = -999.
                continue 
            vec2.SetCoordinates(0.,reco_jets['eta'][reco_idx],reco_jets['phi'][reco_idx],0.)
            distances[j] = rt.Math.VectorUtil.DeltaR(vec1,vec2)
            
        # now find the minimum distance, beware of negative values
        # see https://stackoverflow.com/a/37973409
        valid_idx = np.where(distances >= 0.)[0]
        
        if(len(valid_idx) == 0):
            unmatched_truth.append(i)
            continue
        
        match_idx = valid_idx[distances[valid_idx].argmin()]
        matched_indices.append((i, match_idx))
        reco_indices[match_idx] = -1.
    unmatched_reco = reco_indices[reco_indices > -1]
    
    return {'truth_reco':matched_indices, 'unmatched_truth':unmatched_truth, 'unmatched_reco':unmatched_reco}

# Convenience function. This converts from
# indices with respect to a list of selected jets, to indices
# with respect to all indices in the original tree (passed as inputs).
@jit
def IndexConversion(index_matrix, reco_indices, truth_indices):
    index_matrix[:,0] = truth_indices[index_matrix[:,0]]
    index_matrix[:,1] = reco_indices[index_matrix[:,1]]
    return index_matrix
     
def MatchRecoJets(ur_trees, jet_defs, R, eta_max, truth_e_min, tree_name = 'JetMatchTree', input_GeV = True):

    scaling_factor = 1.0e3
    if(not input_GeV): scaling_factor = 1 # assuming MeV if not GeV
        
    reco_jet_defs = [x for x  in jet_defs.keys() if x != 'Truth']
    index_type = ('int',np.dtype('i4')) # allows us to choose how we store matching indices, e.g. int, short
    branch_buffer = {jet_defs[reco_def][1] + 'Match':rt.std.vector(index_type[0])() for reco_def in reco_jet_defs}

    for dfile, tree in ur_trees.items():   
        # tree for saving jet matching info
        f = rt.TFile(dfile, 'UPDATE')
        tree_name = tree_name
        t = rt.TTree(tree_name, tree_name)
        branches = {}
        for key,val in branch_buffer.items():
            branches[key] = t.Branch(key, val)
    
        # Determine which jets pass our global eta cut. In practice, we have have already applied this during jet clustering itself.
        eta = {key:tree[val[0]].array(val[1] + 'Eta') for key, val in jet_defs.items()}
        jet_indices = {key: np.abs(x) <= eta_max for key,x in eta.items()}
    
        # Apply our truth jet energy cut. Recall that jets have things stored in MeV for now, whereas the cut is in GeV.
        truth_energy = tree[jet_defs['Truth'][0]].array(jet_defs['Truth'][1] + 'E')
        jet_indices['Truth'] = jet_indices['Truth'] * (truth_energy >= scaling_factor * truth_e_min)
    
        # We will also need phi info for performing the matching.
        phi = {key:tree[val[0]].array(val[1] + 'Phi') for key, val in jet_defs.items()}

        nevents = ur_trees[dfile]['event'].numentries
        for i in range(nevents):
        
            # For each event, store the jets that have passed our pre-selection cuts.
            selected_jets = {}
        
            for key in jet_indices.keys():
                selected_jets[key] = {'eta':eta[key][i][jet_indices[key][i]],'phi':phi[key][i][jet_indices[key][i]]}
            
            # TODO: Consider eliminating this conditional, see if it really speeds things up at all.
            # Speedup: Skip to next event if *no* truth jets were selected (there will be no matching to do anyway).
            # Make sure to fill our branch buffers with -1 for each reco jet, to explicitly label them as unmatched.
            # (We want the length of these vectors to *always* match the number of reco jets!)
            if(len(selected_jets['Truth']['eta']) == 0): 
                for rdef in reco_jet_defs:
                    buffer_key = jet_defs[rdef][1] + 'Match'
                    branch_buffer[buffer_key].clear()
                    n = len(jet_indices[rdef][i]) # total number of reco jets for this event (including those not passing cuts)
                    for j in range(n): branch_buffer[buffer_key].push_back(int(-1))
                t.Fill()
                continue
            
            # Keeping track of which indices are present and dropped.
            # For each jet definition, this dictionary lists all jet indices w.r.t. the tree, that have passed cuts.
            # E.g. if EM jets at positions 0, 4 and 5 in our branch passed cuts, then jet_tree_indices['EM']=np.array([0,4,5]).
            # More user-friendly than a boolean array of full length, e.g. [True, False, False, False, True, True,...].
            # Note that jet_indices[key] is an Awkward array, so jet_indices[key][i] gives the i'th entry, for the i'th event,
            # whose length might differ from the j'th entry jet_indices[key][j] (a.k.a. a "jagged array").
            jet_tree_indices = {key:np.linspace(0,len(jet_indices[key][i])-1,len(jet_indices[key][i]),dtype=np.dtype('i2'))[jet_indices[key][i]] for key in jet_indices.keys()}        
        
            # Now perform matching for each reco jet definition.
            for rdef in reco_jet_defs:
                buffer_key = jet_defs[rdef][1] + 'Match' # key for accessing TTree branch buffer
                branch_buffer[buffer_key].clear()

                # Info on (reco,truth) pairs, as well as any unmatched reco and truth jets.
                matches, unmatch_true, unmatch_reco = JetMatching(reco_jets=selected_jets[rdef], truth_jets=selected_jets['Truth'], max_distance=R).values()
            
                # If we don't find matches, we can skip the rest of this loop. 
                # Make sure to fill the vectors with a bunch of -1's to indicate unmatched reco jets.
                # (We want the length of these vectors to *always* match the number of reco jets!)
                if(matches == []):
                    n = len(jet_indices[rdef][i])
                    for j in range(n): branch_buffer[buffer_key].push_back(int(-1))
                    continue
                
                # Convert from indices amongst selected jets, to indices amongst all jets in tree.
                matches = IndexConversion(np.array(matches,dtype=index_type[1]), jet_tree_indices[rdef], jet_tree_indices['Truth'])           
     
                # Now prepare to fill the branch buffer. Each position correspond to a reco jet in our tree.
                # We will fill each position with the index of the truth jet (in the tree) to which it's matched.
                # If a particular reco jet is not matched, we will label it with a -1.
        
                # We prep results in a numpy array, we'll push_back the whole thing when ready.
                match_list = np.full(len(jet_indices[rdef][i]),-1,dtype=index_type[1])
                for row in matches: match_list[row[1]] = row[0]            
            
                # Now prepare to fill the branch buffer. Each position correspond to a reco jet in our tree.
                # We will fill each position with the index of the truth jet (in the tree) to which it's matched.
                # If a particular reco jet is not matched, we will label it with a -1.
            
                # Note that the first column of matches gives truth jet indices, the second column gives reco indices.
                # We will first fill the vector with -1's, then make replacements as necessary (we can treat it like np vector).
                for entry in match_list: branch_buffer[buffer_key].push_back(int(entry)) # TODO: Why do we need to cast to Python int?
            
            # The branch buffers have been filled, time to write this event to the tree.
            t.Fill()
    
        # Done writing events. Write tree to file.
        t.Write(tree_name,rt.TObject.kOverwrite)
        f.Close()
    return