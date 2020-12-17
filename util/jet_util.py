# Utilities for jet clustering, et cetera.
import sys
import ROOT as rt
import uproot as ur
import numpy as np


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

# Returns pairs & lists of matched and unmatched indices. These indices are w.r.t.
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