'''
ML4P Point Cloud Data Script
Author: Russell Bate
russell.bate@cern.ch
russellbate@phas.ubc.ca

Notes: Version 2 of the STMC data script.
- single tracks
- clusters within DeltaR of 1.2 of track
- energy weighted cluster average for center'''

#====================
# Load Utils ========
#====================
import numpy as np
import uproot as ur
import awkward as ak
import ROOT, os
import time as t

from util import deep_set_util as dsu
from util.deep_set_util import dict_from_tree, DeltaR, find_max_dim_tuple
from util.deep_set_util import find_index_1D

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
# Load Data Files ===
#====================
MAX_EVENTS = int(5e6)
MAX_CELLS = 1600


## MAIN STMC EVENT SELECTION ##
def STMC(rootfilename, geofile, outdir, save=True, savetag=''):

	ret_dict = dict() # this is what we return from processing the file
	timing_dict = dict()
	ret_dict['start_time'] = t.time()
	timing_dict['start_time'] = t.time()
	# pre file defs
	file_nEvts = 0 # events per file we keep
	file_max_nPoints = 0 # largest point cloud in this file
	file_t_tot = 0 # total time for this file
	file_num_zero_tracks = 0 # number of zero tracks for this file

	infostr = '' # for verbose information

	
	# Check for the file
	if not os.path.isfile(rootfilename):
		infostr += print('File {} not found..'.format(
			rootfilename))
		ret_dict['info_str'] = infostr
		ret_dict['nEvts'] = None
		ret_dict['nPoints'] = None
		ret_dict['timing'] = timing_dict
		ret_dict['num_zero_tracks'] = None
		return ret_dict
	
	geo_file = ur.open(geofile)
	CellGeo_tree = geo_file["CellGeo"]
	geo_dict = dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

	cell_geo_ID = geo_dict['cell_geo_ID']
	cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

	## EVENT DICTIONARY ##
	t0 = t.time()
	event = ur.open(rootfilename)
	event_tree = event["EventTree"]
	event_dict = dict_from_tree(tree=event_tree, branches=ak_event_branches, np_branches=np_event_branches)
	
	## TRACK DICTIONARY ##
	track_dict = dict_from_tree(tree=event_tree,
				branches=trk_proj_eta+trk_proj_phi)

	
	#=============================
	# APPLY CUTS FOR EVENT INDEX =
	#=============================
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
	timing_dict['event_cut'] = events_cuts_time


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
	timing_dict['indices'] = t1 - t0


	#=========================#
	## DIMENSIONS OF X ARRAY ##
	#=========================#
	t0 = t.time()
	max_dims = find_max_dim_tuple(event_indices, event_dict)
	file_nEvts = max_dims[0]
	file_max_nPoints = max_dims[1]
	
	# Create arrays
	Y_new = np.zeros((max_dims[0],3))
	X_new = np.zeros(max_dims)
	Eta_new = np.zeros(max_dims[0])
	t1 = t.time()
	max_dims_time = t1 - t0
	timing_dict['max_dims'] = t1 - t0


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

		## Centering tracks in the EM barrel ##
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
			cluster_Eta = event_dict['cluster_Eta'][evt].to_numpy()[cluster_nums]
			cluster_Phi = event_dict['cluster_Phi'][evt].to_numpy()[cluster_nums]
			cluster_E = event_dict['cluster_E'][evt].to_numpy()[cluster_nums]
			cl_E_tot = np.sum(cluster_E)

			nClust_current_total = 0
			target_ENG_CALIB_TOT = 0

			# cant vectorize because of function :-(
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

		else:
			raise ValueError(
				'No clusters found in {} \n event: {}'.format(rootfilename,
					evt))


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
		trk_full = np.empty((len(calo_numbers), 4)) # eta, phi, rPerp, calo num
		
		for j, (eta_key, phi_key) in enumerate(zip(trk_proj_eta, trk_proj_phi)):
			
			cnum = eta_trk_dict[eta_key]
			layer = calo_dict[cnum]
			
			eta = track_dict[eta_key][evt][track_idx]
			phi = track_dict[phi_key][evt][track_idx]

			# no need for else, False already instantiated at array creation
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
					valErrStr = 'Found negative rPerp'
					valErrStr += '\nEvent number: {}'.format(evt)
					valErrStr += '\nEta: {}'.format(eta)
					valErrStr += '\nPhi: {}'.format(phi)
					valErrStr += '\nrPerp: {}'.format(rPerp)
					raise ValueError(valErrStr)
					
				trk_full[j,2] = rPerp
				
		# Fill in track array
		trk_proj_num = np.count_nonzero(trk_bool)
		
		# Sometimes there is no track projection, place in EMB1
		if trk_proj_num == 0:
			trk_proj_num = 1
			trk_arr = np.empty((1, 6))
			file_num_zero_tracks += 1
			trk_arr[:,0] = event_dict['trackP'][evt][track_idx]
			trk_arr[:,1] = event_dict['trackEta'][evt][track_idx] - eta_ctr
			trk_arr[:,2] = event_dict['trackPhi'][evt][track_idx] - phi_ctr
			trk_arr[:,3] = 1532.18 # just place it in EMB1
			trk_arr[:,4] = 1 # track flag
			trk_arr[:,5] = 1 # place layer in EMB1

		# Create points of track spread out over number of projections
		# this is not always six!
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
	timing_dict['array_fill'] = t1 - t0


	#==========================#
	## SAVE INDIVIDUAL ARRAYS ##
	#==========================#
	t0 = t.time()
	if save:
		np.save(outdir+'Eta_STMC_v2_{}'.format(savetag), Eta_new)
		np.save(outdir+'X_STMC_v2_{}'.format(savetag), X_new)
		np.save(outdir+'Y_STMC_v2_{}'.format(savetag), Y_new) 
	t1 = t.time()
	
	save_time = t1 - t0
	timing_dict['save_time'] = t1 - t0

	thisfile_t_tot = events_cuts_time+max_dims_time+indices_time\
		  +array_construction_time+save_time
	   
	
	infostr += '\nArray dimension: '+str(max_dims)
	infostr += '\nNumber of null track projection: '+str(file_num_zero_tracks)
	infostr += '\nTime to create dicts and select events: '+str(events_cuts_time)
	infostr += '\nTime to find dimensions and make new array: '+str(max_dims_time)
	infostr += '\nTime to construct index array: '+str(indices_time)
	infostr += '\nTime to populate elements: '+str(array_construction_time)
	infostr += '\nTime to copy to save numpy files: '+str(save_time)
	infostr += '\nTime for this file: '+str(thisfile_t_tot)
	infostr += '\nTotal events: '+str(file_nEvts)
	infostr += '\nCurrent size: '+str((file_nEvts,max_dims[1],6))
	infostr += '\nTotal time: '+str(thisfile_t_tot)+'\n'

	ret_dict['timing'] = timing_dict
	ret_dict['infostr'] = infostr
	ret_dict['end_time'] = t.time()
	return ret_dict

