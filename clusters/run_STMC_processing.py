#!/bin/bash
# -*- coding: utf-8 -*-
'''
Simple script to run data processing in parallel with multiple cores or threads
author: Russell Bate
email: russell.bate@cern.ch, russellbate@phas.ubc.ca

Notes:  
'''

import numpy as np
import uproot as ur
import awkward as ak
import time as t
import ROOT
import matplotlib.pyplot as plt
import argparse, sys, os, subprocess, pickle, warnings, traceback, pathlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

#====================
# Load Utils ========
#====================
sys.path.append('/home/russbate/MLPionCollaboration/LCStudies')
from util import deep_set_util as dsu
from util.deep_set_util import dict_from_tree, DeltaR, find_max_dim_tuple
from util.deep_set_util import find_index_1D
import STMC_processing
from STMC_processing import STMC

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
#=============================================================================#


## PRETTY OUTPUT ##
print('\n==============================================')
print('==  Charged Pion Data Processing ...........==')
print('==  Single Track Multiple Cluster Selection ==')
print('==============================================\n')
print('Numpy version: {}'.format(np.__version__))
print('ROOT version: {}'.format(ROOT.__version__))
print('Uproot version: {}'.format(ur.__version__))
print('Awkward version: {}\n'.format(ak.__version__))

## ARGPARSING ##
parser = argparse.ArgumentParser(description='Main script to create numpy\
	.npy files from root files for STMC Pion data for ML4P.')
# positional argument
parser.add_argument('file_directory',
	action='store',
	type=str, default=None,
	help='Directory where the root files live.')
parser.add_argument('--outdir', dest='outdir',
	action='store', type=str, default=None,
	help='Output directory for the *.npy files.')
parser.add_argument('--geo_loc', action="store", dest='gl', default=\
					'/fast_scratch_1/atlas_images/v01-45/cell_geo.root',
					type=str)
parser.add_argument('-f', dest='force', action='store_true',
	help='Will force the creation of input and output files.',
	default=False)
parser.add_argument('--nfiles', dest='nfiles',
	action='store', type=str, default=None,
	help='How many files to process. If left to default, all files will '\
		+'be processed.')
parser.add_argument('--start', dest='start', action='store', default=0,
	type=int,
	help='Which file number to start from in the list.')
parser.add_argument('--debug', dest='debug', action='store_true',
	help='Run this script with debugging enabled.')
parser.add_argument('--verbose', dest='verbose', action='store_true',
	help='Print out as much information into the terminal as possible.',
	default=False)
parser.add_argument('--test', dest='test', default=False,
	action='store_true',
	help='Process a single file.')
parser.add_argument('--threading', dest='threading', default=False,
	action='store_true',
	help='Uses multi-threading.')
parser.add_argument('--threads', dest='threads', default=1,
	type=int, action='store',
	help='Number of threads to use per core.')
parser.add_argument('--processes', dest='processes', default=False,
	action='store_true',
	help='Uses multiple CPUs.')
parser.add_argument('--workers', dest='workers', default=1,
	type=int, action='store',
	help='Number of CPUs to use.')
args = parser.parse_intermixed_args()

InDir = args.file_directory
OutDir = args.outdir
Debug = args.debug
GeoLoc = args.gl
Verbose = args.verbose
NFiles = args.nfiles
Threading = args.threading
Threads = args.threads
Processes = args.processes
Workers = args.workers
Test = args.test
Start = args.start

if InDir[-1] != '/':
	InDir = InDir + '/'
if OutDir is None:
	OutDir = InDir
if OutDir[-1] != '/':
	OutDir = OutDir + '/'
	
if os.path.isdir(InDir):
	InDir_last = InDir.split('/')[-1]
else:
	raise ValueError('\nDirectory {} not found!\n'.format(InDir))

if not os.path.isdir(OutDir):
	''' error handling for os.mkdir '''
	print('{} Is not a directory. Would you like this to be created?')
	create = False
	while create is False:
		answer = input('    Enter \'yes/y\' or \'no/n\'')
		if answer == 'yes' or answer == 'y':
			create == True
		elif answer == 'no' or ansewr == 'n':
			sys.exit('\nExiting..\n')
		else: 
			pass
	# make the directory    
	os.system('mkdir {}'.format(OutDir))


#===========================================#
## LOOK FOR ROOT FILES IN OUTPUT DIRECTORY ##
#===========================================#
cmd='ls {}'.format(InDir)
if Verbose or Debug:
	print('cmd = {}'.format(cmd))

ls_dir_path = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
grep_root = subprocess.run('grep .root'.split(),
	input=ls_dir_path.stdout, stdout=subprocess.PIPE)

# get a list of all the containers
raw_files = grep_root.stdout.decode('utf-8').split()

NAvailableFiles = len(raw_files)

# Set NFiles
if not NFiles is None:
	int_str_dict = dict()
	for x in range(1000):
		int_str_dict[str(x)] = x

	try:
		NFiles = int_str_dict[NFiles]
	except KeyError as kerr:
		print(traceback.format_exc())
		raise KeyError('\nRequested files exceeds hard coded limit of 1000. '\
			+'Change this or reduce the file number using hadd.\n')

	# Compare with NAvailableFiles
	if NFiles > NAvailableFiles:
		raise ValueError('Not enough files in file directory for --nfile. '\
			+'Requested {} but have {}.'.format(NFiles, NAvailableFiles))
	
	elif NAvailableFiles == 0:
		raise FileNotFoundError('No root files exist in this directory!')

	elif (Start + NFiles) > NAvailableFiles:
		raise ValueError('Too many files requested for starting point of '\
			+'{}'.format(NFiles))

	elif Start > NAvailableFiles:
		raise ValueError('Starting point of {} '.format(Start)\
			+'too high for {} available files.'.format(NAvailableFiles))
	
else:
	NFiles == NAvailableFiles

if not NFiles is None:
	if Processes > NFiles:
		warnings.warn('Requested Processes greater than number of files'\
					  +'\nRequested {}'.format(Processes)
					  +'\nReducing Processes to {}'.format(NFiles))
		Processes = NFiles
	
file_list_full = []
for raw_file in raw_files[:NFiles]:
	file_list_full.append('{}{}'.format(
		InDir, raw_file.rstrip()))

if (Verbose or Debug) and not Test:
	print('\nPrinting files in file list.')
	print('--'*20+'\n')
	for file in file_list_full:
		print(file)
	print('\n -- Number of files: {}\n'.format(len(file_list_full)))


#-----------------------------------#
## Run Distributed Data Processing ##
#-----------------------------------#

#========#
## TEST ##
#========#
if Test:
	tTest0 = t.time()
	# maybe just run one file here
	if Verbose:
		print('\nRunning in testing mode..')
		print('--'*20+'\n')
	
	results_dict_raw = dict()
	results_returned = dict()
	file = file_list_full[0]
	filenum = 0

	with ProcessPoolExecutor(max_workers=Workers) as e:

		if Verbose:
			print('Submitting process_NTuples to ProcessPoolExecutor: [{}]'.format(0))
			print('    File: {}'.format(file))
			print('    Array suffix: {}'.format(str(filenum)))
			print('    Destination: {}'.format(OutDir))
		
		results_dict_raw[0] = e.submit(STMC,
			rootfilename=file, geofile=GeoLoc, outdir=OutDir,
			save=True, savetag=str(filenum) )

	if Verbose:
		print('Processes finished. Showing results!\n')

	for i in range(len(results_dict_raw)):
		res = results_dict_raw[i]
		results_returned[i] = res.result()
		if Verbose:
			print('Result type: {}'.format(type(res)))
			print('Result: {}'.format(res))
			print(res.result())
	tTest1 = t.time()
	print('\nDone!')
	sys.exit('  {:8.4f} (s)\n'.format(tTest1 - tTest0))

if Debug:
	sys.exit('\nExiting due to debugging mode.\n')


#==================#
## PROCESSES ONLY ##==========================================================#
#==================#
if Processes and not Threading:
	tp0 = t.time()
	if Verbose:
		print('\nRunning using processes only..')
		print(' -- {} CPUs'.format(Workers))
		print('--'*20+'\n')
	
	results_dict_raw = dict()
	results_returned = dict()
	
	with ProcessPoolExecutor(max_workers=Workers) as e:
		
		for i, file in enumerate(file_list_full[Start:Start+NFiles]):
			filenum = str(i + Start)
			if Verbose:
				print('Submitting STMC to ProcessPoolExecutor: [{}]'.format(i))
				print('    File: {}'.format(file))
				print('    Array suffix: {}'.format(str(i)))
				print('    Destination: {}'.format(OutDir))

			results_dict_raw[i] = e.submit(STMC,
				rootfilename=file, geofile=GeoLoc, outdir=OutDir,
				save=True, savetag=filenum )

		if Verbose:
			print('\n .. Waiting on ProcessPoolExecutor ..\n')

	if Verbose:
		print('Processes finished. Showing results!\n')

	for i in range(len(results_dict_raw)):
		res = results_dict_raw[i]
		results_returned[i] = res.result()
		if Verbose:
			print('\nProcess: {}'.format(i))
			print('Result type: {}'.format(type(res)))
			print('Result: {}'.format(res))
			print(res.result())

	tp1 = t.time()
	results_returned['total_time'] = tp1 - tp0

	## Save results as a dictionary
	try:

		pklfilename = '{}multiprocess-{}_file-{}-to-{}_dict.pickle'.format(
			OutDir, Workers, Start, Start+NFiles)

		with open(pklfilename, 'wb') as picklefile:
			pickle.dump(results_returned, picklefile)
	
	except PickleError as pke:
		warnings.warn('Unable to pickle file. Printing PickleError: \n')
		print(traceback.format_exc())
		print()

	if Verbose:
		print('\n..Done!..')
		print('  {:8.4f} (s)\n'.format(tp1 - tp0))
#==========================================================================


## THREADS ONLY -----------------------------------------------------------
elif Threading and not Processes:
	tth0 = t.time()
	if Verbose:
		print('\nRunning using multithreading..')
		print(' -- {} threads'.format(Threads))
		print('--'*20+'\n')
	
	results_dict_raw = dict()
	results_returned = dict()
	
	with ThreadPoolExecutor(max_workers=Threads) as e:		
		
		for i, file in enumerate(file_list_full[Start:Start+NFiles]):
			filenum = str(i + Start)
			if Verbose:
				print('Submitting STMC to ProcessPoolExecutor: [{}]'.format(i))
				print('    File: {}'.format(file))
				print('    Array suffix: {}'.format(str(i)))
				print('    Destination: {}'.format(OutDir))

			results_dict_raw[i] = e.submit(STMC,
				rootfilename=file, geofile=GeoLoc, outdir=OutDir,
				save=True, savetag=filenum )

		if Verbose:
			print('\n .. Waiting on ThreadPoolExecutor ..\n')

	if Verbose:
		print('Processes finished. Showing results!\n')

	for i in range(len(results_dict_raw)):
		res = results_dict_raw[i]
		results_returned[i] = res.result()
		if Verbose:
			print('\nProcess: {}'.format(i))
			print('Result type: {}'.format(type(res)))
			print('Result: {}'.format(res))
			print(res.result())

	tth1 = t.time()
	results_returned['total_time'] = tth1 - tth0

	## Save results as a dictionary
	try:

		pklfilename = '{}multithread-{}_file-{}-to-{}_dict.pickle'.format(
			OutDir, Threads, Start, Start+NFiles)

		with open(pklfilename, 'wb') as picklefile:
			pickle.dump(results_returned, picklefile)
	
	except PickleError as pke:
		warnings.warn('Unable to pickle file. Printing PickleError: \n')
		print(traceback.format_exc())
		print()

	if Verbose:
		print('\n..Done!..')
		print('  {:8.4f} (s)\n'.format(tth1 - tth0))
#==========================================================================


## PROCESSES AND THREADS --------------------------------------------------
elif Processes and Threading:
	print('\nRunning using processes and threads..\n')
#==========================================================================


## FOR LOOP ---------------------------------------------------------------
else:
	print('\nNo processes or threads selected..')
	usr_in = input('Are you sure you would like to continue? (y/n)\n')
	if not (usr_in == 'y' or usr_in == 'yes'):
		sys.exit()
	tloop0 = t.time()

	if Verbose:
		print('\nRunning using the dreaded for loop ..')
		print('--'*20+'\n')
	
	results_dict_raw = dict()
	results_returned = dict()
			
	for i, file in enumerate(file_list_full[Start:Start+NFiles]):
		filenum = str(i + Start)
		if Verbose:
			print('Running STMC [{}] in for loop'.format(i))
			print('    File: {}'.format(file))
			print('    Array suffix: {}'.format(str(i)))
			print('    Destination: {}'.format(OutDir))

		results_dict_raw[i] = STMC(
			rootfilename=file, geofile=GeoLoc, outdir=OutDir,
			save=True, savetag=filenum )

		if Verbose:
			print('\n .. Finished file {} ..\n'.format(file))

	if Verbose:
		print('Processes finished. Showing results!\n')

	for i in range(len(results_dict_raw)):
		res = results_dict_raw[i]
		results_returned[i] = res
		if Verbose:
			print('\nLoop: {}'.format(i))
			print('	Result: ')
			print(res.result())

	tloop1 = t.time()
	results_returned['total_time'] = tloop1 - tloop0

	## Save results as a dictionary
	try:

		pklfilename = '{}forloop-{}_file-{}-to-{}_dict.pickle'.format(
			OutDir, Workers, Start, Start+NFiles)

		with open(pklfilename, 'wb') as picklefile:
			pickle.dump(results_returned, picklefile)
	
	except PickleError as pke:
		warnings.warn('Unable to pickle file. Printing PickleError: \n')
		print(traceback.format_exc())
		print()

	if Verbose:
		print('\n..Done!..')
		print('  {:8.4f} (s)\n'.format(tloop1 - tloop0))
#==============================================================================

