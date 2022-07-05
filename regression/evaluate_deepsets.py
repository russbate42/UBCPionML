'''
##=================================##
## Evaluate Trained DeepSet Models ##
##=================================##

author: Russell Bate
russellbate@phas.ubc.ca
russell.bate@cern.ch
'''

## General Python Imports
#======================================
import numpy as np
import pickle, sys, argparse, copy
import time as t
from time import perf_counter as cput
from datetime import datetime
DATE = datetime.today().strftime('%Y-%m-%d')
print()


## Local ML Packages
#======================================
module_path = '/home/russbate/MLPionCollaboration/LCStudies/'
sys.path.append(module_path)
from util import deep_set_util as dsu
import pfn_models
from pfn_models import PFN_base, PFN_wDropout, PFN_wTNet, DNN


## Read in training params
#======================================
parser = argparse.ArgumentParser(description='Flags for evaluation of DeepSets')
parser.add_argument('--file_loc', action="store", dest='fl', default=\
                    '/fast_scratch_1/atlas/normalized/', type=str)
parser.add_argument('--output_loc', action="store", dest='ol', default=\
                    '/fast_scratch_1/atlas/predictions/', type=str)
parser.add_argument('--model', action="store", dest="model", default=None,
                   type=str)
parser.add_argument('--batch_size', action="store", dest="bs", default=None,
                   type=int)
parser.add_argument('--data', action="store", dest='datafiles',
                    default='STMC', type=str)
parser.add_argument('--save_tag', action="store", dest='savetag',
                    default=None, type=str)
parser.add_argument('--full_data', action="store_true", dest="full_data")
parser.add_argument('--drop_remainder', action="store", dest="drop_remainder",
                    default=False, type=bool)
parser.add_argument('--graph_execution', action="store_true",
                    dest="graph_execution")
parser.add_argument('--GPU', action="store", dest="gpu", default=None, type=int)
parser.add_argument('--events', action="store", dest="events", default=None,
                   type=int)
parser.add_argument('--save_results', action="store_true", dest="save_results")
parser.add_argument('--generator', action="store", dest="generator", default=True,
                   type=bool)
parser.add_argument('--verbose', action="store", dest='vb',
                    default=1, type=int)
args = parser.parse_args()


file_loc = args.fl
output_loc = args.ol
drop_remainder = args.drop_remainder
verbose = args.vb
full_data = args.full_data
GENERATOR = args.generator
BATCH_SIZE = args.bs
MODEL = args.model
GPU = str(args.gpu)
MODEL = args.model
NEVENTS = args.events
GRAPH_EXECUTION = args.graph_execution


## Add information to the savestring!
if args.savetag is None:
    extra_info = ''
else:
    extra_info = '_'+args.savetag

## Arg checks!!!
if full_data and args.events is not None:
    raise ValueError("Cannot pass ful_data=True and any argument to --events\
    at the same time.")
if full_data == False and args.events is None:
    raise ValueError("Need to give number of events or use --full_data=True\
    to specify full data set.")


##================##
## TF Environment ##
##================##
import tensorflow as tf
from tensorflow import keras
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

## Eager execution
if GRAPH_EXECUTION == True:
    print('Running in GRAPH execution mode!')
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
else:
    print('Running in EAGER execution mode!')


    ## Choose which files to train from!
datafiles = args.datafiles
# the following three are from the STMC_v2 dataset
if datafiles == 'STMC':
    Xfile_ev = 'X_STMC_v2_100_test_norm.npy'
    Yfile_ev = 'Y_STMC_v2_100_test_norm.npy'
    Etafile_ev = 'Eta_STMC_v2_100_test_norm.npy'
    pred_str = Xfile_ev[:13]
    y_str = Yfile_ev[:23]
    e_str = Etafile_ev[:25]
elif datafiles == 'CL-TR':
    sys.exit('Have not created this data-set yet. Exiting')
    #Xfile = 'X_STMC_v2_Cl_25.npy'
    # Yfile = 'Y_STMC_v2_25_norm2.npy'
    # Etafile = 'Eta_STMC_v2_25_norm2.npy'
elif datafiles == 'TR-CL':
    sys.exit('Have not created this data-set yet. Exiting')
    # Xfile = 'X_STMC_v2_Tr_25.npy'
    # Yfile = 'Y_STMC_v2_25_norm2.npy'
    # Etafile = 'Eta_STMC_v2_25_norm2.npy'
# this is a new dataset with Cluster Only
elif datafiles == 'PIPM':
    Xfile_ev = 'X_CLO_PIPM_100_test_norm.npy'
    Yfile_ev = 'Y_CLO_PIPM_100_test_norm.npy'
    Etafile_ev = 'Eta_CLO_PIPM_100_test_norm.npy'
    pred_str = Xfile_ev[:13]
    y_str = Yfile_ev[:23]
    e_str = Etafile_ev[:25]
elif datafiles == 'PI0':
    Xfile_ev = 'X_CLO_PI0_100_test_norm.npy'
    Yfile_ev = 'Y_CLO_PI0_100_test_norm.npy'
    Etafile_ev = 'Eta_CLO_PI0_100_test_norm.npy'
    pred_str = Xfile_ev[:13]
    y_str = Yfile_ev[:23]
    e_str = Etafile_ev[:25]
elif datafiles == 'PIPM+PI0':
    Xfile_ev1 = 'X_CLO_PI0_100_test_norm.npy'
    Yfile_ev1 = 'Y_CLO_PI0_100_test_norm.npy'
    Etafile_ev1 = 'Eta_CLO_PI0_100_test_norm.npy'
    Xfile_ev2 = 'X_CLO_PI0_100_test_norm.npy'
    Yfile_ev2 = 'Y_CLO_PI0_100_test_norm.npy'
    Etafile_ev2 = 'Eta_CLO_PI0_100_test_norm.npy'
    pred_str = 'X_CLO_100_combined_test'
    y_str = 'Y_CLO_100_combined_test'
    e_str = 'Eta_CLO_100_combined_test'
    ''' Maybe do the concatenation step in here? Or at least a switch...'''
else:
    raise ValueError('Incorrect option passed to --data')


##==================##
## Save Information ##
##==================##
#  os.system('rm '+output_loc+'/Y_large.npy')
pred_infostr = '{}/'.format(MODEL)+pred_str+'_predictions_{}{}'.format(DATE,
                                                                extra_info)
y_infostr = '{}/'.format(MODEL)+y_str+'_{}{}'.format(DATE,
                                                     extra_info)
e_infostr = '{}/'.format(MODEL)+e_str+'_{}{}'.format(DATE,
                                                          extra_info)
print()
print(pred_infostr)
print(y_infostr)
print(e_infostr)
print()

#=======================#
## GENERATOR FUNCTIONS ##
#=======================#

def sgen_helper(batch_size, X, drop_remainder=True):
    num_tot = X.shape[0]
        
    theList = []
    if drop_remainder == True:
        nbatch = np.floor(num_tot/batch_size).astype(int)
    else:
        nbatch = np.ceil(num_tot/batch_size).astype(int)
    
    for i in range(nbatch):
        low = i*batch_size
        high = (i+1)*batch_size
        if high > num_tot:
            theList.append((low,num_tot))
            break
        else:
            theList.append((low,high))
            
    return theList


def eval_gen(batch_size, X, drop_remainder=True):
    
    slices  = sgen_helper(batch_size, X, drop_remainder=drop_remainder)

    for (k,l) in slices:
        yield X[k:l,:,:]

#=====================================#
## Load Data =========================#
#=====================================#
print()
print('Loading data..')
t0 = cput()
if args.full_data:
    X = np.load(file_loc+Xfile_ev, mmap_mode='r+')
    Y = np.load(file_loc+Yfile_ev, mmap_mode='r+')
    E = np.load(file_loc+Etafile_ev, mmap_mode='r+')
    NEVENTS = X.shape[0]
else:
    X = np.load(file_loc+Xfile_ev, mmap_mode='r+')\
            [:NEVENTS,:,:]
    Y = np.load(file_loc+Yfile_ev, mmap_mode='r+')\
            [:NEVENTS]
    E = np.load(file_loc+Etafile_ev, mmap_mode='r+')\
            [:NEVENTS]


##=============##
## OPEN OUTPUT ##
##=============##
if drop_remainder:
    bit_extra = NEVENTS % BATCH_SIZE
    last_idx = NEVENTS - bit_extra
else:
    last_idx = NEVENTS

## Check filepath
dir_path = os.path.dirname(output_loc+pred_infostr)
if not os.path.exists(dir_path):
    print('Directory {}'.format(dir_path))
    print('not found. Creating new directory.')
    print()
    os.makedirs(dir_path)

prediction = np.lib.format.open_memmap(output_loc+pred_infostr,
                mode='w+', dtype=np.float64, shape=(last_idx,1))
Y_out = np.lib.format.open_memmap(output_loc+y_infostr,
                mode='w+', dtype=np.float64, shape=(last_idx,1))
E_out = np.lib.format.open_memmap(output_loc+e_infostr,
                mode='w+', dtype=np.float64, shape=(last_idx,1))
print('copying Y and Eta..')
Y_out[:,0] = Y
E_out[:,0] = E
t1 = cput()
print('Time to load data and create outputs: {:6.4f} (s)'.format(t1-t0))
print()
print('X size: {}'.format(X.shape))
print()

##=============##
## Load Models ##
#==============##
print('Loading models..')
load_mod_bool = False
while load_mod_bool == False:
    try:
        loaded_model = keras.models.load_model('models/'+MODEL)
        load_mod_bool = True
    except ImportError:
        print('Caught Exception: ImportError')
        print('This arises as a result of loading hdf5 files.')
        sys.exit('Exiting program.')
    except IOError:
        print('Model failed to load with specified string.')
        print()
        # sys.exit('Exiting program.')
        raise IOError
    
loaded_model.summary()


#=====================================#
##======= Evaluate Model ============##
#=====================================#
print()
print('Evaluating model..')
print()
t0 = cput()

## Generators ##
if GENERATOR:

    print()
    print('Evaluating using generators...')
    print('Number of events: {}'.format(NEVENTS))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Drop remainder = {}'.format(drop_remainder))
    print();print()
    eval_dat = eval_gen(batch_size=BATCH_SIZE,
                        X=X, drop_remainder=drop_remainder)
    
    print('entering model.predict()')
    prediction[:,:] = loaded_model.predict(eval_dat,
                        max_queue_size=10,
                        verbose=verbose
                        )
    
else:
    print('evaluating without generators...')
    print('Warning this is deprecated and needs work...')
    print('Behavior may be unexpected.')
    prediction = loaded_model.predict(X_train,
                      Y_train,
                      batch_size=BATCH_SIZE,
                      validation_data=(X_val, Y_val),
                      epochs=EPOCHS,
                      verbose=verbose
                      )
t1 = cput()
print()
print()
print('Time to evaluate: {:8.2f} (s)'.format(t1-t0))
print('                  {:8.2f} (min)'.format((t1-t0)/60))
print('                  {:8.2f} (hour)'.format((t1-t0)/3600))
print()
print('Flushing memory..')
t0 = cput()
prediction.flush()
Y_out.flush()
E_out.flush()
t1 = cput()
print('Time to flush: {:8.2f} (min)'.format((t1-t0)/60))
