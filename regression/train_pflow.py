'''
##=============================##
## Train Particle Flow Network ##
##=============================##

author: Russell Bate
russellbate@phas.ubc.ca
russell.bate@cern.ch
'''

## META-DATA ##
datapath_prefix = '/fast_scratch_1/atlas/'
module_path = '/home/russbate/MLPionCollaboration/LCStudies/util/'
BATCH_SIZE=2000
LEARNING_RATE=1e-3
EPOCHS=4
MODEL='PFN_base'
GPU="6"
NEVENTS=int(5e5)
SAVE_MODEL = False
SAVE_RESULTS = False
GRAPH_EXECUTION = False

## General Python Imports
#======================================
import numpy as np
import pickle
import time as t
import sys
from time import perf_counter as cput
import argparse
from datetime import datetime
DATE = datetime.today().strftime('%Y-%m-%d')
print()

## Local ML Packages
#======================================
sys.path.append(module_path)
import deep_set_util as dsu
import pfn_models
from pfn_models import PFN_base, PFN_wDropout, PFN_wTNet, DNN, PFN_large


## Read in training params
#======================================
parser = argparse.ArgumentParser(description='flags for training options of Particle'\
                                +' Flow Network')
parser.add_argument('--default', action="store", dest="hard_code",
                    default=False, type=bool)
parser.add_argument('--file_loc', action="store", dest='fl', default=\
                    '/fast_scratch_1/atlas/normalized/',
                   type=str)
parser.add_argument('--output_loc', action="store", dest='ol', default=\
                    '/fast_scratch_1/atlas/normalized/',
                   type=str)
parser.add_argument('--batch_size', action="store", dest="bs", default=None,
                   type=int)
parser.add_argument('--learning_rate', action="store", dest="lr", default=None,
                   type=float)
parser.add_argument('--epochs', action="store", dest="ep", default=None,
                   type=int)
parser.add_argument('--train_val', action="store", dest='tv', default=\
                    [90,10], type=int, nargs="+")
parser.add_argument('--model', action="store", dest="model", default=None,
                   type=str)
parser.add_argument('--GPU', action="store", dest="gpu", default=None, type=int)
parser.add_argument('--max_queue', action="store", dest="MQ", default=10, type=int)
parser.add_argument('--events', action="store", dest="events", default=None,
                   type=int)
parser.add_argument('--full_data', action="store_true", dest="full_data")
parser.add_argument('--generator', action="store", dest="generator", default=True,
                   type=bool)
parser.add_argument('--drop_remainder', action="store", dest="drop_remainder", default=True,
                   type=bool)
parser.add_argument('--graph_execution', action="store_true",
                    dest="graph_execution")
parser.add_argument('--save_model', action="store_true", dest="save_model")
parser.add_argument('--save_results', action="store_true", dest="save_results")
parser.add_argument('--save_hist', action="store_true", dest="save_history")
parser.add_argument('--save_np', action="store_true", dest="save_numpy")
parser.add_argument('--data', action="store", dest='datafiles',
                    default='STMC', type=str)
parser.add_argument('--evaluate', action="store_true", dest='eval')
parser.add_argument('--save_tag', action="store", dest='savetag',
                    default=None, type=str)
parser.add_argument('--verbose', action="store", dest='vb',
                    default=1, type=int)
# parser.add_argument('--batch_mult', action='store', dest='batch_mult',
                    # default=None, type=int)

args = parser.parse_args()
evaluate_model = args.eval
file_loc = args.fl
output_loc = args.ol
train_ratio, val_ratio = args.tv
drop_remainder = args.drop_remainder
verbose = args.vb
full_data = args.full_data
SAVE_HISTORY = args.save_history
SAVE_NUMPY = args.save_numpy
max_queue = args.MQ
SAVE_RESULTS = args.save_results

if SAVE_HISTORY and SAVE_NUMPY:
    SAVE_RESULTS = True
    print('save results set to true')    

if verbose not in [0,1,2]:
    raise ValueError('Verbose must be in [0,1,2].')

if drop_remainder == False:
    sys.exit('Code not set up for drop_remainder=False yet. Exiting.')

if train_ratio + val_ratio != 100:
    raise ValueError('Train_ratio and val_ratio need to add to 100!')

if full_data and args.events is not None:
    raise ValueError("Cannot pass ful_data=True and any argument to --events\
    at the same time.")
if full_data == False and args.events is None:
    raise ValueError("Need to give number of events or use --full_data=True\
    to specify full data set.")

if args.hard_code:
    print('starting training with hard coded values..')
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Learning rate: {}'.format(LEARNING_RATE))
    print('Epochs: {}'.format(EPOCHS))
    print('Model: {}'.format(MODEL))
    print('GPU: {}'.format(GPU))
    if args.full_data:
        print('Training on full dataset... [?] events.')
    else:
        print('Training on {} events'.format(NEVENTS))
    
else:
    BATCH_SIZE = args.bs
    LEARNING_RATE = args.lr
    MODEL = args.model
    EPOCHS = args.ep
    GPU = str(args.gpu)
    NEVENTS = args.events
    GRAPH_EXECUTION = args.graph_execution
    SAVE_MODEL = args.save_model
    GENERATOR = args.generator
    
    exec_str = ""
    if GRAPH_EXECUTION == True:
        exec_str += "GRAPH execution"
    else:
        exec_str += "EAGER execution"
        
    args_list = [BATCH_SIZE, LEARNING_RATE, MODEL, EPOCHS, GPU]
    
    if any([arg is None for arg in args_list]):
        print('Mandatory flags are: {}'.format(args_list)); print()
        print('Insufficient flags supplied. Exiting program.')
        sys.exit()
    else:
        print('Training with values: ')
        print('Batch size: {}'.format(BATCH_SIZE))
        print('Learning rate: {}'.format(LEARNING_RATE))
        print('Epochs: {}'.format(EPOCHS))
        print('Model: {}'.format(MODEL))
        print('Training on GPU: {}'.format(GPU))
        if full_data:
            print('Training on full data')
        else:
            print('Training on {} events'.format(NEVENTS))
        print('Eager or graph? -- {}'.format(exec_str))


## Choose which files to train from!
datafiles = args.datafiles
# the following three are from the STMC_v2 dataset
if datafiles == 'STMC':
    Xfile_tr = 'X_STMC_v2_400_train_norm.npy'
    Yfile_tr = 'Y_STMC_v2_400_train_norm.npy'
    Etafile_tr = 'Eta_STMC_v2_400_train_norm.npy'
    Xfile_ev = 'X_STMC_v2_100_test_norm.npy'
    Yfile_ev = 'Y_STMC_v2_100_test_norm.npy'
    Etafile_ev = 'Eta_STMC_v2_100_test_norm.npy'
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
    Xfile_tr = 'X_CLO_PIPM_400_train_norm.npy'
    Yfile_tr = 'Y_CLO_PIPM_400_train_norm.npy'
    Etafile_tr = 'Eta_CLO_PIPM_400_train_norm.npy'
    Xfile_ev = 'X_CLO_PIPM_100_test_norm.npy'
    Yfile_ev = 'Y_CLO_PIPM_100_test_norm.npy'
    Etafile_ev = 'Eta_CLO_PIPM_100_test_norm.npy'
elif datafiles == 'PI0':
    Xfile_tr = 'X_CLO_PI0_400_train_norm.npy'
    Yfile_tr = 'Y_CLO_PI0_400_train_norm.npy'
    Etafile_tr = 'Eta_CLO_PI0_400_train_norm.npy'
    Xfile_ev = 'X_CLO_PI0_100_test_norm.npy'
    Yfile_ev = 'Y_CLO_PI0_100_test_norm.npy'
    Etafile_ev = 'Eta_CLO_PI0_100_test_norm.npy'
elif datafiles == 'PIPM+PI0':
    Xfile_tr1 = 'X_CLO_PI0_400_train_norm.npy'
    Yfile_tr1 = 'Y_CLO_PI0_400_train_norm.npy'
    Etafile_tr1 = 'Eta_CLO_PI0_400_train_norm.npy'
    Xfile_ev1 = 'X_CLO_PI0_100_test_norm.npy'
    Yfile_ev1 = 'Y_CLO_PI0_100_test_norm.npy'
    Etafile_ev1 = 'Eta_CLO_PI0_100_test_norm.npy'
    Xfile_tr2 = 'X_CLO_PI0_400_train_norm.npy'
    Yfile_tr2 = 'Y_CLO_PI0_400_train_norm.npy'
    Etafile_tr2 = 'Eta_CLO_PI0_400_train_norm.npy'
    Xfile_ev2 = 'X_CLO_PI0_100_test_norm.npy'
    Yfile_ev2 = 'Y_CLO_PI0_100_test_norm.npy'
    Etafile_ev2 = 'Eta_CLO_PI0_100_test_norm.npy'
else:
    raise ValueError('incorrect option passed to --data')

## Add information to the savestring!
if args.savetag is None:
    extra_info = ''
else:
    extra_info = '_'+args.savetag

    
#=======================#
## GENERATOR FUNCTIONS ##
#=======================#
def gen_helper(batch_size, X, Y, drop_remainder=True):
    num_tot = X.shape[0]
    if num_tot != Y.shape[0]:
        raise ValueError('Data input sizes mis-matched for generator')
        
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


def data_gen(batch_size, X, Y, epochs, shuffle_per=1, drop_remainder=True):
    
    slices  = gen_helper(batch_size, X, Y, drop_remainder=drop_remainder)
    indices = np.arange(X.shape[0], dtype=int)
    
    for i in range(epochs):
        
        if shuffle_per == None:
            pass        
        elif i % shuffle_per == 0:
            np.random.shuffle(indices)
        
        for (k,l) in slices:
            idx_slc = indices[k:l]
            yield (X[idx_slc,:,:], Y[idx_slc])
            

def data_gen2(batch_size, X, Y, epochs, shuffle_per=1, drop_remainder=True,
             val=False):
    
    slices  = gen_helper(batch_size, X, Y, drop_remainder=drop_remainder)
    print('slices: {}'.format(slices))
    
    if val == False:
        tv_str = 'Training'
    else:
        tv_str = 'Validation'
        
    for i in range(epochs):
        print(tv_str+' epoch number: {}'.format(i))
        
        if shuffle_per == None:
            pass
        elif i % shuffle_per == 0:
            print('Shuffling..')
            indices = np.arange(X.shape[0], dtype=int)
            np.random.shuffle(indices)
            Xnew = X[indices,:,:]
            Ynew = Y[indices]
            print('Done..')
        
        for (j, (k,l)) in enumerate(slices):
            print(tv_str+' batch number: {}'.format(j))
            yield (Xnew[k:l,:,:], Ynew[k:l])
    

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
    

#=====================================#
## Load Data =========================#
#=====================================#
print()
print('Loading data..')
t0 = cput()
if datafiles == 'PIPM+PI0': # have to do something different for combined!

    MAX_NPTS = 1300 # this is hard coded and arbitrary
    maxClust = 7882736

    ## if the data files exist concatenated continuing is simple
    if args.full_data:
        NEVENTS = 2*maxClust
        x_str = 'X_combined_clust_full.npy'
        y_str = 'Y_combined_clust_full.npy'
        e_str = 'Eta_combined_clust_full.npy'

    x_exists = os.path.exists(file_loc+x_str)
    y_exists = os.path.exists(file_loc+y_str)
    e_exists = os.path.exists(file_loc+e_str)

    if x_exists and y_exists and e_exists:
        X = np.load(file_loc+x_str, mmap_mode='r+')
        Y = np.load(file_loc+y_str, mmap_mode='r+')
        E = np.load(file_loc+e_str, mmap_mode='r+')

    ## if files are not concatenated then we must do it here
    else:
        print('concatenating pion datasets..')
        if args.full_data:
            x_str = 'X_combined_clust_full.npy'
            y_str = 'Y_combined_clust_full.npy'
            e_str = 'Eta_combined_clust_full.npy'
            NEVENTS = 2*maxClust
            _nevents = maxClust
        else:
            # we will delete this at the end
            x_str = 'X_combined_clust_{}.npy'.format(NEVENTS)
            y_str = 'Y_combined_clust_{}.npy'.format(NEVENTS)
            e_str = 'Eta_combined_clust_{}.npy'.format(NEVENTS)

            if NEVENTS > 2*maxClust:
                raise ValueError('Do not have this number of events available.\n'\
                                'Maximum is {}.'.format(2*maxClust))
            _nevents = np.floor(NEVENTS/2).astype(int)
        try:
            # load both
            X1 = np.load(file_loc+Xfile_tr1, mmap_mode='r+')\
                    [:_nevents,:,:]
            Y1 = np.load(file_loc+Yfile_tr1, mmap_mode='r+')\
                    [:_nevents]
            E1 = np.load(file_loc+Etafile_tr1, mmap_mode='r+')\
                    [:_nevents]
            X2 = np.load(file_loc+Xfile_tr2, mmap_mode='r+')\
                    [:_nevents,:,:]
            Y2 = np.load(file_loc+Yfile_tr2, mmap_mode='r+')\
                    [:_nevents]
            E2 = np.load(file_loc+Etafile_tr2, mmap_mode='r+')\
                    [:_nevents]

            # load larger array from memmap
            X = np.lib.format.open_memmap(
                    file_loc+x_str, mode='w+', dtype=np.float64,
                    shape=(NEVENTS, MAX_NPTS, 4))
            Y = np.lib.format.open_memmap(
                    file_loc+y_str, mode='w+', dtype=np.float64,
                    shape=(NEVENTS,))
            E = np.lib.format.open_memmap(
                    file_loc+e_str, mode='w+', dtype=np.float64,
                    shape=(NEVENTS,))

            # copy and flush
            X[:_nevents,:,:] = X1
            X[_nevents:,:,:] = X2
            X.flush()
            Y[:_nevents] = Y1
            Y[_nevents:] = Y2
            Y.flush()
            E[:_nevents] = E1
            E[_nevents:] = E2
            E.flush()
            print('finished concatenating data sets')

        except OSError:
            print()
            print('-'*79)
            print('Caught OSError. This is likely occuring because the file')
            print('cannot be opened due to OOM. Raising exact error.')
            print('-'*79);print()
            raise OSError

else: # normal circumstances where we do not have to 
    if args.full_data:
        X = np.load(file_loc+Xfile_tr, mmap_mode='r+')
        Y = np.load(file_loc+Yfile_tr, mmap_mode='r+')
        E = np.load(file_loc+Etafile_tr, mmap_mode='r+')
        NEVENTS = X.shape[0]
    else:
        X = np.load(file_loc+Xfile_tr, mmap_mode='r+')\
                [:NEVENTS,:,:]
        Y = np.load(file_loc+Yfile_tr, mmap_mode='r+')\
                [:NEVENTS]
        E = np.load(file_loc+Etafile_tr, mmap_mode='r+')\
                [:NEVENTS]    
t1 = cput()
print('time to load data: {:6.4f} (s)'.format(t1-t0)); print()
print('X size: {}'.format(X.shape))
print()


train_num = np.floor((train_ratio/100.) * NEVENTS).astype(int)
val_num = NEVENTS - train_num

print('train -- val')
print('{} -- {}'.format(train_num, val_num)); print()

# Split
t0 = cput()
X_train = X[:train_num,:,:]
Y_train = Y[:train_num].reshape((train_num,1))

X_val = X[train_num:train_num+val_num,:,:]
Y_val = Y[train_num:train_num+val_num].reshape((val_num,1))
t1 = cput()

print('time to fiddle with data: {:6.2f} (m)'.format((t1-t0)/60)); print()


##=============##
## Load Models ##
##=============##
print('Loading models..')
if MODEL == 'PFN_base':
    model = PFN_base(num_points=X.shape[1], num_features=X.shape[2],
                     name=MODEL)
elif MODEL == 'PFN_wTNet':
    model = PFN_wTNet(num_points=X.shape[1], num_features=X.shape[2],
                     name=MODEL)
elif MODEL == 'PFN_wDropout':
    model = PFN_wDropout(num_points=X.shape[1], num_features=X.shape[2],
                     name=MODEL)
elif MODEL == 'PFN_large':
    model = PFN_large(num_points=X.shape[1], num_features=X.shape[2],
                     name=MODEL)
elif MODEL == 'DNN':
    model = DNN(num_features=X.shape[1],
                     name=MODEL)
elif MODEL == 'PFN_wAttention':
    sys.exit('Not built yet saaaarrryy :) ..')
else:
    sys.exit('Unknown model. Quitting program.')

model.compile(loss='mse', optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE))
model.summary()


#===================================#
##========== Callbacks ============##
#===================================#
# need this here for the defined callbacks
infostring = '{}_{}--LR_{:.0e}--BS_{}--EP_{}--EV_{}--{}{}'.format(MODEL,
                datafiles, LEARNING_RATE, BATCH_SIZE, EPOCHS, NEVENTS, DATE,
                extra_info)
checkpoint_filepath = 'models/model_checkpoint_'+infostring
checkpoint_callback_verbose=1
early_stopping_verbose=1

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    verbose=checkpoint_callback_verbose,
    save_best_only=True
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0005,
    patience=5,
    verbose=early_stopping_verbose,
    mode='min',
    baseline=None,
    restore_best_weights=False
)

callbacks = [model_checkpoint_callback, early_stopping_callback]


#=====================================#
##========== Train Model ============##
#=====================================#
print()
print('Training model..')
print()
t0 = cput()

## Generators ##
if GENERATOR == True:
    if drop_remainder:
        steps_per_epoch = np.floor(train_num/BATCH_SIZE).astype(int)
        val_steps = np.floor(val_num/BATCH_SIZE).astype(int)
    else:
        steps_per_epoch = np.ceil(train_num/BATCH_SIZE).astype(int)
        val_steps = np.ceil(val_num/BATCH_SIZE).astype(int)
        
    print()
    print('Training using generators...')
    print('Number of events: {}'.format(NEVENTS))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Epochs: {}'.format(EPOCHS))
    print('Drop remainder = {}'.format(drop_remainder))
    print('steps_per_epoch: {}'.format(steps_per_epoch))
    print('validation_steps: {}'.format(val_steps))
    print('Split: {}/{}'.format(train_ratio,val_ratio))
    print();print()
    train_tuple = data_gen(batch_size=BATCH_SIZE,
                            X=X_train, Y=Y_train, epochs=EPOCHS,
                            shuffle_per=1,
                            drop_remainder=drop_remainder)
    val_tuple = data_gen(batch_size=BATCH_SIZE,
                          X=X_val, Y=Y_val, epochs=EPOCHS,
                          shuffle_per=1,
                          drop_remainder=drop_remainder)
    
    print('entering fit')
    history = model.fit(train_tuple,
                        validation_data=val_tuple,
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps-1,
                        max_queue_size=max_queue,
                        verbose=verbose,
                        callbacks=callbacks
                        )
    
## Mem-maps ##
else:
    print('Training without generators...')
    history = model.fit(X_train,
                      Y_train,
                      batch_size=BATCH_SIZE,
                      validation_data=(X_val, Y_val),
                      epochs=EPOCHS,
                      verbose=verbose,
                      callbacks=callbacks
                      )
t1 = cput()

print()
print()
print('Time to train: {:8.2f} (s)'.format(t1-t0))
print('               {:8.2f} (min)'.format((t1-t0)/60))
print('               {:8.2f} (hour)'.format((t1-t0)/3600))
print()


## Make Predictions
#======================================
if evaluate_model:
    ''' This needs work, well the function that makes the data generators
    needs to work with a single input. Possibly, we call the evaluate
    script from within!? '''

    print('making predictions..');print()
    t0 = cput()
    prediction = model.predict(X_test)
    t1 = cput()
    print('Time to make predictions: {:8.2f} (s)'.format(t1-t0))
    print()


##==================##
## Save Information ##
##==================##

if SAVE_RESULTS == True:
    print('saving results..');print()
    with open('results/history_'+infostring+'.pickle', 'wb')\
             as histfile:
        pickle.dump(history.history, histfile)

    # Evaluate needs to be set to true
    # np.savez('results/target_preds_'+infostring,
    #         args=(Y_test, np.squeeze(prediction,
    #         axis=1), E_test), kwds=('target', 'prediction', 'Eta'))

elif SAVE_HISTORY == True:
    print('Saving history of model only.')
    with open('results/history_'+infostring+'.pickle', 'wb')\
             as histfile:
        pickle.dump(history.history, histfile)
    
elif SAVE_NUMPY == True:
    print('Saving results of evaluation as numpy arrays.')
    np.savez('results/target_preds_'+infostring,
            args=(Y_test, np.squeeze(prediction,
            axis=1), E_test), kwds=('target', 'prediction', 'Eta'))

else:
    print('No saved results flag. No results will be saved.')

if SAVE_MODEL == True:
    print('saving model..')
    model.save('models/model_'+infostring)
    
else:
    print('No saved model flag. Model will not be saved.')
    
## Delete combined results if we are training on both
if datafiles == 'PIPM+PI0':
    pass # remove files here
    
print()
print()
print('..le fin..')
print()


    
