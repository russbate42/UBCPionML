'''
##=============================##
## Train Particle Flow Network ##
##=============================##

author: Russell Bate
russellbate@phas.ubc.ca
'''

## META-DATA ##
datapath_prefix = '/fast_scratch/atlas/'
module_path = '/home/russbate/MLPionCollaboration/LCStudies/util/'
BATCH_SIZE=2000
LEARNING_RATE=1e-3
EPOCHS=800
MODEL='PFN_base'
GPU="5"
NEVENTS=int(2e5)

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
from pfn_models import PFN_base, PFN_wDropout, PFN_wTNet


## Read in training params
#======================================
parser = argparse.ArgumentParser(description='flags for training options of Particle'\
                                +' Flow Network')
parser.add_argument('--default', action="store", dest="hard_code",
                    default=False, type=bool)
parser.add_argument('--batch_size', action="store", dest="bs", default=None,
                   type=int)
parser.add_argument('--learning_rate', action="store", dest="lr", default=None,
                   type=float)
parser.add_argument('--epochs', action="store", dest="ep", default=None,
                   type=int)
parser.add_argument('--model', action="store", dest="model", default=None,
                   type=str)
parser.add_argument('--GPU', action="store", dest="gpu", default=None, type=int)
parser.add_argument('--events', action="store", dest="events", default=None,
                   type=int)
parser.add_argument('--full_data', action="store", dest="full_data", default=False,
                   type=bool)
parser.add_argument('--graph_execution', action="store_true",
                    dest="graph_execution")

args = parser.parse_args()

if args.full_data and args.events is not None:
    raise ValueError("Cannot pass ful_data=True and any argument to --events\
    at the same time.")
    
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
    
    exec_str = ""
    if GRAPH_EXECUTION == True:
        exec_str += "GRAPH execution"
    else:
        exec_str += "EAGER execution"
        
    args_list = [BATCH_SIZE, LEARNING_RATE, MODEL, EPOCHS, GPU, NEVENTS]
    
    if any([arg is None for arg in args_list]):
        print('Insufficient flags supplied. Exiting program.')
        sys.exit()
    else:
        print('Training with values: ')
        print('Batch size: {}'.format(BATCH_SIZE))
        print('Learning rate: {}'.format(LEARNING_RATE))
        print('Epochs: {}'.format(EPOCHS))
        print('Model: {}'.format(MODEL))
        print('Training on GPU: {}'.format(GPU))
        print('Training on {} events'.format(NEVENTS))
        print('Eager or graph? -- {}'.format(exec_str))

        
## TF Environment
#======================================
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
    

## Load Data
#======================================
print()
print('Loading data..')
t0 = cput()
if args.full_data:
    X = np.load(datapath_prefix+'X_STMC_full_norm.npy', mmap_mode='r+')
    Y = np.load(datapath_prefix+'Y_STMC_full_norm.npy', mmap_mode='r+')    
else:
    X = np.load(datapath_prefix+'X_STMC_full_norm.npy', mmap_mode='r+')[:NEVENTS,:,:]
    Y = np.load(datapath_prefix+'Y_STMC_full_norm.npy', mmap_mode='r+')[:NEVENTS]
t1 = cput()
print('time to load data: {:6.4f} (s)'.format(t1-t0)); print()
print('X size: {}'.format(X.shape))
print()

train_num, val_num, test_num = dsu.tvt_num(X, tvt=(70,15,15))
print('train -- val -- test')
print('{} -- {} -- {}'.format(train_num, val_num, test_num)); print()

# Split
t0 = cput()

X_train = X[:train_num,:,:]
Y_train = Y[:train_num].reshape((train_num,1))

X_val = X[train_num:train_num+val_num,:,:]
Y_val = Y[train_num:train_num+val_num].reshape((val_num,1))

X_test = X[train_num+val_num:,:,:]
Y_test = Y[train_num+val_num:]
t1 = cput()
print('time to split: {:6.2f} (m)'.format((t1-t0)/60)); print()


## Load Models
#======================================
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
elif MODEL == 'PFN_wAttention':
    print('Not built yet saaaarrryy :) ..')
    sys.exit()
else:
    print('Unknown model. Quitting program.')
    sys.exit()

model.compile(loss='mse', optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE))
model.summary()


## Train Model
#======================================
print()
print('Training model..')
print()

t0 = cput()
history = model.fit(X_train,
                  Y_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_val, Y_val),
                  epochs=EPOCHS,
                  verbose=1
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
print('making predictions..');print()
t0 = cput()
prediction = model.predict(X_test)
t1 = cput()
print('Time to make predictions: {:8.2f} (s)'.format(t1-t0))
print()


## Save Information
#======================================
infostring = '{}_LR--{:.0e}_BS--{}_EP--{}_EV--{}_{}'.format(MODEL,
                LEARNING_RATE, BATCH_SIZE, EPOCHS, NEVENTS, DATE)
print('saving_files..')
with open('results/history_'+infostring+'.pickle', 'wb')\
         as histfile:
    pickle.dump(history.history, histfile)


np.savez('results/target_preds_'+infostring,
        args=(Y_test, np.squeeze(prediction, axis=1)), kwds=('target', 'prediction'))
print()

print('saving model..')
model.save('models/model_'+infostring)

print()
print()
print('..le fin..')
print()


    
