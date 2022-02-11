'''
##=============================##
## Train Particle Flow Network ##
##=============================##

author: Russell Bate
russellbate@phas.ubc.ca
'''

## META-DATA ##
datapath_prefix = '/data/atlas/rbate/'
module_path = '/home/russbate/MLPionCollaboration/LCStudies/util/'
BATCH_SIZE=2000
LEARNING_RATE=1e-3
EPOCHS=10
MODEL='PFN_base'

## General Python Imports
#======================================
import numpy as np
import pickle
import time as t
import sys
from time import perf_counter as cput
import argparse
print()

## Local ML Packages
#======================================
sys.path.append(module_path)
import deep_set_util as dsu
import pfn_models
from pfn_models import PFN_base, PFN_wDropout, PFN_wTNet

## TF Environment
#======================================
import tensorflow as tf
from tensorflow import keras
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


## Read in training params
#======================================
parser = argparse.ArgumentParser(description='testing argparse')
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

args = parser.parse_args()

if args.hard_code:
    print('starting training with hard coded values..')
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Learning rate: {}'.format(LEARNING_RATE))
    print('Epochs: {}'.format(EPOCHS))
    print('Model: {}'.format(MODEL))
    
else:
    BATCH_SIZE = args.bs
    LEARNING_RATE = args.lr
    MODEL = args.model
    EPOCHS = args.ep
    
    args_list = [BATCH_SIZE, LEARNING_RATE, MODEL]
    
    if any([arg is None for arg in args_list]):
        print('Insufficient flags supplied. Exiting program.')
        sys.exit()
    else:
        print('Training with values: ')
        print('Batch size: {}'.format(BATCH_SIZE))
        print('Learning rate: {}'.format(LEARNING_RATE))
        print('Epochs: {}'.format(EPOCHS))
        print('Model: {}'.format(MODEL))


## Load Data
#======================================
print()
print('Loading data..')
t0 = cput()
X = np.load(datapath_prefix+'X_STMC_full_norm.npy', mmap_mode='r')[:50000,:,:]
Y = np.load(datapath_prefix+'Y_STMC_full_norm.npy', mmap_mode='r')[:50000]
t1 = cput()
print('time to load data: {:f4.2}'.format(t1-t0)); print()
print('X size: {}'.format(X.shape))
print()

train_num, val_num, test_num = dsu.tvt_num(X, tvt=(70,15,15))
print('train -- val -- test')
print('{} -- {} -- {}'.format(train_num, val_num, test_num)); print()

# Shuffle split
t0 = cput()
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices,:,:]
Y = Y[indices]

X_train = X[:train_num,:,:]
Y_train = Y[:train_num].reshape((train_num,1))

X_val = X[train_num:train_num+val_num,:,:]
Y_val = Y[train_num:train_num+val_num].reshape((val_num,1))

X_test = X[train_num+val_num:,:,:]
Y_test = Y[train_num+val_num:]
t1 = cput()
print('time to shuffle\'n\'split: {:f4.2} (m)'.format((t1-t0)/60)); print()

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
else:
    print('Unknown model. Quitting program.')
    sys.exit()

model.compile(loss='mse', optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE))
model.summary()

print()
print('Training model..')
print()

history = model.fit(X_train,
                  Y_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_val, Y_val),
                  epochs=EPOCHS,
                  verbose=1
                  )
print()
print('..le fin..')
print()


    
