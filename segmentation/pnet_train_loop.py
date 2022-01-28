'''
##+===========================##
### TIME DISTRIBUTED PointNet ##
##============================##

author: Russell Bate
russellbate@phas.ubc.ca
'''


VISIBLE_GPUS = "6"
datapath_prefix = "/data/atlas/rbate/"
module_path = '/home/russbate/MLPionCollaboration/LCStudies/'
log_file = ""


## General imports
#======================================
import numpy as np
import matplotlib.pyplot as plt
import time as t
from time import perf_counter as cput
import logging


## local ML Packages
import sys
sys.path.append(module_path)
from util import resolution_util as ru
from util import plot_util as pu
from util import deep_set_util as dsu
import pnet_models
from pnet_models import PointNet_delta


# logging
# log_fh = logging.FIleHandler(log_file)
# log_sh = logging.StreamHandler(sys.stdout)
# log_eh = logging.StreamHandler(sys.stderr)
# log_format = '%(asctime)s %(levelname)s: %(message)s'
# log_level = 'INFO'



## TensorFlow
#======================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = VISIBLE_GPUS

import tensorflow as tf
from tensorflow import keras

## IMPORTANT ## ====== ## DISABLE EAGER EXECUTION WITH TensorFlow!! ##
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

print("TensorFlow version {}".format(tf.__version__))
print("Keras version {}".format(keras.__version__))


## Load Data
#======================================
t0 = cput()
Xraw = np.load(datapath_prefix+'Rho_X_20_files.npy', mmap_mode='r')[:,:,:]
Yraw = np.load(datapath_prefix+'Rho_Y_segm_20_files.npy', mmap_mode='r')[:,:,:]
print(Xraw.shape)
print(Yraw.shape)

X = np.lib.format.open_memmap(datapath_prefix+'XR_notebook.npy',
                             mode='w+', dtype=np.float32, shape=(Xraw.shape[0], Xraw.shape[1], 4))

Y = np.lib.format.open_memmap(datapath_prefix+'YR_notebook.npy',
                             mode='w+', dtype=np.float32, shape=(Yraw.shape[0], Yraw.shape[1], Yraw.shape[2]))
t1 = cput()

nz_mask = (Yraw[:,:,0] + Yraw[:,:,1]) != 0

# Make sure that non-zero elements are copied as zeros due to mis-match
X[np.invert(nz_mask),:] = 0

print()
print('Time to load memory mapped data: '+str(t1-t0)+' (s)')


## Create Target
#======================================
t0 = cput()
target_ratio = np.zeros(nz_mask.shape)

target_ratio[nz_mask] = Yraw[nz_mask,0] / (Yraw[nz_mask,0] + Yraw[nz_mask,1])

Y = target_ratio
t1 = cput()
print()
print('Time to create targets: '+str(t1-t0)+' (s)')


## Point Normalization
#======================================
t0 = cput()
X[:,:,1:4] = dsu.to_xyz(np.ndarray.copy(Xraw[:,:,1:4]), nz_mask)
t1 = cput()

## ENERGY ##
log_E_mask = Xraw[:,:,0] > 0
X[log_E_mask,0] = np.log(np.ndarray.copy(Xraw[log_E_mask,0]))

## X ##
X[:,:,1] = np.ndarray.copy(Xraw[:,:,1])/3000

## Y ##
X[:,:,2] = np.ndarray.copy(Xraw[:,:,2])/1000

## Z ##
X[:,:,3] = np.ndarray.copy(Xraw[:,:,3])/1000
t2 = cput()

print()
print('Time to convert to xyz: {} (s)'.format(t1-t0))
print('Time to normalize: {} (s)'.format(t2-t1))
print('Total time: {} (s)'.format(t2-t0))
print('Target shape: {}'.format(Y.shape))
print('Input shape: {}'.format(X.shape))


## Set Up Datasets
#======================================
train_num, val_num, test_num = dsu.tvt_num(X, tvt=(70,15,15))
print()
print('Number of training samples: '+str(train_num))
print('Number of validation samples: '+str(val_num))
print('Number of test samples: '+str(test_num))

Y = np.atleast_3d(Y)

X_train = X[:train_num,:,:]
Y_train = Y[:train_num,:]

X_val = X[train_num:train_num+val_num,:,:]
Y_val = Y[train_num:train_num+val_num,:]

X_test = X[train_num+val_num:,:,:]
Y_test = Y[train_num+val_num:,:]

nz_test_mask = np.ndarray.copy(nz_mask[train_num+val_num:,:])
print(Y_train.shape)
print(X_train.shape)


## Compile Model
#======================================
pnet_delta = PointNet_delta(shape=(X.shape[1], 4), name='PointNet_Delta')

pnet_delta.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
pnet_delta.summary()
print()
print()


## Train Model
#======================================
t0 = cput()
delta_hist = pnet_delta.fit(x=X_train, y=Y_train,
                           epochs=30,
                           batch_size=300,
                           validation_data=(X_val, Y_val),
                           verbose=1)
t1 = cput()

print('Time to train: {} (s)'.format(t1-t0))
print('{} (min)'.format((t1-t0)/60))
print('{} (hour)'.format((t1-t0)/3600))


## Loss Curves
#======================================
plot_dict = delta_hist

fig = plt.figure(figsize=(8,6))
plt.plot(plot_dict.history['val_loss'], label="Validation")
plt.plot(plot_dict.history['loss'], label="Training")
plt.ylim(0.008,0.02)
plt.yticks(fontsize=13)
plt.xlim(0,len(plot_dict.history['loss'])-1)
plt.xticks(fontsize=13)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right', ncol=1, frameon=True, fancybox=True)
plt.text(21.5, 1.75e-2, 'PointNet Delta', fontsize=13)
plt.text(21.5, 1.7e-2, 'LR=1e-3', fontsize=13)
plt.text(21.5, 1.65e-2, 'Batch: 300', fontsize=13)
plt.text(21.5, 1.6e-2, 'Train: 2.5e5', fontsize=13)
plt.savefig('Plots/January22/LossCurves_pnetDelta_LR1e-3_batch300__Train2.5e5_2022-01-27.png',
        format='png')

## Predictions
#======================================
t0 = cput()
predictions_delta = pnet_delta.predict(X_test)
t1 = cput()
print(predictions_delta.shape)
print()
print('Time to make predictions: {} (s)'.format(t1-t0))


## Plot Preds
#======================================
predictions_delta = predictions_delta.reshape(predictions_delta.shape[0],
                                              predictions_delta.shape[1])
Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])

plt.cla(); plt.clf()
fig = plt.figure(figsize=(10,6))

EMbins = np.linspace(0,1,50, endpoint=True)
plt.hist(predictions_delta[nz_test_mask], color='indianred', bins=EMbins, density=True,
        alpha=.35, edgecolor='black', label='Predictions')
plt.hist(Y_test[nz_test_mask], color='goldenrod', bins=EMbins, density=True,
        alpha=.35, edgecolor='black', label='Truth')
plt.title('NonEM Versus EM Per Cell', fontsize=16)
plt.xlabel('EM Fraction', fontsize=14)
plt.xlim(0,1)
plt.ylim(0,5)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc='upper left', frameon=True, fancybox=True, prop={'size': 13})
plt.tight_layout()
plt.savefig('Plots/January22/predsHist_pnetDelta_LR1e-3_batch300__Train2.5e5_2022-01-27.png',
        format='png')

print()
print('Finished')
print()


