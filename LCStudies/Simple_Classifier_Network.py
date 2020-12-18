
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot as ur
import atlas_mpl_style as ampl
ampl.use_atlas_style()
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time as t
import project_variables
import pickle
from contextlib import redirect_stdout

path_prefix = project_variables.path_prefix
plotpath = path_prefix+'classifier/Plots/'
modelpath = path_prefix+'classifier/Models/'

import sys
sys.path.append(path_prefix)
sys.path
from util import ml_util as mu

# Model specific imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.utils import np_utils

t_total0 = t.time()
## !! DECLARE GPU HERE

############################################
## ORGANIZE GPU STRATEGY USING TENSORFLOW ##
############################################
# figure out how to get this to stop dumping in the terminal
time_0 = t.time()
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4","/gpu:5","/gpu:6"])
ngpu = strategy.num_replicas_in_sync
time_1 = t.time()
print()
print('Number of devices: {}'.format(ngpu))
print('Time to distribute: '+str(time_1-time_0)); print()

# metadata
layers = project_variables.layers
cell_size_phi = project_variables.cell_size_phi
cell_size_eta = project_variables.cell_size_eta
len_phi = project_variables.len_phi
len_eta = project_variables.len_eta
cell_shapes = project_variables.cell_shapes

######################
## UNPACK DATA SETS ##
######################
print('..loading data from pickle files..')
def load_obj(name):
    with open('Data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

t_load0 = t.time()
p0 = load_obj('pi0')
pp = load_obj('piplus')
pm = load_obj('piminus')
pcells = load_obj('pcells')
pdata = load_obj('pdata')
t_load1 = t.time()
print(str(t_load1 - t_load0)+' s'); print()

##################################
## CREATE SIMPLE BASELINE MODEL ##
##################################

# define baseline fully-connected NN model
tmodel0 = t.time()
def baseline_nn_model(number_pixels, learning_rate=5e-5):
    # create model
    with strategy.scope():    
        model = Sequential()
        used_pixels = number_pixels
        model.add(Dense(number_pixels, input_dim=number_pixels,                   kernel_initializer='normal',                                   activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(used_pixels, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(int(used_pixels/2), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, kernel_initializer='normal',                           activation='softmax'))
        # compile model
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',                             optimizer=optimizer, metrics=['acc'])
    return model
tmodel1 = t.time()
print('..finished defining baseline model..')
print(str(tmodel1-tmodel0)+' s'); print()


#################
## TVT NETWORK ##
#################
training_dataset = ['pi0','piplus']

# create train/validation/test subsets containing 70%/10%/20%
# of events from each type of pion event
for p_index, plabel in enumerate(training_dataset):
    mu.splitFrameTVT(pdata[plabel],trainfrac=0.7)
    pdata[plabel]['label'] = p_index

# merge pi0 and pi+ events
pdata_merged = pd.concat([pdata[ptype] for ptype in training_dataset])
pcells_merged = {
    layer : np.concatenate([pcells[ptype][layer]
                            for ptype in training_dataset])
    for layer in layers
}

plabels = np_utils.to_categorical(pdata_merged['label'],len(training_dataset))


#################
## BUILD MODEL ##
#################
models = {}
tdefmodels0 = t.time()
for layer in layers:
    npix = cell_shapes[layer][0]*cell_shapes[layer][1]
    models[layer] = baseline_nn_model(npix)
    # Print the model information to a summary file
    out_str = 'Models/'+layer+\
        '_baseline_model_summary.txt'
    with open(out_str, 'w') as f:
        with redirect_stdout(f):
            models[layer].summary()
tdefmodels1 = t.time()
print('..finished creating models for each layer..')
print(str(tdefmodels1-tdefmodels0)+' s'); print()

####################
## RUN MODEL EMB1 ##
####################
# current_model = models['EMB1']
# pcells_merged_current = {
#     layer : np.concatenate([pcells[ptype]['EMB1']
#                             for ptype in training_dataset])
# }

# print('..training and validating model EMB1..'); print()

# # unsure about model history..
# current_model_history = current_model.fit(
#     pcells_merged_current[pdata_merged.train],
#     plabels[pdata_merged.train],
#     validation_data=(
#         pcells_merged_current[pdata_merged.val],
#         plabels[pdata_merged.val]
#     ),
#     epochs=10, batch_size=50*ngpu, verbose=2
# )

# model_history = current_model_history.history

# # get overall performance metric
# current_model_performance = current_model.evaluate(
#     pcells_merged_current[pdata_merged.test],
#     plabels[pdata_merged.test]
# )
    
# # get network scores for the dataset
# current_model_scores = current_model.predict(
#     pcells_merged_current
# )
    
#############################
## RUN MODEL ON ALL LAYERS ##
#############################
model_history = {}
model_performance = {}
model_scores = {}
print()

for layer in layers:
    print('On layer: ' + layer);print()
    
    # train+validate model
    model_history[layer] = models[layer].fit(
        pcells_merged[layer][pdata_merged.train], plabels[pdata_merged.train],
        validation_data = (
            pcells_merged[layer][pdata_merged.val], plabels[pdata_merged.val]
        ),
        epochs = 100, batch_size = 100*ngpu, verbose = 1,
    )
    model_history[layer] = model_history[layer].history
    # get overall performance metric
    model_performance[layer] = models[layer].evaluate(
        pcells_merged[layer][pdata_merged.test], plabels[pdata_merged.test],
        verbose = 0,
    )
    
    # get network scores for the dataset
    model_scores[layer] = models[layer].predict(
        pcells_merged[layer]
    )
    
    print('Finished layer: ' + layer); print()

t_total1 = t.time()
print()
print('Total time using 2 GPUs, batch size 100*ngpu, 100 epochs:')
print(str(t_total1 - t_total0)+' s')

