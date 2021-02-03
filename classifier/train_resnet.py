
#import libraries and some constants

import os, sys, pickle
import numpy as np
import pandas as pd
import uproot as ur

path_prefix = os.getcwd() + '/../'
# importing custom utilities
if(path_prefix not in sys.path): sys.path.append(path_prefix)
from util import resolution_util as ru
from util import plot_util as pu
from util import ml_util as mu
from models import resnet

# tensorflow and keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable some of the tensorflow info printouts, only display errors
import tensorflow as tf
from keras.utils import np_utils

# Paths
plotpath = path_prefix+'classifier/Plots/'
modelpath = path_prefix+'classifier/Models/'

# metadata
layers = ["EMB1", "EMB2", "EMB3", "TileBar0", "TileBar1", "TileBar2"]
cell_size_phi = [0.098, 0.0245, 0.0245, 0.1, 0.1, 0.1]
cell_size_eta = [0.0031, 0.025, 0.05, 0.1, 0.1, 0.2]
len_phi = [4, 16, 16, 4, 4, 4]
len_eta = [128, 16, 8, 4, 4, 2]
cell_shapes = {layers[i]:(len_eta[i],len_phi[i]) for i in range(len(layers))}


# Get the data.
inputpath = path_prefix+'data/pion/'
rootfiles = ["pi0", "piplus", "piminus"]
branches = ['runNumber', 'eventNumber', 'truthE', 'truthPt', 'truthEta', 'truthPhi', 'clusterIndex', 'nCluster', 'clusterE', 'clusterECalib', 'clusterPt', 'clusterEta', 'clusterPhi', 'cluster_nCells', 'cluster_sumCellE', 'cluster_ENG_CALIB_TOT', 'cluster_ENG_CALIB_OUT_T', 'cluster_ENG_CALIB_DEAD_TOT', 'cluster_EM_PROBABILITY', 'cluster_HAD_WEIGHT', 'cluster_OOC_WEIGHT', 'cluster_DM_WEIGHT', 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 'cluster_cell_dR_min', 'cluster_cell_dR_max', 'cluster_cell_dEta_min', 'cluster_cell_dEta_max', 'cluster_cell_dPhi_min', 'cluster_cell_dPhi_max', 'cluster_cell_centerCellEta', 'cluster_cell_centerCellPhi', 'cluster_cell_centerCellLayer', 'cluster_cellE_norm']

trees = {
    rfile : ur.open(inputpath+rfile+".root")['ClusterTree']
    for rfile in rootfiles
}
pdata = {
    ifile : itree.pandas.df(branches, flatten=False)
    for ifile, itree in trees.items()
}

# Selecting events -- making sure we have as much signal as background.

n_indices = {}
n_max = int(np.min(np.array([len(pdata[key]) for key in trees.keys()])))
rng = np.random.default_rng()

# If we have a piminus key, assume the dataset are piplus, piminus, pi0
if('piminus' in trees.keys()):
    n_indices['piplus']  = int(np.ceil((n_max / 2)))
    n_indices['piminus'] = int(np.floor((n_max / 2)))
    n_indices['pi0']     = n_max
    
# Otherwise, assume we already have piplus (or piplus + piminus) and pi0, no merging needed
else: n_indices = {key:n_max for key in trees.keys}
indices = {key:rng.choice(len(pdata[key]), n_indices[key], replace=False) for key in trees.keys()}

# Make a boolean array version of our indices, since pandas is weird and doesn't handle non-bool indices?
bool_indices = {}
for key in pdata.keys():
    bool_indices[key] = np.full(len(pdata[key]), False)
    bool_indices[key][indices[key]] = True

# Apply the (bool) indices to pdata
for key in trees.keys():
    pdata[key] = pdata[key][bool_indices[key]]

# prepare pcells -- immediately apply our selected indices
pcells = {
    ifile : {
        layer : mu.setupCells(itree, layer, indices = indices[ifile])
        for layer in layers
    }
    for ifile, itree in trees.items()
}

# Now with the data extracted from the trees into pcells, we merge pdata and pcells as needed.
# Note the order in which we concatenate things: piplus -> piplus + piminus.
if('piminus' in trees.keys()):
    
    # merge pdata
    pdata['piplus'] = pdata['piplus'].append(pdata['piminus'])
    del pdata['piminus']
    
    # merge contents of pcells
    for layer in layers:
        pcells['piplus'][layer] = np.row_stack((pcells['piplus'][layer],pcells['piminus'][layer]))
    del pcells['piminus']
    
    
    
# Now split things into training/validation/testing data.
training_dataset = ['pi0','piplus']

# create train/validation/test subsets containing 70%/10%/20%
# of events from each type of pion event
for p_index, plabel in enumerate(training_dataset):
    mu.splitFrameTVT(pdata[plabel],trainfrac=0.7)
    pdata[plabel]['label'] = p_index

# merge signal and background now
pdata_merged = pd.concat([pdata[ptype] for ptype in training_dataset])
pcells_merged = {
    layer : np.concatenate([pcells[ptype][layer]
                            for ptype in training_dataset])
    for layer in layers
}
plabels = np_utils.to_categorical(pdata_merged['label'],len(training_dataset))


# Tensorflow setup.
ngpu = 1
gpu_list = ["/gpu:"+str(i) for i in range(ngpu)]
strategy = tf.distribute.MirroredStrategy(devices=gpu_list)
ngpu = strategy.num_replicas_in_sync
print ('Number of devices: {}'.format(ngpu))


models = {}
model_history = {}
model_scores = {}
model_performance = {}


# Prepare the ResNet model.
tf.keras.backend.set_image_data_format('channels_last')
lr = 5e-5
input_shape = (128,16)
model_resnet = resnet(strategy, lr=lr)(input_shape)


# Minor extra data prep -- key names match those defined within resnet model in models.py!
pcells_merged_unflattened = {'input' + str(i):pcells_merged[key].reshape(tuple([-1] + list(cell_shapes[key]))) for i,key in enumerate(pcells_merged.keys())}
rn_train = {key:val[pdata_merged.train] for key,val in pcells_merged_unflattened.items()}
rn_valid = {key:val[pdata_merged.val] for key,val in pcells_merged_unflattened.items()}
rn_test = {key:val[pdata_merged.test] for key,val in pcells_merged_unflattened.items()}

nepochs = 10
batch_size = 20 * ngpu
verbose = 1 # 2 for a lot of printouts

model_key = 'resnet'
models[model_key] = model_resnet

# train+validate model
model_history[model_key] = models[model_key].fit(
    x=rn_train,
    y=plabels[pdata_merged.train],
    validation_data=(
        rn_valid,
        plabels[pdata_merged.val]
    ),
    epochs=nepochs,
    batch_size=batch_size,
    verbose=verbose
)
    
model_history[model_key] = model_history[model_key].history
    
# get overall performance metric
model_performance[model_key] = models[model_key].evaluate(
    x=rn_test,
    y=plabels[pdata_merged.test],
    verbose=0
)
    
# get network scores for the dataset
model_scores[model_key] = models[model_key].predict(
    pcells_merged_unflattened
)

rn_dir = modelpath + 'resnet' # directory for saving ResNet
try: os.makedirs(rn_dir)
except: pass

models[model_key].save(rn_dir + '/' + 'resnet.h5')

with open(rn_dir + '/' + 'resnet.history','wb') as model_history_file:
    pickle.dump(model_history[model_key], model_history_file)
    
print('Done.')