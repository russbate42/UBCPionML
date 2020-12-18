###############################################
## SIMPLE MODEL HYPER PARAMETER OPTIMIZATION ##
###############################################

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot as ur
import atlas_mpl_style as ampl
ampl.use_atlas_style()
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import time as t
import project_variables
import pickle
from contextlib import redirect_stdout
from scipy.stats import truncnorm

path_prefix = project_variables.path_prefix
plotpath = path_prefix+'classifier/Plots/'
modelpath = path_prefix+'classifier/Models/'
datapath = project_variables.datapath

import sys
sys.path.append(path_prefix)
sys.path

from util import ml_util as mu

import network_models as ntm

# metadata
layers = project_variables.layers
cell_size_phi = project_variables.cell_size_phi
cell_size_eta = project_variables.cell_size_eta
len_phi = project_variables.len_phi
len_eta = project_variables.len_eta
cell_shapes = project_variables.cell_shapes

########################################
## Distribute GPUS and Model Building ##
########################################

time_0 = t.time()
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1","/gpu:3"])
ngpu = strategy.num_replicas_in_sync
time_1 = t.time()
print()
print('Number of devices: {}'.format(ngpu))
print('Time to distribute: '+str(time_1-time_0)+' s')
print()

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

###############################
## Create t/v/test data sets ##
###############################
''' Can we do this in another module and pickle the files
in order to speed up the process? '''
print('..creating training, validation, and test sets..')
t_tvt0 = t.time()
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
t_tvt1 = t.time()
print(str(t_tvt1-t_tvt0)+' s');print()


#################################################
## MC COMPONENT, BUILD MULTIPLE MODELS AND RUN ##
#################################################
''' Here we make guesses of the learning rate and dropout
rate based on previous models to define our schedule. This
could possibly be modularized? Std dropout rate is .2,
learning rate is 5e-5 so we begin with these guesses'''
print('..running annealing..');print()
N = 10 #number of temperature steps
M = 4 #number of sweeps per temperature
# set temperature schedule relative to the cost function that
# we are dealing with. Here it is delta ROC curve
T_i = .005 # try .01? was .05

def Tf_solver(cost,prob):
    ''' cost is the change in expected outcome,
    prob is the probability of accepting anyways '''
    return -cost/np.log(prob)

# solve for the rate for temp schedule
def rate_solve(Tf, Ti, ns):
    ''' ns is the number of temperature steps
    cost is the change in the function in the exponent (ROC)'''
    return np.exp((np.log(Tf)-np.log(Ti))/(ns-1))

def geometric_temp(Ti, rate, ns):
    a = np.arange(N)
    return Ti*rate**a

def new_rate(old_val, sigma):
	''' need to modify bounds because truncnorm is dumb '''
	if old_val > .5:
		lower = 2*old_val - 1
		upper = 1
		A = (lower-old_val)/sigma
		B = (upper-old_val)/sigma
		new_r = truncnorm.rvs(a=A, b=B, loc=old_val,
					scale=sigma)
	else:
		A = -old_val/sigma
		B = old_val/sigma
		new_r = truncnorm.rvs(a=A, b=B, loc=old_val,
					scale=sigma)
	return new_r

def layer_size(current_size, scale_factor):
    ''' designed to never sample lower than the size,
    and is symmetric about this number. A,B are cuts '''
    sigma = current_size*scale_factor
    A = (-num_layer - mean)/sigma
    B = (num_layer - mean)/sigma
    ## Truncnorm distribution test
    delta = truncnorm.rvs(a=A, b=B, loc=mean,
            scale=sigma, size=1)
    # round to nearest integer
    delta = np.rint(delta)

    return current_size + delta

## Solve for the final temperature
T_f = Tf_solver(cost=.001,prob=1./100.)
## Solving for the geometric rate
g_rate = rate_solve(Tf=T_f, Ti=T_i, ns=N)

## Set the Temperature schedule
T = geometric_temp(Ti=T_i, rate=g_rate, ns=N)
print('..temp schedule..');print(T)
print()

## Set the standard deviation for our parameters
sigma_lr = np.linspace(5e-5,1e-5,num=N)
print('..sigma learning rate..');print(sigma_lr)
print()

sigma_dr = np.linspace(5e-2,5e-3,num=N)
print('..sigma dropout rate..');print(sigma_dr)
print()

# create dictionaries of metrics
models = {}
model_history = {}
model_performance = {}
model_scores = {}
roc_fpr = {}
roc_tpr = {}
roc_thresh = {}
roc_auc = {}

## TRAIN VALIDATE TEST for each layer
for layer in layers:
    print()
    print('..working on '+layer+' ..');print()
    data_str = datapath+'Simple_MC_Optimization/'+layer+'.txt'
    tf = open(data_str, 'w')
    tf.write('#############################\n')
    tf.write('## '+layer+' ROC CURVE ######\n')
    tf.write('#############################\n')
    tf.write('## 100 epochs, 300 batch size, 3 GPU\n')
    tf.write('## Random seed: 648967\n')
    tf.write('## Learning  Dropout  Area_Under_ROC\n\n')
    
    # Initial values re-initialized for every layer!
    np.random.seed(648967)
    new_lr, old_lr = 5e-5, 0
    new_dr, old_dr = .2, 0
    old_auc = 0
    
    for i in range(N):      
        for j in range(M):
                     
            ## NEW MODEL
            npix = cell_shapes[layer][0]*cell_shapes[layer][1]
            # import model from network_models
            current_model = ntm.baseline_nn_model(strategy,
                            number_pixels=npix,
                            l_rate=new_lr,
                            dropout_rate=new_dr)           
            models[layer] = current_model

            ## TRAIN + VALIDATE MODEL
            model_history[layer] = models[layer].fit(
                pcells_merged[layer][pdata_merged.train],
                plabels[pdata_merged.train],

                validation_data = (
                    pcells_merged[layer][pdata_merged.val],
                    plabels[pdata_merged.val]),
                
                epochs = 100, batch_size = 100*ngpu,
                verbose = 0,
            )

            # HISTORY: save dict of model history
            model_history[layer] = model_history[layer].history
            # PERFORMANCE: overall network performance
            model_performance[layer] = models[layer].evaluate(
                pcells_merged[layer][pdata_merged.test],
                plabels[pdata_merged.test],
                verbose = 0)   
            # SCORES: get overall network score
            model_scores[layer] = models[layer].predict(
                pcells_merged[layer])

            # METRICS (ROC)
            roc_fpr[layer], roc_tpr[layer], roc_thresh[layer] =\
            roc_curve(
                plabels[pdata_merged.test][:,1],
                model_scores[layer][pdata_merged.test,1],
                drop_intermediate=False)
            roc_auc[layer] = auc(roc_fpr[layer], roc_tpr[layer])

            current_auc = roc_auc[layer]
            
            # determine step from last best selection
            delta_LR = new_lr - old_lr
            delta_DR = new_dr - old_dr
            # change in area under curve
            delta_roc = current_auc - old_auc
            
            # Write information to text file
            ''' this will save steps even if they are bad '''
            file_str = str(new_lr) + '  ' + str(new_dr) + '  '
            file_str = file_str + str(current_auc)
            tf.write(file_str)
            
            ## ANNEALING ##
            ''' note here old_auc also is kept/used as the 'best' '''
            # If there is an improvement, change rates
            if delta_roc >= 0:
                old_lr = new_lr
                old_dr = new_dr
                old_auc = current_auc
                tf.write('  accept\n')
            
            # Otherwise, some probability of changing anyways
            else:
                rand_u = np.random.rand()
                # uses pre-defined temperature function
                cost = np.exp(-np.abs(delta_roc)/T[i])
                ## main condition test
                if rand_u <= cost:
                    old_lr = new_lr
                    old_dr = new_dr
                    old_auc = current_auc
                    tf.write('  acceptc\n')
                else:
                    tf.write('  reject\n')
            
            ## Choose new learning rates based on scheudle
            # and the previous (last sweep) rates
            new_lr = new_rate(old_val=old_lr, sigma=sigma_lr[i])
            new_dr = new_rate(old_val=old_dr, sigma=sigma_dr[i])

    tf.close()
