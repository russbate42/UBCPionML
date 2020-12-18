
import pandas as pd
import uproot as ur
import project_variables
import pickle
import numpy as np
import time as t
import sys

inputpath = project_variables.inputpath
branches = project_variables.branches
rootfiles = project_variables.rootfiles
layers = project_variables.layers

path_prefix = project_variables.path_prefix
sys.path.append(path_prefix)
sys.path
from  util import ml_util as mu

print(); print('..extracting root files..'); print()
t0 = t.time()
trees = {
    rfile : ur.open(inputpath+rfile+".root")['ClusterTree']
    for rfile in rootfiles
}

pdata = {
    ifile : itree.pandas.df(branches, flatten=False)
    for ifile, itree in trees.items()
}
t1 = t.time()

print('..loaded dictionaries from root files..')
print(str(t1-t0)+' s'); print()

# set up cells
print('..setting up cells using ml_util..')
ta = t.time()
pcells = {
    ifile : {
        layer : mu.setupCells(itree, layer)
        for layer in layers
    }
    for ifile, itree in trees.items()
}
tb = t.time()
print(str(tb-ta)+' s'); print()

# separate data into respective categories
p0 = pdata['pi0']
pp = pdata['piplus']
pm = pdata['piminus']

# Print information about event files
np0 = len(p0)
npp = len(pp)
npm = len(pm)

print("Number of pi0 events: {}".format(np0))
print("Number of pi+ events: {}".format(npp))
print("Number of pi- events: {}".format(npm))
print("Total: {}".format(np0+npp+npm))

#############################
## Save files using Pickle ##
#############################
print()
print('..saving dict as pickle files..')

t2 = t.time()
with open('Data/'+ 'pi0' + '.pkl', 'wb') as f:
    pickle.dump(p0, f, pickle.HIGHEST_PROTOCOL)
with open('Data/'+ 'piplus' + '.pkl', 'wb') as f:
    pickle.dump(pp, f, pickle.HIGHEST_PROTOCOL)
with open('Data/'+ 'piminus' + '.pkl', 'wb') as f:
    pickle.dump(pm, f, pickle.HIGHEST_PROTOCOL)
with open('Data/'+ 'pcells' + '.pkl', 'wb') as f:
    pickle.dump(pcells, f, pickle.HIGHEST_PROTOCOL)
with open('Data/'+ 'pdata' + '.pkl', 'wb') as f:
    pickle.dump(pdata, f, pickle.HIGHEST_PROTOCOL)
f.close()
t3 = t.time()
print(str(t3-t2)+' s'); print()

############################
## Save Files using numpy ##
############################
# print('..saving dict as numpy files..')
# t4 = t.time()
# np.save('Data/pi0_data.npy', p0)
# np.save('Data/piplus_data.npy', pp)
# np.save('Data/piminus_data.npy', pm)
# np.save('Data/pcells.npy', pcells)
# t5 = t.time()
# print(str(t5-t4)+' s'); print()


## Pickle loader functions
# def save_obj(obj, name ):
#     with open('obj/'+ name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# def load_obj(name ):
#     with open('obj/' + name + '.pkl', 'rb') as f:
#         return pickle.load(f)
    
