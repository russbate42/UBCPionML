
import numpy as np
import pandas as pd
import project_variables as prv
import uproot as ur
import pickle
import sys
import time as t

path_prefix = '/home/russbate/start_tf/LCStudies'
sys.path.append(path_prefix)
sys.path
from util import ml_util as mu

# import vars from project_variables
rootfiles = prv.rootfiles
inputpath = prv.inputv10
datapath = prv.datapath

print('..extracting root files..');print()
t0 = t.time()
trees = {
	rfile : ur.open(inputpath+rfile+'.root')['ClusterTree']
	for rfile in rootfiles
}
t1 = t.time()
print('..finishing extracting root files: '+\
	str(t1-t0)+' (s)..')

with open(datapath+'v10_data.txt', 'w') as outfile:
	sys.stdout = outfile
	cluster = trees['pi0']
	print(cluster.show())

# file.keys()
# clusters = file[b'ClusterTree;1']
# print(clusters)
# clusters.show()
