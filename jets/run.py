# notebook as a script

skip_dR = False

# Debug: Uses only one input file, which will speed things up.
debug = True

classification_threshold = 0.6

# select our source for training -- options are 'pion' (default), 'pion_reweighted' and 'jet'
source = 'pion'




# Imports - generic stuff
import numpy as np
import pandas as pd
import ROOT as rt
import uproot as ur # uproot for accessing ROOT files quickly (and in a Pythonic way)
import sys, os, glob, uuid # glob for searching for files, uuid for random strings to name ROOT objects and avoid collisions
import subprocess as sub
from numba import jit
from pathlib import Path
from IPython.utils import io # For suppressing some print statements from functions.

path_prefix = os.getcwd() + '/../'
if(path_prefix not in sys.path): sys.path.append(path_prefix)
from util import ml_util as mu # for passing calo images to regression networks
from util import qol_util as qu # for progress bar
from util import jet_util as ju




# To display our plots, let's get a dark style that will look nice in presentations (and JupyterLab in dark mode).
dark_style = qu.PlotStyle('dark')
light_style = qu.PlotStyle('light')
plot_style = dark_style
plot_style.SetStyle() # sets style for plots - still need to adjust legends, paves




data_dir = path_prefix + 'data/jet'
training_data_dir = path_prefix + 'data/pion' # TODO: deal with situation where source is jet
fj_dir = path_prefix + '/setup/fastjet/fastjet-install/lib/python3.8/site-packages'
plot_dir = path_prefix + 'jets/Plots/' + source

try: os.makedirs(plot_dir)
except: pass




# ----- Calorimeter meta-data -----
layers = ["EMB1", "EMB2", "EMB3", "TileBar0", "TileBar1", "TileBar2"]
nlayers = len(layers)
cell_size_phi = [0.098, 0.0245, 0.0245, 0.1, 0.1, 0.1]
cell_size_eta = [0.0031, 0.025, 0.05, 0.1, 0.1, 0.2]
len_phi = [4, 16, 16, 4, 4, 4]
len_eta = [128, 16, 8, 4, 4, 2]
assert(len(len_phi) == nlayers)
assert(len(len_eta) == nlayers)
meta_data = {
    layers[i]:{
        'cell_size':(cell_size_eta[i],cell_size_phi[i]),
        'dimensions':(len_eta[i],len_phi[i])
    }
    for i in range(nlayers)
}





# our "local" data dir
jet_data_dir = path_prefix + 'jets/data'
data_filenames = glob.glob(jet_data_dir + '/*.root')
if(debug): data_filenames = [data_filenames[0]]
    
    
    
    
    
# Access the files & trees with uproot
tree_names = {'cluster':'ClusterTree','event':'EventTree','score':'ScoreTree'}
ur_trees = {file:{tree_key:ur.open(file)[tree_name] for tree_key,tree_name in tree_names.items()} for file in data_filenames}



global_eta_cut = 0.3 # eta cut to be applied to all jets -- those we make and those we're given
global_truth_e_cut = 25. # GeV -- recall that jet energies are stored in MeV!

# pavetext with info on cuts
cut_info = [
    '|#eta_{j}| <' + ' {val:.1f},'.format(val=global_eta_cut),
    'E_{j}^{true}' + ' > {val:.0f} [GeV],'.format(val=global_truth_e_cut),
    'All reco jets matched',
    'to truth w/ #Delta R < 0.3 .'
]

cut_pave = rt.TPaveText(0.675, 0.5, 0.875, 0.7, 'NDC')
cut_pave.SetFillColorAlpha(plot_style.canv,0.1)
cut_pave.SetBorderSize(0)
cut_pave.SetTextColor(plot_style.text)
cut_pave.SetTextFont(42)
cut_pave.SetTextSize(0.03)
cut_pave.SetTextAlign(12)
for line in cut_info: 
    cut_pave.AddText(line)
    
    
R = 0.4
pt_min = 0.
eta_max = global_eta_cut
tree_name = 'JetTree'

ju.ClusterJets(ur_trees, 
               'AntiKt4MLTopoJets',
               R=R, 
               pt_min = pt_min, 
               eta_max = global_eta_cut, 
               fj_dir = fj_dir, 
               classification_threshold = classification_threshold,
               tree_name = 'JetTree'
)

# update our uproot tree access dictionary, adding our new tree!
#tree_names['jet'] = tree_name
#ur_trees = {file:{tree_key:ur.open(file)[tree_name] for tree_key,tree_name in tree_names.items()} for file in data_filenames}


print('Done.')