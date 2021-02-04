## Overview

This directory contains some notebooks and code for supervised training of topo-cluster classifiers (charged versus neutral pions). Some neural network layers and models are defined in `layers.py` and `models.py`, respectively, and are used in `TopoClusterClassifier_Jan.ipynb`. 

The script `train_resnet.py` is basically just a copy of this notebook, with the training code left in only for ResNet (since it may take a little time to train, and you might not want to have to keep a Jupyter notebook open that whole time). This script takes one argument, which is used as a suffix when naming the resulting `.h5` and `.history` files, to prevent overwriting previous training.

The `Models` directory contains some subdirectories, that correspond to different training data choices -- note that the Jupyter notebook workflows may need to be updated to account for these subdirectories. Within each of this subdirectories, there are further subdirectories for different models.

