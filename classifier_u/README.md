## Overview

This directory contains some notebooks and code for unsupervised training of topo-cluster classifiers (charged versus neutral pions). The training and evaluation is handled in `TopoClusterClassifierUnsupervised.ipynb`, with a custom network layer defined in `cluster_layer.py`. The code and the approach is largely based on [this tutorial](https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/).

The current strategy involves training a set of autoencoders -- one per calorimeter layer. We then use the *encoders* from each of these to compress images down to a lower-dimensional latent space, and perform k-means clustering in that space (looking for some predetermined number of categories). Thus, we develop a classifier based on each calorimeter layer.

The outputs of these classifiers are, in turn, combined for use as inputs to one final classifier -- which again performs k-means clustering (into the final number of categories that we've predetermined).

The `Plots` directory contains some example images from performing this unsupervised classification.

Note that the `Models` directory does *not* currently have the dataset-dependent subdirectory structure that is present in `classifier/Models` and `regression/Models`, as this has (so far) only been trained on a single dataset (`user.angerami.mc16_13TeV.361021.Pythia8EvtGen_JZ1W.e3569_s3170_r10788.images_v8_01-20-g6323cb4_OutputStream` or `user.angerami.mc16_13TeV.361022.Pythia8EvtGen_JZ2W.e3668_s3170_r10788.images_v8_01-20-g6323cb4_OutputStream`, whichever contained `EventTree` information.)