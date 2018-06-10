# caloml-atlas

Machine Learning toolkit for calorimeter topo-cluster classification and regression using simulated data from the ATLAS experiment. 

Feel free to contact joakim.olsson[at]cern.ch if you'd like to contribute! 

Machine Learning is awesome :)  

## Image pre-processing

Images are created from ESD (Event Summary Data) files using the [MLTree](https://github.com/jmrolsson/MLTree) Athena package, which generates a root TTree that contains the images as well as some other info. Six images are saved for each cluster, corresponding to the barrels layers of the EM (EMB1, EMB2, EMB3) and HAD calorimeters (TileBar0, TileBar2, TileBar3). Normalized cell energies are used as pixel values. The image size is 0.4x0.4 in eta-phi space. 

The outputs from [MLTree](https://github.com/jmrolsson/MLTree) can be converted into numpy arrays with [mltree2array.py](util/mltree2array.py)

## Topo-cluster classification

The performance of the following Machine Learning techniques are compared:
- Simple fully-connected Neural Network (flattening the images and only using the 512 pixels in the EMB1 layer).
- Using all 2D images with a deeper Convolutional Neural Network.

Everything is in the [TopoClusterClassifier.ipynb](classifier/TopoClusterClassifier.ipynb) notebook, so it is easy to modify and play around with! 

Also want to compare with other ML methods: logistic regression, SVD, Naive Bias, Gaussians, etc. 

## Energy regression

Coming soon...
