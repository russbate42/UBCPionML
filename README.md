# caloml-atlas

Machine Learning toolkit for calorimeter topo-cluster classification and regression using simulated data from the ATLAS experiment. 

Feel free to contact joakim.olsson[at]cern.ch if you'd like to contribute! 

Machine Learning is awesome :)  

## Image pre-processing

Images are created from ESD (Event Summary Data) files using the [MLTree](https://github.com/jmrolsson/MLTree) Athena package, which generates a root TTree that contains the images as well as some other info. Six images are saved for each cluster, corresponding to the barrels layers of the EM (EMB1, EMB2, EMB3) and HAD calorimeters (TileBar0, TileBar2, TileBar3). Normalized cell energies are used as pixel values. The image size is 0.4x0.4 in eta-phi space. 

The outputs from [MLTree](https://github.com/jmrolsson/MLTree) can be converted into numpy arrays with [mltree2array.py](util/mltree2array.py)

## Training using keras

Coming soon...
