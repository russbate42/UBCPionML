## Jets

The goal here is to use our networks for topo-cluster classification and energy regression to produce "smart topo-clusters", and use these to build jets.

The general strategy is to produce these topo-clusters by applying classification and energy regression in tandem (with the latter depending on the former).

In `JetClustering.ipynb`, we have a workflow that does just this -- it applies classification and regression networks to correct topo-clusters' energies, and then performs jet clustering. Note that there is a wide range of input parameters, as we must specify which classification/regression models to use, as well as the input dataset. The data must have some event-level info (in a tree called `EventTree`) since we want to consider all topo-clusters in a given event, and we would also like some reference jets against which we can compare those we cluster. This workflow makes use of `fastjet`, which can be installed using the script in `../setup/fastjet`. Don't worry if it crashes -- it currently tries to install `fj-core`, which doesn't work out but we don't need it for now.

`EnergyEvaluation.ipynb` is a shorter workflow, that just applies classification and energy regression to the topo-clusters (and compares the resulting energies to the truth energy). This is useful for gauging the performance of our networks on the given data, as there's little point in performing jet clustering until we see that our handling of the underlying topo-clusters is going smoothly. Some of the functions for this workflow are in `energy_ratio.py`.

`JetTrainingDataPrep.ipynb` is a bit of an experimental workflow, that can be used to create training data out of data that contains an `EventTree`. This is useful if the dataset does not have any labeling of topo-clusters (as is the case in practice, except for our single-pion data). The workflow attempts to label topo-clusters as resulting from charged or neutral pions, based on the Pythia8 generator-level pion to which a given topo-cluster is closest in (eta,phi). I'm not sure how valid this approach is, or if there is a more sensible way to match things.

Plots from `JetClustering.ipynb` will be saved in `Plots`, and those from `EnergyEvaluation.ipynb` will be saved in `clusterPlots`.