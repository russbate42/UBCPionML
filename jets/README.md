## Jets

The goal here is to use our networks for topo-cluster classification and energy regression to produce "smart topo-clusters", and use these to build jets.

There's more than one possible approach to this, so as this is work is currently in its early stages, let's lay out some ideas:

### Classification

The original stated purpose of my SCGSR project is to perform classification of clusters, so that we can remove electrons from jets, and identify isolated pions. Our plan has shifted somewhat, as we're now also considering the clusters' *energy calibration*, but classification is still an important part of this process one way or another.

Before we build jets, we may want to use classification to eliminate topo-clusters that (we think) correspond to electrons. Similarly, we may want to eliminate isolated pions. To achieve this, we can train a network using [a version of Max's existing notebooks](https://github.com/janTOffermann/LCStudies/blob/master/classifier/TopoClusterClassifier_Jan.ipynb). This notebook features multiple networks, but I assume that to start, we will go with the best-performing one (and I think there is a clear winner). *Tweaking this network is something I will investigate*.

Using the classification network, we can apply a score to each topo-cluster, corresponding to its likelihood to be part of a jet (that we'd like to keep). When we get to jet-clustering, we will only consider topo-cluster whose scores are above a certain threshold. *This threshold is something I'll have to investigate, likely with a parameter scan.*

### Regression

Besides determining *which* clusters we'd like to use for jet clustering, we also want to correct their measured energies to account for calorimeter response. Once again, this is something that Max and company [have already put work into](https://cds.cern.ch/record/2724632), and [we also have a notebook](https://github.com/janTOffermann/LCStudies/blob/master/regression/TopClusterRegressionRewrite.ipynb) that trains a network to perform these corrections.

The others may already have an answer to this question, but I'm curious as to how to best employ this in tandem with classification:

#### Option A:
We train one energy regression network on a combination of our $\pi^\pm$ and $\pi^0$. We apply this network to all clusters that we classify as in our jet. For this approach, *our classification only needs to be binary*: electrons and isolated pions are both considered "background" and pions in jets are considered "signal".

#### Option B:
We train separate energy regression networks, one for $\pi^\pm$ and one for $\pi^0$. This makes sense if we expect some difference in calorimeter response between the charged and neutral pions. Again, we will apply this network to all clusters that we classify as in our jet. However, this requires not only identifying that a given cluster is part of our jet (i.e. not an electron or isolated pion), but it *also* requires identifying if it is charged or neutral. This could be accomplished by either two binary classifications in tandem, or by *a single network performing multi-label classification.*