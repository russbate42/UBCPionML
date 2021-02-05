import numpy as np
import ROOT as rt
import uproot as ur
import uuid

def EnergyRatioHist(ur_trees, charged_key, neutral_key, classification_threshold = 0.6, nbins = 10000, xmin = 1.0e-3, xmax = 1.0e2, color=rt.kViolet-6):
    energy_ratio_hist = rt.TH1F(str(uuid.uuid4()), 'Predicted Energy / E_{CALIB}^{TOT};E_{pred} / E_{CALIB}^{TOT};Count',nbins,xmin,xmax)
    for dfile, trees in ur_trees.items():
        scores = trees['score'].array('charged_likelihood_combo')
        charged_e = trees['score'].array(charged_key)
        neutral_e = trees['score'].array(neutral_key)
        true_e = trees['cluster'].array('cluster_ENG_CALIB_TOT')
    
        for i in range(len(scores)):
            if(true_e[i] == 0.): continue
            if(scores[i] > classification_threshold): energy_ratio_hist.Fill(charged_e[i] / true_e[i])
            else: energy_ratio_hist.Fill(neutral_e[i] / true_e[i])

    energy_ratio_hist.SetFillColorAlpha(color,1.)
    energy_ratio_hist.SetLineColorAlpha(color,1.)
    
    return energy_ratio_hist

def EnergyRatioHist2D(ur_trees, charged_key, neutral_key, class_min = 0., class_max = 1., nsteps = 20, nbins = 10000, xmin = 1.0e-3, xmax = 1.0e2):
    # to use our existing infrastructure, we start off by making a bunch of 1D histograms, that correspond with "rows" in our 2D histogram
    classifier_thresholds = np.linspace(class_min, class_max, nsteps)
    hists = [EnergyRatioHist(ur_trees, charged_key, neutral_key, x, nbins, xmin, xmax) for x in classifier_thresholds]
    # now construct a 2D histogram, and fill it from the 1D histograms
    title = 'Classifier Threshold vs. Predicted Energy / E_{CALIB}^{TOT};E_{pred} / E_{CALIB}^{TOT};Classifier Threshold;Count'
    h = rt.TH2F(str(uuid.uuid4()), title, nbins, xmin, xmax, nsteps, class_min, class_max)
    
    for i in range(nsteps):
        [h.SetBinContent(j+1, i+1, hists[i].GetBinContent(j+1)) for j in range(nbins)] # note the 1-indexing for histogram objects
    return h