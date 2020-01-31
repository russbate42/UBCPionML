# Let's define some utility functions we'll want to be using for resolutions

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats


def responsePlot(x, y, name='', statistic='median'):
    xbin = [10**exp for exp in np.arange(-1.0, 3.1, 0.1)]
    ybin = np.arange(0., 3.1, 0.1)
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]
    profileXMed = stats.binned_statistic(
        x, y, bins=xbin, statistic=statistic).statistic

    plt.cla()
    plt.clf()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.hist2d(x, y, bins=[xbin, ybin], norm=LogNorm())
    plt.plot([0.1, 1000], [1, 1], linestyle='--')
    plt.plot(xcenter, profileXMed)
    plt.xscale('log')
    plt.ylim(0, 3)
    plt.xlabel('Cluster Calib Hits')
    plt.ylabel('Cluster Energy over Calib Hits')
    plt.colorbar()
    # plt.legend()
    if name != '':
        plt.savefig(name+'.pdf')
    plt.show()

    return xcenter, profileXMed


def stdOverMean(x):
    std  = np.std(x)
    mean = np.mean(x)
    return std / mean

def iqrOverMed(x):
    # get the IQR via the percentile function
    # 84 is median + 1 sigma, 16 is median - 1 sigma
    q68, q16 = np.percentile(x, [84, 16])
    iqr = q68 - q16
    med = np.median(x)
    return iqr / med

def resolutionPlot(x, y, name='', statistic='std'):
    xbin = [10**exp for exp in  np.arange(-1.0, 3.1, 0.1)]
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]
    if statistic == 'std': # or any other baseline one?
        resolution = stats.binned_statistic(x, y, bins=xbin,statistic=statistic).statistic
    elif statistic == 'stdOverMean':
        resolution = stats.binned_statistic(x, y, bins=xbin,statistic=stdOverMean).statistic
    elif statistic == 'iqrOverMed':
        resolution = stats.binned_statistic(x, y, bins=xbin,statistic=iqrOverMed).statistic

    plt.cla(); plt.clf()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.plot(xcenter, resolution)
    plt.xscale('log')
    plt.ylim(0,2)
    plt.xlabel('Cluster Calib Hits')
    plt.ylabel('Cluster Energy RMS over Mean')
    if name != '':
        plt.savefig(name+'.pdf')
    plt.show()

    return xcenter, resolution
