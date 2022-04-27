# Let's define some utility functions we'll want to be using for resolutions

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats

from . import plot_util as pu


def responsePlot(x, y, figfile='', statistic='median',
                 xlabel='Cluster Calib Hits', ylabel='Cluster Energy / Calib Hits',
                 atlas_x=-1, atlas_y=-1, simulation=False, cblabel='Clusters',
                 textlist=[]):
    xbin = [10**exp for exp in np.arange(-1.0, 3.1, 0.1)]
    ybin = np.arange(0., 3.1, 0.1)
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]
    profileXMed = stats.binned_statistic(
        x, y, bins=xbin, statistic=statistic).statistic

    plt.cla()
    plt.clf()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.hist2d(x, y, bins=[xbin, ybin], norm=LogNorm(),zorder = -1)
    plt.plot([0.1, 1000], [1, 1], linestyle='--', color='black')
    plt.plot(xcenter, profileXMed, color='red')
    plt.xscale('log')
    plt.ylim(0, 3)
    pu.ampl.set_xlabel(xlabel)
    pu.ampl.set_ylabel(ylabel)
    # ampl.set_zlabel('Clusters')
    cb = plt.colorbar()
    cb.ax.set_ylabel(cblabel)
    # plt.legend()

    pu.drawLabels(fig, atlas_x, atlas_y, simulation, textlist)

    if figfile != '':
        plt.savefig(figfile)
    plt.show()

    return xcenter, profileXMed


def stdOverMean(x):
    std  = np.std(x)
    mean = np.mean(x)
    return std / mean

def iqrOverMed(x):
    # get the IQR via the percentile function
    # 84 is median + 1 sigma, 16 is median - 1 sigma
    q84, q16 = np.percentile(x, [84, 16])
    iqr = q84 - q16
    med = np.median(x)
    return iqr / med

def choose_res(x, y, bins, statistic):
    # Declare statistics for plot
    if statistic == 'std': # or any other baseline one?
        resolution = stats.binned_statistic(x, y, bins=bins,
                                            statistic=statistic).statistic
    elif statistic == 'stdOverMean':
        resolution = stats.binned_statistic(x, y, bins=bins,
                                            statistic=stdOverMean).statistic
    elif statistic == 'iqrOverMed':
        resolution = stats.binned_statistic(x, y, bins=bins,
                                            statistic=iqrOverMed).statistic
    else:
        raise ValueError('Incorrect value passed to statistic argument.')
    return resolution

        
def resolutionPlot(x, y, figfile='', statistic='std',
                   xlabel='Cluster Calib Hits', ylabel='Energy IQR over Median',
                   atlas_x=-1, atlas_y=-1, simulation=False,
                   textlist=[], colors=None, labels=None,
                   leg_font_size=12):
    
    # determine ranges for scale on plot
    xbin = [10**exp for exp in  np.arange(-1.0, 3.1, 0.1)]
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]

    # start plotting
    plt.cla(); plt.clf()
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    if type(x) == list:
        if np.any(len(x) != np.array([len(y), len(colors), len(labels)])):
            raise ValueError('Args passed to resolutionPlot differ in length.')
        for xi, yi, ci, li in zip(x, y, colors, labels):
            resi = choose_res(x=xi, y=yi, bins=xbin, statistic=statistic)
            plt.plot(xcenter, resi, color=ci, label=li)
        res=None
            
    else:
        res = choose_res(x, y, bins=xbin, statistic=statistic)
        plt.plot(xcenter, res)

    plt.xscale('log')
    plt.xlim(0.1, 1000)
    plt.ylim(0,2)
    pu.ampl.set_xlabel(xlabel)
    pu.ampl.set_ylabel(ylabel)

    pu.drawLabels(fig, atlas_x, atlas_y, simulation, textlist)
    
    if labels is not None:
        plt.legend(loc='upper right', prop={'size':leg_font_size})

    if figfile != '':
        plt.savefig(figfile)
    plt.show()

    return xcenter, res
