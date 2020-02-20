import matplotlib.font_manager
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import atlas_mpl_style as ampl
ampl.use_atlas_style()

# set plotsytle choices here
params = {'legend.fontsize': 13,
          'axes.labelsize': 18}
plt.rcParams.update(params)

ampl.set_color_cycle('Oceanic',8)

def histogramOverlay(frames, data, labels, xlabel, ylabel, figfile = '', 
                        x_min = 0, x_max = 2200, xbins = 22, normed = True, y_log = False,
                        atlas_x = -1, atlas_y = -1, simulation = False,
                        textlist = []):
    xbin = np.arange(x_min, x_max, (x_max - x_min) / xbins)

    plt.cla()
    plt.clf()
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    zorder_start = -1 * len(data) # hack to get axes on top
    for i, datum in enumerate(data):
        plt.hist(frames[i][datum], bins = xbin, normed = normed, 
            alpha = 0.5, label=labels[i], zorder=zorder_start + i)
    
    plt.xlim(x_min, x_max)
    if y_log:
        plt.yscale('log')

    ampl.set_xlabel(xlabel)
    ampl.set_ylabel(ylabel)

    if atlas_x >= 0 and atlas_y >= 0:
        ampl.draw_atlas_label(atlas_x, atlas_y, simulation = simulation, fontsize = 18)

    drawLabels(fig, atlas_x, atlas_y, simulation, textlist)
    
    fig.axes[0].zorder = len(data)+1 #hack to keep the tick marks up
    plt.legend()
    if figfile != '':
        plt.savefig(figfile)
    plt.show()

def lineOverlay(xcenter, lines, labels, xlabel, ylabel, figfile = '',
                    x_min = 0.1, x_max = 1000, x_log = True, y_min = 0, y_max = 2, y_log = False,
                    linestyles=[], colorgrouping=-1,
                    extra_lines = [],
                    atlas_x=-1, atlas_y=-1, simulation=False,
                    textlist=[]):
    plt.cla()
    plt.clf()

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for extra_line in extra_lines:
        plt.plot(extra_line[0], extra_line[1], linestyle='--', color='black')

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, line in enumerate(lines):
        if len(linestyles) > 0:
            linestyle = linestyles[i]
        else:
            linestyle = 'solid'
        if colorgrouping > 0:
            color = colors[int(np.floor(i / colorgrouping))]
        else:
            color = colors[i]
        plt.plot(xcenter, line, label = labels[i], linestyle=linestyle,color=color)

    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ampl.set_xlabel(xlabel)
    ampl.set_ylabel(ylabel)

    drawLabels(fig, atlas_x, atlas_y, simulation, textlist)

    plt.legend()
    if figfile != '':
        plt.savefig(figfile)
    plt.show()


def drawLabels(fig, atlas_x=-1, atlas_y=-1, simulation=False,
               textlist=[]):
    if atlas_x >= 0 and atlas_y >= 0:
        ampl.draw_atlas_label(atlas_x, atlas_y, simulation=simulation, fontsize=18)

    for textdict in textlist:
        fig.axes[0].text(
            textdict['x'], textdict['y'], textdict['text'], 
            transform=fig.axes[0].transAxes, fontsize=18)
