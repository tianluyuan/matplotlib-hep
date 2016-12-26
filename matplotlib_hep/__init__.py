from __future__ import division

import logging
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.stats as stats
import scipy as sp
import numpy as np


__all__ = ['histpoints', 'make_split', 'calc_nbins', 'plot_pull', 'centers', 'calc_bins']

def centers(x):
    return (x[:-1]+x[1:])*0.5

def edges(x):
    """ returns bin edges that correspond to centers of x
    """
    c = centers(x)
    return np.concatenate(([2*x[0]-c[0]], c, [2*x[-1]-c[-1]]))

def calc_nbins(x):
    n =  (np.max(x) - np.min(x)) / (2 * len(x)**(-1/3) * (np.percentile(x, 75) - np.percentile(x, 25)))
    return np.floor(n)

def calc_bins(x):
    nbins = calc_nbins(x)
    return np.linspace(np.min(x), np.max(x)+2, num=nbins+1)

def poisson_limits(N, kind, confidence=0.6827):
    alpha = 1 - confidence
    upper = np.zeros(len(N))
    lower = np.zeros(len(N))
    if kind == 'gamma':
        lower = stats.gamma.ppf(alpha / 2, N)
        upper = stats.gamma.ppf(1 - alpha / 2, N + 1)
    elif kind == 'sqrt':
        err = np.sqrt(N)
        lower = N - err
        upper = N + err
    else:
        raise ValueError('Unknown errorbar kind: {}'.format(kind))
    # clip lower bars
    lower[N==0] = 0
    return N - lower, upper - N

def histpoints(x, bins=None, xerr='binwidth', yerr='sqrt', density=None,
               weights=None, **kwargs):
    """
    Plot a histogram as a series of data points.

    Compute and draw the histogram of *x* using individual (x,y) points
    for the bin contents.

    By default, vertical poisson error bars are calculated using the
    gamma distribution.

    Horizontal error bars are omitted by default.
    These can be enabled using the *xerr* argument.
    Use ``xerr='binwidth'`` to draw horizontal error bars that indicate
    the width of each histogram bin.

    Parameters
    ---------

    x : (n,) array or sequence of (n,) arrays
        Input values. This takes either a single array or a sequence of
        arrays, which are not required to be of the same length.

    """
    if bins is None:
        bins = calc_nbins(x)

    h, bins = np.histogram(x, bins=bins, weights=weights)
    width = bins[1:] - bins[:-1]
    center = centers(bins)
    area = sum(h * width)

    if weights is not None and yerr == 'sumw2':
        sumw2 = np.histogram(x, bins=bins,
                             weights=weights**2)[0]
    else:
        sumw2 = h

    yerr = poisson_limits(sumw2, yerr)

    if xerr == 'binwidth':
        xerr = width / 2

    if density is not None:
        h = h / area
        yerr = yerr / area
        area = 1.

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    plt.errorbar(center, h, xerr=xerr, yerr=yerr, capsize=0,
                 markersize=0, **kwargs)

    return center, (yerr[0], h, yerr[1]), area

def make_split(ratio, gap=0.12):
    cax = plt.gca()
    box = cax.get_position()
    xmin, ymin = box.xmin, box.ymin
    xmax, ymax = box.xmax, box.ymax
    gs = GridSpec(2, 1, height_ratios=[ratio, 1 - ratio], left=xmin, right=xmax, bottom=ymin, top=ymax)
    gs.update(hspace=gap)

    ax = plt.subplot(gs[0])
    plt.setp(ax.get_xticklabels(), visible=False)
    bx = plt.subplot(gs[1], sharex=ax)

    return ax, bx

def plot_pull(data, func):
    ax, bx = make_split(0.8)

    plt.sca(ax)

    x, y, norm = histpoints(data)

    lower, upper = ax.get_xlim()

    xs = np.linspace(lower, upper, 200)
    plt.plot(xs, norm * func(xs), 'b-')

    #plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))

    plt.sca(bx)

    resid = y[1] - norm * func(x)
    err = np.zeros_like(resid)
    err[resid >= 0] = y[0][resid >= 0]
    err[resid < 0] = y[2][resid < 0]

    pull = resid / err

    plt.errorbar(x, pull, yerr=1, color='k', fmt='o')
    plt.ylim(-5, 5)
    plt.axhline(0, color='b')

    plt.sca(ax)

    return ax, bx

