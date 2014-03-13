""" Series of functions for fitting and binning of data.  Adapted from legacy
scripts at git@github.com:hugadams/npsurfacecounter.git"""

# SHOULD ANY OF THIS BE MIGRATED INTO PRIMARY PYPARTY?

import numpy as np
from scipy.optimize import curve_fit

def gauss(x, *p0):
    ''' Pass in best estimates for mu and sigma, it will use these as a 
    starting curve and fit to data from np.curvefit function.
    
    p0 is two arguments, the estimates for starting mean and sigma.  
    Needed by least squares fit. Notation to be compatible with curve_fit 
    requires this variable length argument notation.'''
    A, mu, sigma = p0
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def hist_max(counts, bins, idx_start=0, idx_stop=None):
    ''' Finds the max bin index and value from the histogram.  Can pass 
    indicies to drop so that it can avoid. '''    
    if not idx_stop:
        idx_stop = len(counts)
    crange = counts[idx_start:idx_stop]
    countmax = max(crange)
    # Ok here's trick.  Crange may be reduced, so I find the index of the
    # max on that range.  This might be creducedmax=8
    # but on my real range, this might be index 58, if idx start was 50.
    # Therefore, I add idx_start to the index

    max_idx = np.where(crange == countmax)[0]
    if max_idx.shape[0] > 1:
        logger.warn(
            'There are two bins in histogram that have idenitcal maxima and this might trip out gaussian autfit functions!  Using first one.')
    max_idx = max_idx[0] + idx_start
    binmax = bins[max_idx]
    # index of maximum, bin(x) value, count(y) value at max
    return max_idx, binmax, countmax


def bin_above_below(series, left, right, height, shuffle=True):
    ''' Given a bin of width right-left, this splits the bin at given height.  
        The data corresponding to this bin (presumably in series) is evently distributed.
        For example, if the bin has 100 entries, and the height is 50, the data will be
        sliced into two regions.  One from 0-50 and one from 50-100, to be returned as 
        below and above respectively.
        
        kwds:
           series: Series of data presumably form which the bin was derived originally.
           left/right: start/stop of slice of series.
           height: y value up the bin.  If greater than the bin top, all data in the bin is returned
                   as "below".
           shuffle: Should data in series be randomized so as not to give any ordering 
                    bias when split between above, below.
           
        returns:
            Tuple of series (above, below) with the data from the original series partitioned between.
    '''
    if shuffle:
        shuffitup(series)  # In place operation

    height = int(round(height))
    sample = series[(series >= left) & (series <= right)]  # Slice by values
    counts = len(sample)
    if height >= counts:
        # Empty series is easier for type checking in my use case
        below, above = sample, Series()

    else:
        below = sample.iloc[0:height]  # just doing sample[] works too
        above = sample.iloc[height::]

    return below, above


def range_slice(series, start, stop, style='value'):
    ''' Convienence wrapper to slice a single column of a df by either value or index.  Force user to specify
    to reduce possibility of error (aka pass integer when meant to pass float).'''

    if style == 'index':
        start, stop = int(start), int(stop)
    elif style == 'value':
        working = series.ix[series >= start]
        working = working.ix[working <= stop]
    else:
        raise AttributeError(
            'range slice can only slice by index or value, you passed %s' % style)
    return working


def fit_normal(counts, binpoints):
    ''' Fits a normal distribution to a histogram assuming the max of the histogram
    makes up the mean.  To fit with cropped data, pass cropped data into this.  For
    now this is incomplete and maybe not useful because it requires knowledge of 
    standard deviation as well.'''

    idx_max, mu, amp_at_mean = hist_max(counts, binpoints)  # Skip first bin

    # Get the standard devation of a histogram
    # Does not rely on original data at all
    std = data_from_histogram(counts, binpoints).std()

    # Fit normal distribution at each point along binpoints array
    # Scale it based on the max value of the histogram
    normal = mlab.normpdf(binpoints, mu, std)
    scaleup = amp_at_mean / normal[idx_max]
    return normal * scaleup


def data_from_histogram(counts, binpoints):
    ''' Gets the standard devation from histogram data (binned by midpoint, not edges).
    Doesn't use real data at all.'''
    psuedo_data = []
    for i, point in enumerate(binpoints):
        # [1,1,1, + 2,2,2,2,2, + etc...]
        psuedo_data = psuedo_data + ([point] * counts[i])
    return np.array([psuedo_data])

### Probably deprecated, using wrong sigma and shit ###


def optimize_gaussian(counts, binpoints):
    ''' Takes in counts of a histogram and bin_midpoints (see get_bin_points function) and
    uses least squares regression to fit a guassian.
    
    Density counts is an array of weights, actual counts from np.histogram density=True'''
    mu, amp_at_mean = hist_max(counts, binpoints)[1:3]  # Skip first bin
    sig = data_from_histogram(counts, binpoints).std()
    ### Make a best guest a initial params... [1.0, 0.0, 1.0] works fine ###
    p0 = [amp_at_mean, mu, sig]
    coeff, var_matrix = curve_fit(gauss, binpoints, counts, p0=p0)
    hist_fit = gauss(binpoints, *coeff)

    # Fit coefficients if you want to keep them
    fit_amp, fit_mean, fit_sig = coeff
    return (hist_fit, fit_amp, fit_mean, fit_sig)


def psuedo_symmetric(counts, binpoints, idx_start=0, idx_stop=None):
    ''' Takes in a histogram, examines left edge to max, and then returns symmetric values
    as a pseudo-dataset. '''
    max_idx, x_mean, y_mean = hist_max(
        counts, binpoints, idx_start=idx_start, idx_stop=idx_stop)
    # idx start ... idx mean ... idx start ... idx mean +1 (for symm)
    symrange = range(idx_start, (1 + max_idx + (max_idx - idx_start)))
    symcounts = np.array([counts[data_idx] for data_idx in symrange])
    symbins = np.array([binpoints[data_idx] for data_idx in symrange])
    return symcounts, symbins


def get_bin_points(binarray, position='c'):
    ''' Takes in an array of bin edges (returned from histogram) as returned from a histogram
    and finds the values at the midpoints, left or right edges.  Useful when plotting a scatter 
    plot over bins for example.'''

    # Return center xvalue of bins
    if position == 'c':
        # I don't quite understand this notation
        return (binarray[:-1] + binarray[1:]) / 2.0

    # Return left edge xvalue of bins
    elif position == 'l':
        return np.array([binarray[i] for i in range(len(binarray - 1))])

    # Return right edge xvalue of bins
    # Note, bins are defined by their left edges, so this has to compute the bin width.
    # Does so by looking at first two elements.  THEREFORE SPACING MUST BE
    # EVEN!
    elif position == 'r':
        width = binarray[1] - binarray[0]
        return np.array([binarray[i] + width for i in range(len(binarray - 1))])
