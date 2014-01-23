#Store some common thresholding functions from skimage:
#http://scikit-image.org/docs/dev/api/skimage.filter.thresholding.html
import numpy as np
from skimage import img_as_bool
from skimage.filter import threshold_adaptive, threshold_otsu
import functools

class ThresholdError(Exception):
    """ """

def _valid_range(n):
    if n < 0 or n > 255:
        raise ThresholdError('n must be between 0 and 255')   

# ALL FUNCTIONS MUST ACCEPT KWARGS
def img2bool(grayimage, **kwargs):
    """ Allow for *args, **kwargs to be passed """
    return img_as_bool(grayimage)


def double(grayimage, nlow=75, nhigh=150):
    """ Lower and upper thresholds on range 0-255"""
    _valid_range(nlow)
    _valid_range(nhigh)
    return (grayimage >= nlow) & (grayimage <= nhigh)


def single(grayimage, n=128):
    """ Single threshold value between 0, 255. """
    return double(grayimage, nlow=n, nhigh=255)

#http://scikit-image.org/docs/dev/api/skimage.filter.html#skimage.filter.threshold_otsu
def otsu(grayimage, nbins=256, invert=False):
    """ Otsu provides an upper threshold limit, meaning anything under threshold
    is foreground.  The invert keyword is provided for the reverse case."""
    thresh = threshold_otsu(grayimage, nbins=nbins)
    binary = (grayimage <= thresh)
    if invert:
        binary = np.invert(binary)
    return binary

def choose_thresh(fcnname, **kwargs):
    try:
        fcn = thr_select[fcnname]
    except KeyError:
        raise ThresholdError('invalid function "%s"; choose from "%s"' 
              % (fcnname, '", "'.join( sorted(thr_select.keys()))) )
    return functools.partial(fcn, **kwargs)

thr_select={
    'single' : single,
    'double' : double,
    'img2bool' : img2bool,
    'adaptive' : threshold_adaptive,
    'otsu' : otsu
    }
