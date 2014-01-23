#Easy access to some common thresholding functions from skimage:
#http://scikit-image.org/docs/dev/api/skimage.filter.thresholding.html
import numpy as np
from skimage import img_as_bool
from skimage.filter import threshold_adaptive, threshold_otsu
import functools

class ThresholdError(Exception):
    """ """

def invertible(fcn):
    """ Return np.invert() of function"""
    @functools.wraps(fcn)
    def wrapper(*args, **kwargs):
        invert = kwargs.pop('invert', None)
        if invert:
            return np.invert(fcn(*args, **kwargs))
        else:
            return fcn(*args, **kwargs)
    return wrapper

def _valid_range(n):
    if n < 0 or n > 255:
        raise ThresholdError('n must be between 0 and 255')   

# ALL FUNCTIONS MUST ACCEPT KWARGS
@invertible
def img2bool(grayimage, **kwargs):
    """ Allow for *args, **kwargs to be passed """
    return img_as_bool(grayimage)

@invertible
def nonzero(grayimage, **kwargs):
    """ All non-zero elements are True """
    return (grayimage > 0) 

@invertible
def double(grayimage, nlow=75, nhigh=150):
    """ Lower and upper thresholds on range 0-255"""
    _valid_range(nlow)
    _valid_range(nhigh)
    return (grayimage >= nlow) & (grayimage <= nhigh)


@invertible
def single(grayimage, n=128):
    """ Single threshold value between 0, 255. """
    return double(grayimage, nlow=n, nhigh=255)

@invertible
def nonwhite(grayimage, **kwargs):
    """ Return any pixel under 255 """
    return (grayimage < 255)

@invertible
def nonblack(grayimage, **kwargs):
    """ Return any pixel over 0 """
    return (grayimage > 0)
    

#http://scikit-image.org/docs/dev/api/skimage.filter.html#skimage.filter.threshold_otsu
@invertible
def otsu(grayimage, nbins=256):
    """ Otsu provides an upper threshold limit, meaning anything under threshold
    is foreground. """
    thresh = threshold_otsu(grayimage, nbins=nbins)
    return (grayimage <= thresh)

@invertible
def adaptive(grayimage, **kwargs):
    return threshold_adaptive(grayimage, **kwargs)

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
    'nonzero' : nonzero,
    'adaptive' : adaptive,
    'otsu' : otsu,
    'nonwhite' : nonwhite,
    'nonblack' : nonblack
    }
