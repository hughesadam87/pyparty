""" Utilities for adding noise to image arrays."""

import numpy as np
from pyparty.utils import to_normrgb, rgb2uint, copyarray, pp_dtype_range

DEFAULT_COVERAGE = 0.10

# Promote a few numpy sampling functions
RANDSAMP = np.random.random_sample #Random 0-1
RANDNORM = np.random.normal

class NoiseError(Exception):
    """ """

def _parse_intensity(img, intensity):
    """ Validate intensity to any pyparty color/intensity, then grayconvert
    if image is gray.  Allows for 'red' to be valid noise for example"""

    if isinstance(intensity, np.ndarray):
        raise NoiseError('Intensity/color cannot be an array.')
    intensity = to_normrgb(intensity)
    if img.ndim == 2:
        r,g,b = intensity
        intensity = 0.2125 * r + 0.7154 * g + 0.0721 * b
    return intensity        

            
def _parse_coverage(coverage):
    if coverage < 0 or coverage > 1:
        raise NoiseError("Coverage must be between 0-1")


@copyarray
def multicolor(img, coverage=DEFAULT_COVERAGE):
    """Generates random color array for noise on range 0-1."""

    _parse_coverage(coverage)
    noisycolor = RANDSAMP(size=img.shape)
    noisey = RANDSAMP(size=img.shape[0:2])   
    mask = (noisey > 1 - coverage)
    img[mask] = noisycolor[mask]
    return img

@copyarray
def color(img, coverage=DEFAULT_COVERAGE, intensity='yellow'):
    """Noise of arbitrary intensity/color added to img at coverage.
    Defaults to red."""
    
    _parse_coverage(coverage)    
    intensity = _parse_intensity(img, intensity)
    noisey = RANDSAMP(size=img.shape[0:2])    
    img[noisey > 1- coverage] = intensity
    return img

@copyarray
def normal(img, coverage=DEFAULT_COVERAGE, stddev=.1, mean=False):
    """
    Wraps np.random.randnormal() for gray, color images.
    
    Parameters
    ----------
    stdev : Percentage of min, max values in image for standard deviation.
        For example, .1 = 10% of gray image is 25.5 (10% of 255)
    mean : Mean value for noise.
        Defaults to half of min/max of image.
        
    """

    _parse_coverage(coverage)
    
    if stddev < 0 or stddev > 1:
        raise AttributeError("Standard deviation factor must in range be 0-1.")
    
    # Ok for color? (Maybe make dtype range?)
    xmin, xmax = pp_dtype_range(img)

    # Need to do mapping of mean and stdev and so on and map!
    if img.ndim == 3:
        raise NotImplementedError("3-channel distributions not yet built")

    if not mean:
        mean = 0.5 * (xmax + xmin)

    stddev = mean * stddev
    
    gnoise = RANDNORM(loc=mean, scale=stddev, size=img.shape)
    gnoise[gnoise>xmax] = xmax
    gnoise[gnoise<xmin] = xmin
    
    noisey = RANDSAMP(size=img.shape[0:2])    
    mask = (noisey > 1 - coverage)    
    img[mask] = gnoise[mask]
    return img



def pepper(img, coverage=DEFAULT_COVERAGE):
    """Black noise"""
    return color(img, coverage, (0,0,0))


def salt(img, coverage=DEFAULT_COVERAGE):
    """White noise"""
    return color(img, coverage, (1,1,1),)


@copyarray
def saltpepper(img, coverage=DEFAULT_COVERAGE):
    """Split coverage in half between white and black pixels"""
    
    #Don't call whitenoise/blacknoise because could overwite pixels
    #giving less coverage
    _parse_coverage(coverage)    
    white = _parse_intensity(img, (1,1,1) )
    black = _parse_intensity(img, (0,0,0) )

    halfcov = 0.5 * coverage
    noise = np.random.random(img.shape[0:2]) #RANDSAMP?
    img[noise > 1 - halfcov] = white
    img[noise <= halfcov] = black
    return img    

if __name__ == '__main__':
    img = RANDSAMP(size=(50,50,3))
    multicolor(img, .50)
    from skimage.color import rgb2gray
    grayimg = rgb2gray(img)
    normal(img, stddev=0.9)
    
    