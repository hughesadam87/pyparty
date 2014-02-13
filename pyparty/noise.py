""" Utilities for adding noise to image arrays."""

import numpy as np
from pyparty.utils import to_normrgb, rgb2uint, copyarray

DEFAULT_COVERAGE = 0.10
RANDSAMP = np.random.random_sample 

class NoiseError(Exception):
    """ """

def _parse_intensity(img, intensity):
    """ Validate intensity to any pyparty color/intensity, then grayconvert
    if image is gray.  Allows for 'red' to be valid noise for example"""

    if isinstance(intensity, np.ndarray):
        raise NoiseError('Intensity/color cannot be an array.')
    intensity = to_normrgb(intensity)
    if img.ndim == 2:
        intensity = rgb2uint(intensity)
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
def color(img, coverage=DEFAULT_COVERAGE, intensity='red',):
    """Noise of arbitrary intensity/color added to img at coverage.
    Defaults to red."""
    
    _parse_coverage(coverage)    
    intensity = _parse_intensity(img, intensity)
    noisey = RANDSAMP(size=img.shape[0:2])    
    img[noisey > 1- coverage] = intensity
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
    noise = np.random.random(img.shape[0:2])
    img[noise > 1 - halfcov] = white
    img[noise <= halfcov] = black
    return img    

if __name__ == '__main__':
    img = RANDSAMP(size=(50,50,3))
    multicolor(img, .50)
    