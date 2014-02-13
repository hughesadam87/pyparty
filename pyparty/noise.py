""" Utilities for adding noise to image arrays."""

import numpy as np
from pyparty.utils import to_normrgb, rgb2uint

DEFAULT_COVERAGE = 0.10

class NoiseError(Exception):
    """ """

def _parse_intensity(img, intensity):
    """ Validate intensity to any pyparty color/intensity, then grayconvert
    if image is gray.  Allows for 'red' to be valid noise for example"""
    intensity = to_normrgb(intensity)
    if img.ndim == 2:
        intensity = rgb2uint(intensity)
    return intensity        

            
def _parse_coverage(coverage):
    if coverage < 0 or coverage > 1:
        raise NoiseError("Coverage must be between 0-1")


def noise(img, intensity, coverage=DEFAULT_COVERAGE):
    """Noise of arbitrary intensity/color added to img at coverage"""
    _parse_coverage(coverage)
    intensity = _parse_intensity(img, intensity)
    noisey = np.random.random(img.shape[0:2])    
    img[noisey > 1- coverage] = intensity
    return img


def pepper(img, coverage=DEFAULT_COVERAGE):
    """Black noise"""
    return noise(img, (0,0,0), coverage)


def salt(img, coverage=DEFAULT_COVERAGE):
    """White noise"""

    return noise(img, (1,1,1), coverage)

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