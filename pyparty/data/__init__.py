# Import sample data.  This import structure was taken directly from skimages
# data import style:
    # https://github.com/scikit-image/scikit-image

import os.path as op
from skimage.io import imread
from pyparty import data_dir

__all__ = ['gwu', 
           'spectrum',
           'lena_who',
           'nanogold',
           'nanolabels',
           'nanobinary',
	   'test_plain',
	   'test_contrast',
           'test_salty'	   
           ]

def help():
    """ What's canonical way to add help to modules?"""
    return 'Sample images: %s' % ",".join(__all__)


def load(f):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.

    Returns
    -------
    img : ndarray
        Image loaded from skimage.data_dir.
    """
    return imread(op.join(data_dir, f))


def gwu():
    """ New George Washington University Logo """
    return load("gwu.png")

def spectrum():
    """ 3d Spectrum : original image by pyparty author """
    return load("spectrum.jpg")

def lena_who():
    """ Yound lady, released with her full permission """
    return load("lena_who.jpg")

def nanogold():
    """ Scanning electron image of Gold Nanoparticles 100,000 X 
    magnification.  Original image is property of Adam Hughes, 
    Reeves Cond. Matter Physics Group, and released for public
    distribution."""
    return load("nanogold.tif")

def nanolabels():
    """ size-segmented (nanogold); see nanogold description"""
    return load("nanolabels.tif")

def nanobinary():
    """ binarized nanogold using trainable pixel classification """
    return load("nanobinary.tif")

def test_plain():
    """ 1024 x1024 resolution; 30nm Diameter circles; 
    pure white (1,1,1) on gray (.5,.5,.5) background. """
    return load("test_plain.png")

def test_contrast():
    """ See test_plain().__doc__.  Added local contrast fluctuations. """
    return load("test_contrast.png")

def test_salty():
    """ See test_contrast.__doc__.  Added 45% salt (1,1,1) noise. """
    return load("test_salty.png")
