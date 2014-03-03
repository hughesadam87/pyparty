# Import sample data.  This import structure was taken directly from skimages
# data import style:
    # https://github.com/scikit-image/scikit-image

import os.path as op
from skimage.io import imread
from pyparty import data_dir
from pyparty.utils import crop

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

def load(f, rez=None):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.
    
    rez : (int, int) or None
        Crop image to desired length by width.

    Returns
    -------
    img : ndarray
        Image loaded from skimage.data_dir.
    """
    img = imread(op.join(data_dir, f))
    if rez:
        rx, ry = rez
        img = crop(img, (0,0,rx,ry))
    return img

def gwu(*args, **kwargs):
    """ New George Washington University Logo """
    return load("gwu.png", *args, **kwargs)

def spectrum(*args, **kwargs):
    """ 3d Spectrum : original image by pyparty author """
    return load("spectrum.jpg", *args, **kwargs)

def lena_who(*args, **kwargs):
    """ Yound lady, released with her full permission """
    return load("lena_who.jpg", *args, **kwargs)

def nanogold(*args, **kwargs):
    """ Scanning electron image of Gold Nanoparticles 100,000 X 
    magnification.  Original image is property of Adam Hughes, 
    Reeves Cond. Matter Physics Group, and released for public
    distribution."""
    return load("nanogold.tif", *args, **kwargs)

def nanolabels(*args, **kwargs):
    """ size-segmented (nanogold); see nanogold description"""
    return load("nanolabels.tif", *args, **kwargs)

def nanobinary(*args, **kwargs):
    """ binarized nanogold using trainable pixel classification """
    return load("nanobinary.tif", *args, **kwargs)

def test_plain(*args, **kwargs):
    """ 1024 x1024 resolution; 30nm Diameter circles; 
    pure white (1,1,1) on gray (.5,.5,.5) background. """
    return load("test_plain.png", *args, **kwargs)

def test_contrast(*args, **kwargs):
    """ See test_plain().__doc__.  Added local contrast fluctuations. """
    return load("test_contrast.png", *args, **kwargs)

def test_salty(*args, **kwargs):
    """ See test_contrast.__doc__.  Added 45% salt (1,1,1) noise. """
    return load("test_salty.png", *args, **kwargs)

if __name__ == '__main__':
    print gwu(rez=(50,5000)).shape  
    