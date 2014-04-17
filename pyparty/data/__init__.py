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
           'test_labeled',
           'test_noise',
           'test_raster',
           'test_binary' 
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

def test_binary(*args,**kwargs):
    """ Binary, rasterized test particles """
    return load("test_binary.png", *args, **kwargs)

def test_labeled(*args,**kwargs):
    """ Colored, rasterized test particles """
    return load("test_labeled.png", *args, **kwargs)

def test_raster(*args,**kwargs):
    """ Rasterized test particles on varying background"""
    return load("test_raster.png", *args, **kwargs)

def test_noise(*args,**kwargs):
    """ Smooth, noisy particles on varying background"""
    return load("test_noise.png", *args, **kwargs)


if __name__ == '__main__':
    print gwu(rez=(50,5000)).shape  
    