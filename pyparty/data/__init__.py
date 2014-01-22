# Import sample data.  This import structure was taken directly from skimages
# data import style:
   # https://github.com/scikit-image/scikit-image

import os.path as op
from skimage.io import imread
from pyparty import data_dir


__all__ = ['gwu', 
           'spectrum',
           'lena_who'
           ]

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
    return load("gwu.png")

def spectrum():
    return load("spectrum.jpg")

def lena_who():
    return load("lena_who.jpg")