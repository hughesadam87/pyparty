""" Various utilities for parsing background styles to return M X N X 3 array.
Mostly designed to work for Canvas but might be useful in general """

import os.path as op
from skimage.io import imread
from pyparty.utils import to_normrgb
import numpy as np
from pyparty.utils import any2rgb

class BackgroundError(Exception):
    """ Background error """

# Not sure best way to implement; EG as utility or form canvas
def from_grid(grid_obj):
    """ Up-normalizes Grid.zz array to 1.0 and passes to any2rgb for array 
    conversion.  This way, the upper lim will always roundto 255"""
    zz_array = grid_obj.zz
    return any2rgb(zz_array / float(zz_array.max()) )

def from_color_res(color, resx, resy=None):
    """ Constant image of resolution (resx, resy) """
    if not resy:
        resy = resx
    background = np.empty( (int(resx), int(resy), 3) )

    color = to_normrgb(color)
    background[:,:,:] = color
    return background        


def from_string(path_or_color, resx=None, resy=None):
    """ Load an image from harddrive or URL; wraps skimage.io.imread. 
    os.path.expanduser is called to allow ~/foo/path calls.
    If not found, path_or_color is assumed to be a colorstring (eg 'aqua')
    """
    
    # Separte method because plan to expand later
    try:
        return imread(op.expanduser( path_or_color) ) #expand user ok for URL
    except IOError:
        if not resx:
            raise BackgroundError("Background string interpreted as color; "
                "please pass resolution as well!")
        else:
            background = path_or_color
        
    # Raise more clear error    
    try:
        return from_color_res(background, resx, resy)
    except Exception:
        raise BackgroundError("Failed to interpret background as a path,"
            "  URL, or color string (eg 'aqua').")