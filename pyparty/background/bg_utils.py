""" Various utilities for parsing background styles to return M X N X 3 array.
Mostly designed to work for Canvas but might be useful in general """

from pyparty.utils import to_normrgb
import numpy as np

class BGError(self):
    """ Background error """

def from_color_res(color, resx, resy=None):
    """ Constant image of resolution (resx, resy) """
    if not resy:
        resy = resx
    background = np.empty( (int(resx), int(resy), 3) )

    color = to_normrgb(color)
    background[:,:,:] = color
    return background        

def from_string(self, path_or_color, resx=None, resy=None):
    """ Load an image from harddrive; wraps skimage.io.imread. 
        os.path.expanduser is called to allow ~/foo/path calls.
        If not found, return from_color_res, so resolution must be
        specified.  If filepath is found, resx, resy are ignored! """
    
    # Separte method because plan to  expan later
    try:
        background = skimage.io.imread(op.expanduser( path) )
    except IOError:
        if not resx:
            raise BGError("Background string interpreted as color; please pass"
            " resolution as well!")
    return from_color_res(background, resx, resy)
    

#def parse_bg(*bg):
    #""" Returns array background from various import types"""
    
    #if len(bg) == 2:
        #...
    #elif len(bg) == 1:
        