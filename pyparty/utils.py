import logging
import math
import numpy as np
from pyparty.config import PCOLOR, COLORTYPE
import matplotlib.colors as colors

logger = logging.getLogger(__name__) 


# COLOR RELATED ATTRIBUTES
CTYPE, CBITS = COLORTYPE

ERRORMESSAGE = 'Valid color arguments include color names ("aqua"), ' +  \
   'rgb-tuples (.2, .4, 0.), ints/floats (0-%s) or (0.0-1.0) or' % CBITS + \
   'hexcolorstrings (#00FFFF).' 

#http://matplotlib.org/api/colors_api.html#matplotlib.colors.ColorConverter            
_rgb_from_string = colors.ColorConverter().to_rgb

class ColorError(Exception):
    """ """

def _pix_norm(value, imax=CBITS):
    """ Normalize pixel intensity to colorbit """
    if value > imax:
        raise ColorError("Pixel intensity cannot exceed %s" % imax)
    return float(value) / imax

def to_normrgb(color):
    """ Returns an rgb len(3) tuple on range 0.0-1.0 with several input styles; 
        wraps matplotlib.color.ColorConvert """
       
    if color is None:
        return PCOLOR

    # If iterable, assume 3-channel RGB
    if hasattr(color, '__iter__'):
        if len(color) != 3:
            raise ColorError("Multi-channel color must be 3-channel;"
                                 " recieved %s" % len(color))
        r, g, b = color
        if r <= 1 and g <= 1 and b <= 1:
            return (r, g, b)

        elif r >= 1 and g >= 1 and b >= 1:
            r, g, b = map(_pix_norm, (r, g, b) )        
            return (r, g, b)

        else:
            raise ColorError("Multi-channel color style ambiguous. (r, g, b)"
                " elements must all be < 1 or all > 1 (normalized to %s pixels)" 
                % CBITS)
    
    if isinstance(color, str):
        return _rgb_from_string(color)
    
    # If single channel --> map accross channels EG 22 --> (22, 22, 22)
    if isinstance(color, int):
        color = float(color)
        
    if isinstance(color, float):
        if color > 1:
            color = _pix_norm(color)
        return (color, color, color)

    raise ColorError(ERRORMESSAGE)

class UtilsError(Exception):
    """ General utilities error """

def coords_in_image(rr_cc, shape):
    """ Taken almost directly from  skimage.draw().  Decided best not to
        do any formatting implicitly in the shape models.
        
        Attributes
        ----------
        rr_cc : len(2) iter
            rr, cc returns from skimage.draw(); or shape_models.rr_cc
            
        shape : len(2) iter
            image dimensions (ie 512 X 512)
        
        Returns
        -------
        (rr, cc) : tuple(rr[mask], cc[mask])

        """

    rr, cc = rr_cc 
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return (rr[mask], cc[mask])            
            
def where_is_particle(rr_cc, shape):
    """ Quickly evaluates if particle rr, cc is fully within, partically in,
        or is outside and image.  Does this by comparin shapes, so is fast,
        but does not track which portions are outside, inside or on edge.
        
        Attributes
        ----------
        rr_cc : len(2) iter
            rr, cc returns from skimage.draw(); or shape_models.rr_cc
            
        shape : len(2) iter
            image dimensions (ie 512 X 512)
        
        Returns
        -------
        'in' / 'out' / 'edge' : str
        """
    
    
    rr_cc_in = coords_in_image(rr_cc, shape)

    # Get dimensions of rr_cc vs. rr_cc_in
    dim_full = ( len(rr_cc[0]), len(rr_cc[1]) )
    dim_in = ( len(rr_cc_in[0]), len(rr_cc_in[1]) ) 
    
    if dim_in == dim_full:
        return 'in'
    elif dim_in == (0, 0):
        return 'out'
    else:
        return 'edge'
    
def rr_cc_box(rr_cc):
    """ Center the rr_cc values in a binarized box."""

    rr, cc = rr_cc
    ymin, ymax, xmin, xmax = rr.min(), rr.max(), cc.min(), cc.max()

    # Center rr, cc mins to 0 index
    rr_trans = rr - ymin
    cc_trans = cc - xmin
    rr_cc_trans = (rr_trans, cc_trans)
    
    dx = xmax-xmin
    dy = ymax-ymin
    
    rect=np.zeros( (dy+1, dx+1), dtype='uint8' ) 
    rect[rr_cc_trans] = 1
    return rect   

def rotate_vector(array, theta, style='degrees', rint=False):
    """ Rotate an array of len(2) pairs [(x1,y1), (x2,y2)] counter-clockwise 
        through theta.  rint rounds output to integer."""
    if style == 'degrees':
        theta = math.radians(theta)
        cos, sin = math.cos, math.sin        
    
    rotMatrix = np.array([
        [cos(theta), -sin(theta)],  
        [sin(theta),  cos(theta)]
                     ])
    
    r_array = np.dot(array, rotMatrix)
    if rint:
        r_array = np.rint(r_array)
    return r_array

def unzip_array(pairs):
    """ Rerturn unzipped array of pairs:
    (1,2), (25,5) --> array(1,25), array(25,25)"""
    return np.array( zip(*(pairs) ) )
        