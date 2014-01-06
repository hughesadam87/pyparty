from __future__ import division
import logging
import math
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.axes import Subplot


from skimage import img_as_float
from skimage.color import gray2rgb

from pyparty.config import PCOLOR, COLORTYPE

logger = logging.getLogger(__name__) 

# COLOR RELATED ATTRIBUTES
CTYPE, CBITS = COLORTYPE

ERRORMESSAGE = 'Valid color arguments include color names ("aqua"), ' +  \
   'rgb-tuples (.2, .4, 0.), ints/floats (0-%s) or (0.0-1.0) or' % CBITS + \
   'hexcolorstrings (#00FFFF).' 

#http://matplotlib.org/api/colors_api.html#matplotlib.colors.ColorConverter            
_rgb_from_string = colors.ColorConverter().to_rgb

class UtilsError(Exception):
    """ General utilities error """ 

class ColorError(Exception):
    """ Particular to color-utilities """   

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
    
    if isinstance(color, bool):
	if color:
	    return (1.,1.,1.)
	return (0.,0.,0.)

    raise ColorError(ERRORMESSAGE)


def any2rgb(array, name=''):
    """ Returns a normalized float 3-channel array regardless of original
    dtype and channel.  All valid pyparty images must pass this
    
    name : str
        Name of array which will be referenced in logger messages"""
    

    # *****
    # Quick way to convert to float (don't use img_as_float becase we want
    # to enforce that upperlimit of 255 is checked
    array = array / 1.0  
	
    # Returns scalar for 1-channel OR 3-channel
    if array.max() > 1:
	# For 8-bit, divide by 255!
	if array.max() > COLORTYPE[1]:
	    raise BackgroundError("Only 8bit ints are supported for now")
	array = array / COLORTYPE[1] 
    
    if array.ndim == 3:
        return array 
    
    elif array.ndim == 2:
        logger.warn('%s color has been converted (1-channel to 3-channel RGB)'
                    % name)
        return gray2rgb(array)
        
    raise BackgroundError('%s must be 2 or 3 dimensional array!' % name )    

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

def rotate_vector(array, theta, style='degrees', rint='up'):
    """ Rotate an array ocounter-clockwise through theta.  rint rounds output 
    to integer; up rounds normally, down does int rounding (ie rounds down).  
    ARRAY MUST BE MEAN-CENTERED if rotating in place.
    
    array may be xy pairs [(x1,y1),(x2,y2)] or N,2 matrix."""

    if style == 'degrees':
        theta = math.radians(theta)
        
    costheta, sintheta = math.cos(theta), math.sin(theta)
    
    rotMatrix = np.array([
        [costheta, -sintheta],  
        [sintheta,  costheta]
                     ])
    r_array = np.dot(array, rotMatrix)
    
    if rint:
        if rint =='up':
            r_array = np.rint(r_array)
        r_array = r_array.astype('int', copy=False) #saves memory
    return r_array

def _parse_ax(*args, **kwargs):
    """ Return axes from args, kwargs """
    
    axes = kwargs.pop('axes', None)       

    if not axes:
	indicies = [idx for (idx, arg) in enumerate(args) if isinstance(arg, Subplot)]
	if len(indicies) < 1:
	    axes = None
	elif len(indicies) > 1:
	    raise UtilsError("Multiple axes not understood")
	else:
	    args = list(args)
	    axes = args.pop(indicies[0])      
    return axes, args, kwargs

# showim(img, ax)
def showim(image, *args, **kwargs):
    """ Similar to imshow with a few more keywords"""

    title = kwargs.pop('title', None)
    axes, args, kwargs = _parse_ax(*args, **kwargs)
        
    if axes:
	axes.imshow(image, *args, **kwargs)
    else:      # matplotlib API asymmetry
	axes = plt.imshow(image, *args, **kwargs).axes        
    
    if title:
	axes.set_title(title)
    return axes    


def unzip_array(pairs):
    """ Inverse of np.array(zip(x,y)). Rerturn unzipped array of pairs:
    (1,2), (25,5) --> array(1,25), array(25,25)."""
    return np.array( zip(*(pairs) ) )

def subplots(*args, **kwds):
    """ Wrapper to plt.subplots(r, c).  Will return flattened axes and discard
    figure.  'flatten' keyword will not flatten if the plt.subplots() return
    is not itself flat.  If flatten=False and fig=True, standard plt.subplots
    behavior is recovered."""
    
    flatten = kwds.pop('flatten', True)
    _return_fig = kwds.pop('fig', False)
        	
    fig, args = plt.subplots(*args, **kwds)
    
    # Seems like sometimes returns flat, sometimes returns list of lists
    # so either way I flatten
    try:
        args = [ax.axes for ax in args] 
    except Exception:
	if flatten:
            args = [ax.axes for row in args for ax in row]
	else:
	    args = [tuple(ax.axes for ax in row) for row in args]
	    
    if _return_fig:
	return (fig, args)
    else:
        return args
                
# Used with crop
def _get_xyshape(image):
    """Returns first two dimensions of an image, whether it is 2d or 3d, 
       as is the case of colored images.

    Parameters
    ----------
    image: a ndarray
    
    Returns:
    -----------
    img_xf, img_yf: shape of first and second dimension of array
    
    Raises
    ------
    UtilsError
        If image shape is not 2 or 3.
    
    """

    ndim = len(image.shape)

    if ndim == 3:
        img_xf, img_yf, z = image.shape

    elif ndim == 2:
	img_xf, img_yf = image.shape

    else:
	raise UtilsError('Image must have dimensions 2 or 3 (received %s)' % ndim)

    return img_xf, img_yf


def crop(image, coords):
    """Crops a rectangle (xi, yi, xf, yf) from an image.  If image
       is 3-dimenionsal (eg color image), slices on first two dimensions.

    Parameters
    ----------
    image: a ndarray
    coords : (xi, yi, xf, yf)
        lenngth-4 iterable with coordiantes corresponding to rectangle corners
	in order (xi, yi, xf, yf)

    Notes
    -----
    Allows for xf/yf > xi/yi for more flexible rectangle drawing.
    Please refer to the numpy indexing API for de-facto slicing. 

    Raises
    ------
    UtilsError
    	If more or less than 4 coordinates are passed.
        If x or y rectangle coordinates exceed the range of image (image.shape)


    Examples
    --------
    >>> from skimage import data
    >>> lena = img_as_float(data.lena())
    >>> crop(lena, (0,0,400,300))	
	
    """

    img_xf, img_yf = _get_xyshape(image)
    
    try:
        xi, yi, xf, yf = coords
    except Exception:
	raise UtilsError("Coordinates must be lenth four iterable of form"
	    "(xi, yi, xf, yf).  Instead, received %s" % coords)


    # Make sure crop limits are in range of image
    for x in (xi, xf):
        if x < 0 or x > img_xf:
	    raise UtilsError('Cropping bounds (%s, %s) exceed'
                ' image horizontal range (%s, %s)' % (xi, xf, 0, img_xf))

    for y in (yi, yf):
        if y < 0 or y > img_yf:
    	    raise UtilsError('Cropping bounds (%s, %s) exceed'
                ' image vertical range (%s, %s)' % (yi, yf, 0, img_yf))

    # Reverse bounds if final exceeds initial
    if yf < yi:
	yi, yf = yf, yi

    if xf < xi:
	xi, xf = xf, xi

    ndim = len(image.shape)
    if ndim == 3:
        image = image[yi:yf, xi:xf, :]
    else:
	image = image[yi:yf, xi:xf]   

    return image