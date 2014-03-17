from __future__ import division
import logging
import math
import random 
import os
import os.path as op
import operator
import functools
from types import GeneratorType

import numpy as np
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes import Subplot
import skimage.color as skcol
from skimage import img_as_float, img_as_ubyte, exposure
from pyparty.config import PCOLOR, COLORTYPE

logger = logging.getLogger(__name__) 

# COLOR RELATED ATTRIBUTES
CTYPE, CBITS = COLORTYPE

ERRORMESSAGE = 'Valid color arguments include color names ("aqua"), ' +  \
    'rgb-tuples (.2, .4, 0.), ints/floats (0-%s) or (0.0-1.0) or' % CBITS + \
    ' hexcolorstrings (#00FFFF).' 

#http://matplotlib.org/api/colors_api.html#matplotlib.colors.ColorConverter            
_rgb_from_string = mplcolors.ColorConverter().to_rgb

class UtilsError(Exception):
    """ General utilities error """ 

class ColorError(Exception):
    """ Particular to color-utilities """   

def _get_ccycle(upto=None):
    """ Return a list of the current color cycle in MPL.
    Hacky workaround to not being able to access the color
    cycle container.  Creates and destroys intermedia figure.
    upto will crop or repeat until cycle reaches enough entries.
    """
    fig, axfoo = plt.subplots()
    c = axfoo._get_lines.color_cycle.next()
    clist = []
    
    # Iterate until duplicate is found
    while c not in clist:
        clist.append(c)
        c = axfoo._get_lines.color_cycle.next()        
        
    if upto is not None:
        if upto <= len(clist):
            pass
        else:
            while len(clist) < upto:
                clist = clist+clist
        clist = clist[0:upto]
    # Remove the temporary figure (Dangerous?)
    plt.close()
    return clist

def rand_color(style=None):
    """ Random color of various styles """
    if style == 'hex':
        r = lambda: random.randint(0,255)
        return ('#%02X%02X%02X' % (r(),r(),r()))

    elif style == 'bright':
        r = lambda: random.uniform(.5, 1.0)
        return ( r(), r(), r() )

    else:
        r = lambda: random.random()
        return  ( r(), r(), r() )
    
def guess_colors(guess, values):
    """ Match closest via squared channel distance between user-guessed
    colors and true values.  All are mapped to rgb, so guesses like 'red'
    are valid.
    """
    if not hasattr(guess, '__iter__') and not hasattr(values, '__iter__'):
        guess, values = [guess], [values]

    if len(guess) != len(values):
        raise UtilsError('Guesses and values must have equal lengths:'
                         ' (%s vs %s)' %(len(guess),len(values) ) )

    guess = map(to_normrgb, guess)
    values = map(to_normrgb, values)
    
    NotImplemented
    #Returns?


def _pix_norm(value, imax=CBITS):
    """ Normalize pixel intensity to colorbit """
    if value > imax:
        raise ColorError("Pixel intensity cannot exceed %s" % imax)
    return float(value) / imax

def _parse_generator(generator, astype=tuple):
    """ Convert generator as tuple, list, dict or generator.
        
    Parameters
    ----------
    astype : container type (tuple, list, dict) or None
        Return expression as tuple, list... if None, return as generator. 

    Notes
    -----
    Mostly useful for operations that in some cases return a dictionary, 
    but also might be useful as a list of kv pairs etc...
    """        
    if not isinstance(generator, GeneratorType):
        raise UtilsError("Generator required; got %s" % type(generator))
    
    if isinstance(astype, str):
        astype = eval(astype)        

    if astype:
        return astype(generator)

    else:
        return generator    

def copyarray(fcn):
    """ Decorator to return copy of an array. ARRAY MUST BE FIRST ARG!! """
    @functools.wraps(fcn)
    def wrapper(*args, **kwargs): #why aren't kwargs found?
        args=list(args)
        args[0] = np.copy(args[0])
        return fcn(*args, **kwargs)
    return wrapper


def invert(image):
    """ Invert a boolean, gray or rgb image.  Inversions are done through
    by subtracts (255-img or (1,1,1) - img).  Image and its inverse should
    sum to white!"""
    return pp_dtype_range(image)[1] - image     


def to_normrgb(color):
    """ Returns an rgb len(3) tuple on range 0.0-1.0 with several input styles; 
        wraps matplotlib.color.ColorConvert.  If None, returns config.PCOLOR by
        default."""

    if color is None:
        color = PCOLOR

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
        if color == 'random':
            color = rand_color(style='hex')            
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
            raise ColorError("Only 8bit ints are supported for now")
        array = array / COLORTYPE[1] 

    if array.ndim == 3:
        return array 

    elif array.ndim == 2:
        logger.warn('%s color has been converted (1-channel to 3-channel RGB)'
                    % name)
        return skcol.gray2rgb(array)

    raise ColorError('%s must be 2 or 3 dimensional array!' % name )    


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


def rgb2uint(image, warnmsg=False):
    """ Returns color image as 8-bit unsigned (0-255) int.  Unsigned 8bit gray 
    values are safer to plotting; so enforced throughout pyparty."""
        # img 127 --> ubyte of 123... 
        # Try this:
        #	print c.grayimage.max(), c.image.max() * 255, img_as_uint(lena()).max()

    # DOES NOT CHECK IMAGE DIMENSIONS; LEAVES THAT TO CALLING OBJECT
    grayimg = img_as_ubyte( skcol.rgb2gray(image) )
    if warnmsg:
        if isinstance(warnmsg, str):
            logger.warn(warnmsg)
        else:
            logger.warn("3-Channel converted to 1-channel (gray).")
    return grayimg    


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


def rr_cc_box(rr_cc, pad=0):
    """ Center the rr_cc values in a binarized box."""

    rr, cc = rr_cc
    ymin, ymax, xmin, xmax = rr.min(), rr.max(), cc.min(), cc.max()

    if pad:
        pad=int(pad)
        ymin, xmin = ymin - pad, xmin - pad
        ymax, xmax = ymax + pad, xmax + pad     

    # Center rr, cc mins to 0 index
    rr_trans = rr - ymin
    cc_trans = cc - xmin
    rr_cc_trans = (rr_trans, cc_trans)   

    dx = (xmax-xmin) 
    dy = (ymax-ymin) 

    rect=np.zeros( (dy+1, dx+1), dtype='uint8' ) 
    rect[rr_cc_trans] = 1
    return rect   


def _parse_path(path):
    """ Validate a path; if None, set to cwd with timestamp."""

    if path==True:
        from time import time as tstamp
        dirname, basename = os.getcwd(), 'canvas_%.0f.png' % tstamp()
        path = op.join(dirname, basename)
        logger.warn("Saving to %s" % path)
    path = op.expanduser(path)
    if op.exists(path):
        raise UtilsError('Path exists: "%s"' % path)    

    # PIL raises ambiguous KeyError 
    if not op.splitext(path)[1]:
        raise UtilsError("Please add an extension to save path")

    return path


def _parse_ax(*args, **kwargs):
    """ Parse plotting *args, **kwargs for an AxesSubplot.  This allows for
    axes and colormap to be passed as keyword or position. 
    Returns AxesSubplot, colormap, kwargs with *args removed"""

    axes = kwargs.pop('axes', None)       
    cmap = kwargs.get('cmap', None)

    if not axes:
        indicies = [idx for (idx, arg) in enumerate(args) if isinstance(arg, Subplot)]
        if len(indicies) < 1:
            axes = None
        elif len(indicies) > 1:
            raise UtilsError("Multiple axes not understood")
        else:
            args = list(args)
            axes = args.pop(indicies[0])      

    if args and not cmap:
        if len(args) > 1:
            raise UtilsError("Please only pass a colormap and/or Axes"
                             " subplot to Canvas plotting")
        elif len(args) == 1:
            kwargs['cmap'] = args[0]            

    # If string, replace cmap with true cmap instance (used by show())
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
        if isinstance(cmap, str):
            if cmap != 'pbinary' and cmap != 'pbinary_r': #special canvas word
                kwargs['cmap'] = cm.get_cmap(cmap)    

    return axes, kwargs

# showim(img, ax)
def showim(image, *args, **kwargs):
    """ Similar to imshow with a few more keywords"""

    if not isinstance(image, np.ndarray):
        raise UtilsError("First argument to showim() must be an ndarray/image, "
                         "got %s instead." % type(image))

    title = kwargs.pop('title', None)
    axes, kwargs = _parse_ax(*args, **kwargs)

    if axes:
        axes.imshow(image, **kwargs)
    else:      # matplotlib API asymmetry
        axes = plt.imshow(image, **kwargs).axes        

    if title:
        axes.set_title(title)
    return axes    


def splot(*args, **kwds):
    """ Wrapper to plt.subplots(r, c).  Will return flattened axes and discard
    figure.  'flatten' keyword will not flatten if the plt.subplots() return
    is not itself flat.  If flatten=False and fig=True, standard plt.subplots
    behavior is recovered."""

    flatten = kwds.pop('flatten', True)
    _return_fig = kwds.pop('fig', False)

    fig, args = plt.subplots(*args, **kwds)

    # Seems like sometimes returns flat, sometimes returns list of lists
    # so either way I flatten    
    if not hasattr(args, '__iter__'):
        args = [args]

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

def _mod_closest(count, testrange=[3,4,5,6]):
    """ Computes n % count for n in range of values and returns n for 
    which the modulo was closest (ie only needed to increase n by 1 for n%3;
    however, may need to increase n by 2 to get n%4...  primarily used for
    selecting plot columns that minimize number of empty cols in multiplots.
    When difference is the same between several column values, returns lowest.
    EG if 3 and 6 have same modulo (for example to 12), 3 is returned.
    """
    score = []
    for j in testrange:
        val = count
        diff = 0
        while val % j != 0:
            val += 1        
            diff += 1
        if diff == 0:
            return j
        score.append((j,diff))
    score = sorted(score, key=operator.itemgetter(1))
    return score[0][0]

# Eventually update w/ gridspect
def multi_axes(count, **kwargs):
    """ """
    figsize = kwargs.pop('figsize', None)#, rcParams['figure.figsize'])
    ncols = kwargs.pop('ncols', 4)
        
    if count <= ncols:
        nrows = 1
        ncols = count

    else:  
#       ncols = _mod_closest(count)
        nrows = int(count/ncols)         
        if count % ncols: #If not perfect division
            nrows += 1
    
    if figsize:
        fig, axes = splot(nrows, ncols, figsize=figsize, fig=True)
    else:
        fig, axes = splot(nrows, ncols,fig=True)
        

    while len(fig.axes) > count:
        fig.delaxes(fig.axes[-1])
    return fig.axes, kwargs
        

def mem_address(obj):
    """ Return memory address string for a python object.  Object must have
    default python object __repr__ (ie it would look something like:
        <pyparty.tools.grids.CartesianGrid object at 0x3ba2fb0>
    The address is merely returned by string parsing. """
    try:
        out = obj.__repr__().split()[-1]
    except Exception as E:
        raise UtilsError("Failed to return memory address by string parsing.  "
                         "Recieved following message: %s" % E.message)
    else:
        return out.strip("'").strip('>')


def grayhist(img, *args, **histkwargs):
    """Plot an image along with its histogram and cumulative histogram.

    ADAPTED FROM SCIKIT IMAGE GALLERY
    http://scikit-image.org/docs/dev/auto_examples/plot_local_equalize.html

    Parameters
    ----------
    bins : (Number bins, defaults to 256)

    cdf : bool(False) or str(color)
        Plot cumulative distribution function over histogram.
    If cdf = color, interpreted as line color eg (cdf = 'r') 
    plots a red line for CDF.   

    lw / ls : CDF Line styles

    xlim : set (xs, xf) or "auto" 
        Return cropped histogram between x-limits.  If "auto", min and max
    brigntess of image are used.  

    Returns
    -------
    tuple : (n, bins, patches) or ([n0, n1, ...], bins, [patches0, patches1,...])

    Notes
    -----
    Unlike standard histogram, this returns axes rather than the
    histogram parameters.  Because this method changes api for xlim,
    IE user can prescribe xlimits through call signature, it is easier to just
    crop the image instead of changing the plot limits to account for the
    various cases.  Therefore, it would return output for cropped image
    histogram, which could lead to confusion.

    See matplotlib hist API for all plt.hist() parameters.
    http://matplotlib.org/api/pyplot_api.html
    """

    if img.ndim == 3:
        img = rgb2uint(img, warnmsg = True)

    # Histogram plotting kwargs
    bins = histkwargs.pop('bins', 256) #used several places
    cdf = histkwargs.pop('cdf', False)
    title = histkwargs.pop('title', None)
    histkwargs.setdefault('color', 'black')
    histkwargs.setdefault('alpha', 0.5)
    histkwargs.setdefault('orientation', 'vertical')

    # CDF line plotting kwargs
    lw = histkwargs.pop('lw', 2)    
    ls = histkwargs.pop('ls', '-')

    xlim = histkwargs.pop('xlim', None)

    # Set the range based on scikit image dtype range 
    # (not quite right for rgb)
    xmin, xmax = pp_dtype_range(img)

    if xlim:
        # ALSO SET VLIM FROM AUTO!
        if xlim =='auto':
            xlim = img.min(), img.max()

        rmin, rmax = xlim
        if rmin < xmin or rmax > xmax:
            raise UtilsError("Range %s out of bounds (%s, %s)" %
                             (xlim, xmin, xmax))
        else:
            xmin, xmax = xlim    

    raveled_img = img[(img >= xmin) & (img <= xmax)]

    if histkwargs['orientation'] == 'horizontal':
        raise UtilsError("horizontal orientation not supported.")

    axes, kwargs = _parse_ax(*args, **histkwargs)    

    # Matplotlib
    if not axes:
        fig, axes = plt.subplots()

    # Display histogram
    histout = axes.hist(raveled_img, bins=bins, **histkwargs)
    axes.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    axes.set_xlabel('Pixel intensity')

    # Display cumulative distribution
    if cdf:
        if cdf is not True:
            lcolor = cdf
        else:
            lcolor = 'r'
        ax_cdf = axes.twinx()
        img_cdf, bins = exposure.cumulative_distribution(img, bins)
        ax_cdf.plot(bins, img_cdf, color=lcolor, lw=lw, ls=ls)

    axes.set_xlim(xmin, xmax) #is necessary
    if title:
        axes.set_title(title)
    return axes


def rgbhist(img, *args, **kwargs):
    """ See imagej version """
    if img.ndim == 2:
        img = skcol.gray2rgb(img) 
        logger.warn("Converting 1-channel gray image to rgb")

    bins = histkwargs.pop('bins', 256) #used several places
    cdf = histkwargs.pop('cdf', False)

    axes, kwargs = _parse_ax(*args, **histkwargs) 
    if not axes:
        fig, axes = plt.subplots()    

    # MAYBE STILL USE DTYPE FROM SKIMAGE IN CASE USERS PASS THEIR OWN RGB
    # IMAGE IN HERE OUTISDE OF PYPARTY?
    xmin, xmax = 0, 1

    # Can probably just call histograam 3 times w/ color arg
    raise NotImplementedError


def pp_dtype_range(img):
    """ Similar to skimage.utils.dtype_range, returns upper and lower limits
    on image of 1-channel and 3-channel.  Can't use skimage because it
    allows for negative floats, which we avoid in 3-channel images."""

    if img.dtype == 'bool':
        xmin, xmax = False, True
    if img.ndim == 2:
        xmin, xmax = 0, 255
    elif img.ndim == 3:
        xmin, xmax =  (0,0,0), (1,1,1) 
    return xmin, xmax

# ------- ZOOMING AND CROPPING 


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
                             ' image X range (%s, %s)' % (xi, xf, 0, img_xf))

    for y in (yi, yf):
        if y < 0 or y > img_yf:
            raise UtilsError('Cropping bounds (%s, %s) exceed'
                             ' image Y range (%s, %s)' % (yi, yf, 0, img_yf))

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

def zoom(image, coords, *imshowargs, **imshowkwds):
    """
    Plot zoomed-in region of rectangularly cropped image'

    Parameters
    ----------
    image: a ndarray
    coords : (xi, yi, xf, yf)
        length-4 iterable of crop coordiantes corresponding 
    axes : None
        Optionally pass in a matplotlib axes instance.
    *imshowargs, **imshowkwds : imshow() args

    Returns
    -------
    Matplotlib Axes
        This is the output of imshow(image, *imshowargs, **imshowkwds)

    Notes
    -----
    Simple wrapper that calls crop, then imshow() on the cropped image.

    Examples
    --------
    >>> from skimage import data
    >>> lena = img_as_float(data.lena())
    >>> zoom(lena, (0,0,400,300), 'gray');
    """    
    
    axes, kwargs = _parse_ax(*imshowargs, **imshowkwds)	
    if not axes:
        fig, axes = plt.subplots()

    if len(coords) != 4:
        raise UtilsError("Coordinates must be lenth four iterable of form"
                         "(xi, yi, xf, yf).  Received %s" % coords)

    xi, yi, xf, yf = coords
        
    cropped_image = crop(image, coords) 
    axes.imshow(image, *imshowargs, **imshowkwds)
    axes.set_xlim(xi, xf)
    axes.set_ylim(yf, yi)
    return axes
    

def zoomshow(image, coords, *imshowargs, **imshowkwds):
    """
    Plot full and cropped image side-by-side. 
    Draws a rectangle on full image to show zooming coordinate.

    Parameters
    ----------
    image: a ndarray
    coords : (xi, yi, xf, yf)
        lenngth-4 iterable with coordiantes corresponding to rectangle corners
    in order (xi, yi, xf, y

    *imshowargs, **imshowkwds : plotting *args, **kwargs
         Passed directly to matplotlib imshow() after removing special keywords
    (SEE NOTES)

     Returns
     -------
     cropped_image, (plots) : tuple
    image, (ax_full, ax_zoomed) 

     Notes
     -----
     Returns both the cropped image and the plots for flexibility.  Plots 
     are returned in this manner to allow user to further draw on them before
     calling show().

     Rectangle has special plotting keywords- "lw", "ls", "color", "orient"

     Examples
     --------
     >>> from skimage import data
     >>> lena = img_as_float(data.lena())
     >>> zoomshow(lena, (0,0,400,300), plt.cm.gray, orient='v', color='r');

    """

    # Pop keywords for rectangle
    lw = imshowkwds.pop('lw', '2')
    ls = imshowkwds.pop('ls', '-')
    color = imshowkwds.pop('color', 'y')
    orient = imshowkwds.pop('orient', 'h')

    if orient in ['h', 'horizontal']:
        subshape = {'nrows':1, 'ncols':2}
    elif orient in ['v', 'vertical']:
        subshape = {'nrows':2, 'ncols':1}
    else:
        raise UtilsError('Plot orientation "%s" not understood' % orient)

    # Normalize coordinates for axhline/axvline
    img_ymax, img_xmax = _get_xyshape(image)

    if len(coords) != 4:
        raise UtilsError("Coordinates must be lenth four iterable of form"
                         "(xi, yi, xf, yf).  Received %s" % coords)

    xi, yi, xf, yf = coords

    xi_norm, xf_norm = xi / img_xmax, xf / img_xmax
    yi_norm, yf_norm = (img_ymax - yi) / img_ymax, \
        (img_ymax - yf) / img_ymax

    f, (ax_full, ax_zoomed) = plt.subplots(**subshape)

    ax_full.imshow(image, *imshowargs, **imshowkwds)      
    cropped_image = crop(image, coords) 
    ax_zoomed.imshow(image, *imshowargs, **imshowkwds)

    ax_zoomed.set_xlim(xi, xf)
    ax_zoomed.set_ylim(yf, yi) #Y REVERSED

    # Add rectangle
    ax_full.axhline(y=yi, xmin=xi_norm, xmax=xf_norm, 
                    linewidth=lw, color=color, ls=ls)
    ax_full.axhline(y=yf, xmin=xi_norm, xmax=xf_norm, 
                    linewidth=lw, color=color, ls=ls)
    ax_full.axvline(x=xi, ymax=yi_norm, ymin=yf_norm, 
                    linewidth=lw, color=color, ls=ls)
    ax_full.axvline(x=xf, ymax=yi_norm, ymin=yf_norm, 
                    linewidth=lw, color=color, ls=ls)


    return cropped_image, (ax_full, ax_zoomed)

if __name__ == '__main__':
    warpedbg = np.random.randint(0, 255, size=(500,500) )
    zoomshow(warpedbg, (40,40,333,333))
#    grayhist(warpedbg, xlim='auto', title='Gray Histogram',cdf=True);