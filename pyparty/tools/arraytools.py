import numpy as np
import math
from pyparty.utils import UtilsError

class ArrayUtilsError(UtilsError):
    """ """

class ArraySetError(ArrayUtilsError):
    """ Reserved for set operations """

def astype_rint(array):
    """ Converts ndarray to int, but rounds integers.  Sometimes don't want 
    accurate rounding, such as with grids"""
    return np.around(array).astype(int)

def column_array(len2iter):
    """ Takes an iterable of 2 arrays [(x, x2, x3) (y, y2, y3)] and returns
    array N x 2 (ie 2 columns)"""
    return np.array(zip(*len2iter))

def boolmask(ndarray):
    """ Given an array, converted to binary and indiciesa evaluating to true
    are used.  Most useful for outlines and other empty arrays. """
    xs, ys = np.where(ndarray.astype(bool))
    return np.array(zip(xs, ys)).T
    #OUTPUT IS 2,N array

# REPLACE
def unzip_array(pairs):
    """ Inverse of np.array(zip(x,y)). Rerturn unzipped array of pairs:
    (1,2), (25,5) --> array(1,25), array(25,25)."""
    return np.array( zip(*(pairs) ) )


def add_xy(ndarray, xy_pair):
    """ Add (x,y) to ndarray.  Array can be of dimentsion N,2 or 2,N """
    try:
        return ndarray + xy_pair
    except ValueError:
        return (ndarray.T + xy_pair).T
           

def rmeanint(x): return int(round(np.mean(x), 0))
    

def findcenter(ndarray):
    """ Requires (2,N) input shape.  If (N,2), result will be len(N) instead of
    Len(2)"""
    return tuple(map(rmeanint, ndarray))
    

def meancenter(ndarray, center=None):
    """ Subtract center (x,y) from N,2 or 2,N array. Center can be passed
    manually."""

    if not center:
        center = findcenter(ndarray)
    cx, cy = center
    # Adding negative center subtracts
    return add_xy(ndarray, (-cx, -cy)) 


def rotate(ndarray, theta, center=None, **kwds):
    """ Meancenter and rotate.  Mean centers if center not passed"""
    ndarray = meancenter(ndarray, center)
    rotated = rotate_vector(ndarray, theta, **kwds)
    return rotated + center


def translate(ndarray, r, theta=0.0):
    """ Translate along a vector with magnitude r and angle theta (IN DEGREES). 
    Merely gets x,y coords of r, theta vector adds to the array element wise."""
    theta = math.radians(theta)
    return add_xy(ndarray, (r*math.cos(theta), r*math.sin(theta)) )


def rotate_vector(ndarray, theta, style='degrees', rint='up'):
    """ Rotate an array ocounter-clockwise through theta.  rint rounds output 
    to integer; up rounds normally, down does int rounding (ie rounds down).  
    ARRAY MUST BE MEAN-CENTERED if rotating in place.
    
    ndarray may be xy pairs [(x1,y1),(x2,y2)] or N,2 matrix."""

    if style == 'degrees':
        theta = math.radians(theta)
        
    costheta, sintheta = math.cos(theta), math.sin(theta)
    
    rotMatrix = np.array([
        [costheta, -sintheta],  
        [sintheta,  costheta]
                     ])
    r_array = np.dot(ndarray, rotMatrix)
    
    if rint:
        if rint =='up':
            r_array = np.rint(r_array)
        r_array = r_array.astype('int', copy=False) #saves memory
    return r_array

def to_spherical(r):
    """ Convert xyz vector a single row of three elements to spherical """
    x,y,z = r
    r = math.sqrt(x**2 + y**2 + z**2)              
    theta = math.atan2(y,x)                          
    phi = math.atan2(z,math.sqrt(x**2 + y**2))     
    return r, theta, phi

def array2sphere(xyz_array):
    """ Cartesion to spherical coords.  xyz_array must be (N,2) or (N,3)!!!
    EG  [( x, y, z          or [ (x1, y1), (x2, y2)]
          x2, y2, z2)]
    """
    xdim, ydim = xyz_array.shape[0:2]
    if ydim == 3:
        return np.apply_along_axis(to_spherical, 1, xyz_array)
    # Return only r/theta, add column zeros for z
    elif ydim ==2:    
        out = np.zeros((xdim, 3))
        out[..., 0:2] = xyz_array #DONT NEED TO COPY, RIGHT?
        return np.apply_along_axis(to_spherical, 1, out)[..., 0:2]
    else:
        raise ArrayUtilsError("xyz_array must be of shape (N,2) or (N,3) (ie "
            "rows of xy or xyz vectors), recived: %s" % str(xyz_array.shape))
    
def nearest(array, value):
    '''Find nearest value in an array, return index and array value'''
    idx=(np.abs(array-value)).argmin()
    return idx, array[idx]  
    
# Set operations
def _parse_set(array1, array2):
    """ Ensure arrays are of same type and shape; no attempt to correct."""
    ndim1, ndim2 = array1.ndim, array2.ndim
    type1, type2 = array1.dtype, array2.dtype
    if ndim1 != ndim2:
        raise ArraySetError("Ndim mismatch: %s vs. %s" % (ndim1, ndim2))
    if type1 != type2:
        raise ArraySetError("Dtype mismatch: %s vs. %s" % (type1, type2))
    

def intersect(array1, array2, bgout=None):
    """ Return array1 only where pixel values are identical to array2."""
    _parse_set(array1, array2)
    return (array1 == array2) * array1    
   
    
def differ(array1, array2, bgount=None):
    _parse_set(array1, array2)
    return (array1 != array2) * array1
    