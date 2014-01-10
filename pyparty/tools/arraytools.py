import numpy as np
import math

def boolmask(ndarray):
    """ Given an array, converted to binary and indiciesa evaluating to true
    are used.  Most useful for outlines and other empty arrays. """
    xs, ys = np.where(ndarray.astype(bool))
    return np.array(zip(xs, ys)).T
    #OUTPUT IS 2,N array

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
    
#def order_cords(ndarray, center, clockwise=True):
    #""" Array must be (2,N) !!.  This will work either way, but if (N,2),
    #arctan function will only return a len(2) array"""
    #centered = meancenter(ndarray, center)
    #centered.sort(
        

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
    return add_xy(ndarray, (math.cos(theta), math.sin(theta)) )


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