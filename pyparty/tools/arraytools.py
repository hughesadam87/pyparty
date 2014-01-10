import numpy as np
import math

def unzip_array(pairs):
    """ Inverse of np.array(zip(x,y)). Rerturn unzipped array of pairs:
    (1,2), (25,5) --> array(1,25), array(25,25)."""
    return np.array( zip(*(pairs) ) )

def meancenter(ndarray, center):
    """ Subtract center (x,y) from N,2 or 2,N array.  """
    try:
        return ndarray - center
    except ValueError:
        return (ndarray.T - center).T


def rotate(ndarray, center, theta, **kwds):
    """ Meancenter and rotate """
    ndarray = meancenter(ndarray, center)
    rotated = rotate_vector(ndarray, theta, **kwds)
    return rotated + center


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