import numpy as np
from math import cos

class PatternError(object):
    """ """

def basic(cx, cy, d_pp, n=2, phi=0.0):
    """ Dimer, Trimer, Square grid 
    
    Attributes
    ----------
    cx, cy : int, int
        Center coordinates in pixels
        
    d_pp : float
        Particle-particle center distance.  
        
    n : int (2-4)
    
    phi : float
        Phase angle in degrees for orientation.
    
    Examples
    --------
    x -- x     x      x-x
             x - x    x-x
    
    Returns
    -------
    Verticies of dimer, trimer or square.
    
    Notes
    -----
    Center coordinate is not returned, unlike hexagonal element which does
    return its center coordinate.
    
    """
     
    if n == 2:
        thetas = ( 0., 180. )
    elif n == 3:
        thetas = ( 90., 210., 330. ) 
    elif n == 4:
        thetas = (45., 125., 225., 315. )
    else:
        raise PatternError('n must be 2,3,4; recieved %s' % n)
    
    thetas += phi
    r_pp = 0.5 * d_pp    

    cx = cx * r_pp * np.cos(thetas)
    cy = cy * r_pp * np.sin(theats)
    
    return zip(cx, cy)

        
def hexagonal(cx, cy, d_pp, phi=0.0):
    """ Returns a hexagonal skeleton (7 point honeycomb).
    
    Attributes
    ----------
    cx, cy : int, int
        Center coordinates in pixels
        
    d_pp : float
        Particle-particle center distance.  
        
    n : int (2-4)
    
    phi : float
        Phase angle in degrees for orientation.
    
    Returns
    -------
    coordinates : tuple
        Center coordinate (cx, cy) followed by hexagon verticies drawn a distance
        d_pp away.  
        
    Notes
    -----
    From center coordnate (cx, cy), goes out a distance d, returns c1.  Goes
    out a distance d dot 60 degrees, returns next point on hexagon.
    
    """
    
    thetas = np.linspace(0,360,7) + phi    
    cxs = cx + d_pp * np.cos(thetas)
    cys = cy + d_pp * np.sin(thetas)

    return [ (cx, cy) ] + zip(cxs, cyx)

    
