from __future__ import division
import logging
import numpy as np
import skimage.draw as draw
import math

from traits.has_traits import CHECK_INTERFACES
from traits.api import Array, Str, Property, Tuple
import matplotlib.patches as mpatch

from pyparty.shape_models.basic import LineSegment
from pyparty.config import XVERTS, YVERTS, RECTLEN, RECTWID, LINEWID, LINELEN
from pyparty.config import CENTER_DEFAULT as CDEF
from pyparty.tools.arraytools import rotate_vector, unzip_array, rotate, \
     rmeanint
from pyparty.shape_models.abstract_shape import ShapeError, FastOriented, \
    CenteredParticle

logger = logging.getLogger(__name__) 

def rint(x): return int(round(x,0))

def _invalid_constructor(attr, cname='Polygon', **kwargs):
    """ Raises either PolyConstructError or NotImplementedError depending on
    which keywords are passed.  Exists to reduced boilerplate.  Tried to 
    integrate as class/static method but it added too much overhead"""

    if attr in kwargs:
        raise PolyConstructError('Cannot construct "%s" from parameter "width"' % 
                                 cname)
    else:
        raise NotImplementedError   
    

class PolygonError(Exception):
    """  """
    
class PolyConstructError(PolygonError):
    """ Raised when a polygon has a bad-constructor call and wants to alert user
    instead of going to the next call.  For example, calling square with
    from_length_width() """
        
class Polygon(FastOriented):
    """ General CLOSED polygon which uses only verticies to wrap 
    skimage.draw.polygon.  Closed is enforced through _validate_verts() and
    is used in call to mpatch.Polygon
    
    Attributes
    ----------
    
    _yverts : array
        Y-coords of verticies of polygon
        
    _xverts : array
        X-coords of verticies of polygon

    Notes
    -----
    Polygon verticies ARE order dependent.  In this manner, interesting shapes
    like bowties can be drawn; however, providing verticies in the wrong order
    can result in null output, and otherwise confusing drawings.  Thus, various
    subclasses for common shapes (SQUARE, RECTANGLE) are available.
    
    There are several constructors for polygons, instead of relying on the need
    to pass 
        
    """
    ptype = Str('polygon')
    
    _yverts = Array
    _xverts = Array
    
    # INTENTIONALLY LEAVING SETTING OF CX DISABLED FOR NOW
    center = Property(Tuple)#, depends_on='_xverts, _yverts')
    unrotated_rr_cc = Property(Array)#, depends_on='orientation, _xverts, _yverts')
    
    # MEMOIZING CAUSES ISSUE WITH COPY.DEEPCOPY (copy metaparticle)
    
    def __init__(self, *args, **kwargs):
        """ Initialize and run any coord validation imposed by shape."""

        self._xverts = kwargs.pop('xverts')
        self._yverts = kwargs.pop('yverts')
        self._validate_verts()        
        super(Polygon, self).__init__(*args, **kwargs)
        
    
    def _validate_verts(self):
        """ Verticies are validated once upon initialization for any polygon.
            Verticies are immutable to user."""
        xneg = self._xverts[self._xverts < 0]
        yneg = self._yverts[self._yverts < 0]
        
        if len(xneg) != 0 or len(yneg) != 0:
            raise PolygonError("Polygon has one ore more negative verticies."
                  " This can result in rendering problems.")
    

    def _get_center(self):
        """ Get mean cx, cy from _xverts, _yverts """
        return ( rmeanint(self._xverts), rmeanint(self._yverts) )

    
    def _get_rr_cc(self):
        """ Very similar to Particle_get_rr_cc(), except rotates verticies then
        draws rr_cc instead of rotating rr_cc itself."""
        
        if self.phi % 360.0 == 0.0:
            xs, ys = self._xverts, self._yverts

        else:
            center = self.center            
            centered = self.xymatrix - center
            rr_cc_rot = rotate_vector(centered, self.phi, rint='up')
            xs, ys = (rr_cc_rot + center).T            

        return draw.polygon(ys, xs)      
    
    
    #May be useful for users, not used by class
    @property
    def xysets(self):
        """ Array of (x,y) pairs; can be passed directly into rotation.  
            Note that unzipping is just zip(*(pairs) )"""
        return np.array(zip(self._xverts, self._yverts))
    
    @property
    def xymatrix(self):
        """ N, 2 array, columns are x and y arrays """
        return np.array( (self._xverts, self._yverts) ).T
    
    # Prevents users from changing verticies
    @property
    def xverts(self):
        return self._xverts
    
    @property
    def yverts(self):
        return self._yverts      
    
    @yverts.setter
    def yverts(self, v):
        raise PolygonError("Verticies are polygon are immutable; please"
                           "initialialize a new polygon.")
    
    @xverts.setter
    def xverts(self, v):
        raise PolygonError("Verticies are polygon are immutable; please"
                           "initialialize a new polygon.")        
    
    def as_patch(self, *args, **kwargs):
        """ Explictly rotate vertex coords for patch. """
        rotated_coords = rotate(self.xymatrix, theta=self.phi, center=self.center)
        return mpatch.Polygon(rotated_coords, closed=True, **kwargs)
    
    @classmethod
    def from_verts(cls, *args, **kwargs):
        kwargs.setdefault('xverts', XVERTS)
        kwargs.setdefault('yverts', YVERTS)
        return cls(*args, **kwargs)     
    
    @classmethod
    def from_xypairs(cls, *args, **kwargs):
        """ Set verticies from list of (x, y) pairs """
        sets = kwargs.pop('sets')
        xv, yv = unzip_array(sets)
        return cls(xverts=xv, yverts=yv, *args, **kwargs)
               
    @classmethod
    def auto_init(cls, *args, **kwargs):
        """ For Particles with multiple constructors (eg polygons), this method
        will infer which constructor to call based on the keywords.  For classes
        with only one constructor (ie __init__), doesn't do anything. 
        """
        
        # Define all of the auto_init methods for all polygons, instead of for
        # each class
        
        for meth in ['from_length_width', 'from_length', 
                     'from_xypairs', 'from_verts']:
            try:
                return getattr(cls, meth)(*args, **kwargs)
            except Exception as EXC:
                if isinstance(EXC, PolyConstructError) or isinstance(EXC, PolygonError):
                    logger.critical("FAILED AT METHOD %s" % meth)
                    raise EXC
                else:
                    logger.info("Passing on polygon constructor ERROR %s" % EXC)

        raise EXC
    
    # If user tries to construct polygon with length/width
    @classmethod
    def from_length_width(cls, *args, **kwargs):       
        _invalid_constructor('width', cname=cls.__name__, **kwargs)  
        
    @classmethod
    def from_length(cls, *args, **kwargs):       
        _invalid_constructor('length', cname=cls.__name__, **kwargs)          
                
    
class Triangle(Polygon):
    """ Not very restrictive; only imposes 3-coordinates. """
    ptype = Str('triangle')
    
    def _validate_verts(self):
        """ """
        # super already validates x, y same length, so only check x
        super(Triangle, self)._validate_verts()
        if len(self._xverts) != 3:
            raise PolygonError('Triangle requires 3 verticies!')

        
    @classmethod
    def from_length(cls, center=CDEF, length=RECTLEN, *args, **kwargs):
        """ Equilateral triangle! """
        cx, cy = center
        center_to_corner = length / math.sqrt(3)     
        
        # Draw four corners from bottom left, clockwise, relative to cx+l/2
        thetas = np.deg2rad( (210, 90, -30) )
        xs = cx + (center_to_corner * np.cos(thetas))
        ys = cy + (center_to_corner * -np.sin(thetas)) #for upright orientation
        return cls( xverts=xs, yverts=ys, *args, **kwargs )
    
    
class Rectangle(Polygon):
    """ Longest axis is considered length """
    ptype = Str('rectangle')    
    
    # Used to reduce boilerplate with subclasses
    _n_unique = 4 # unique values in set() of rectangle coords
            
    def _validate_verts(self):
        """ Ensure length of coords passed is 4, and they contain 3 unique values"""
        super(Rectangle, self)._validate_verts()

        if len(self._xverts) != 4 or len(self._yverts) != 4:
            raise ShapeError("Rectange requires exactly 4 (x,y) verticies")
        
        xvset, yvset, nu = set(self._xverts), set(self._yverts), self._n_unique 
        
        #if len(xvset) != nu or len(yvset) != nu or xvset != yvset:
            #raise ShapeError("Rectangle must contain exactly %s unique values"
                            #" EG: (1,2), (1,3), (8,3), (8,2)  --> 1,5,8" % nu)   

    @classmethod
    def from_length_width(cls, center=CDEF, length=RECTLEN, width=RECTWID, 
                          *args, **kwargs):
        cx, cy = center
        half_length, half_width = rint(length / 2) , rint(width / 2)        
        
        # Draw four corners from bottom left, clockwise, relative to cx+l/2
        thetas = np.deg2rad( ( 225, 135, 45, 315) )
        xs = cx + (half_length * np.cos(thetas))
        ys = cy + (half_width * np.sin(thetas))
        return cls( xverts=xs, yverts=ys, *args, **kwargs )

    
    # Need some semblance of order in the pairs
    @property
    def diagonal(self):
        """ diagonal length """
        l, s = Rectangle.long_small(self.xverts, self.yverts)
        return l.max() - s.min()
    
    @property
    def length(self):
        """ longest side """
        longest = Rectangle.long_small(self.xverts, self.yverts)[0]
        return longest.max() - longest.min()
    
    @property
    def width(self):
        """ shortest side """
        shortest = Rectangle.long_small(self.xverts, self.yverts)[1]
        return shortest.max() - shortest.min()    

    @staticmethod
    def long_small(a, b):
        """ Find longest value separtion in pairs. """
        amax, amin, bmax, bmin = a.max(), a.min(), b.max(), b.min()
        if (amax - amin) > (bmax - bmin):
            longest = a
            smallest = b
        else:
            longest = b
            smallest = a
        return (longest, smallest)    
    
    
class Square(Rectangle):
    """ """
    ptype = Str('square')      
    _n_unique = 2  # _validate_verts() from rectangle
    
    @classmethod
    def from_length_width(cls, *args, **kwargs):       
        _invalid_constructor('width', cname=cls.__name__, **kwargs)   
    
    @classmethod
    def from_length(cls, center=CDEF, length=RECTLEN, *args, **kwargs):
        """ Pass to rectangle's constructor """
        return super(Square, cls).from_length_width(center, length, length, *args, **kwargs)


class Line(Rectangle):
    """ Line is just a thin rectangle """

    ptype = Str('line')       
    
    @classmethod
    def from_length(cls, center=CDEF, length=LINELEN, *args, **kwargs):
        """ Use LINEWID as default width """
        return cls.from_length_width(center, length, LINEWID, 
                                     *args, **kwargs)
    
    @classmethod
    def from_length_width(cls, center=CDEF, length=LINELEN, width=LINEWID, *args, **kwargs):
        try:
            rect = super(Line, cls).from_length_width(center, length, width,
                                                *args, **kwargs)
        except Exception as E1:
            cx, cy = center
            half_length, half_width = rint(length / 2) , rint(width / 2)  

            try:       
                rect = LineSegment(xstart=cx-half_length, xend = cx+half_length, 
                    ystart=cy-half_width, yend=cy+half_width,  width=width, 
                    *args, **kwargs)                
    
            except Exception as E2:
                raise ShapeError('Could not construct Line from Rectangle or '
                'LineSegment constructors.  From Rectangle: "%s"  '
                'From LineSegment: "%s"' % (E1.message, E2.message))
        
            else:
                logger.warn("Creating thin from Segment instead of Polygon")
                
        return rect
      
        
if __name__ == '__main__':
    Line.from_length_width((50,50), 5, 5, orientation=50.0)
    Triangle.from_length((50,50), 10)
    Square.from_length(length=50, center=(200,200), orientation=23.0)
    