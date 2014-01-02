from __future__ import division
import logging
import numpy as np
from math import atan, degrees, cos, sin, radians

import skimage.draw as draw
from traits.has_traits import CHECK_INTERFACES
from traits.api import HasTraits, Property, provides, Bool, Int, Array,\
     Float, Str

from abstract_shape import Particle, CenteredParticle, ParticleInterface, Segment
from pyparty.config import RADIUS_DEFAULT, CENTER_DEFAULT, XRADIUS, YRADIUS, \
     XMID, YMID, BEZIERWEIGHT

logger = logging.getLogger(__name__) 
CHECK_INTERFACES = 2 # 2-error, 1-warn, 0-pass

def rotate_vector(array, angle, style='degrees'):
    import math
    if style == 'degrees':
        angle = math.radians(angle)
        cos, sin = math.cos, math.sin
            
    rotMatrix = np.array([[cos(angle), -sin(angle)],  
                   [sin(angle),  cos(angle)]])    
    
    return np.dot(array, rotMatrix)

class ShapeError(Exception):
    """ """

@provides(ParticleInterface)     
class Circle(CenteredParticle):
    """ description
    
    Attributes
    ----------
    """    
    ptype=Str('circle')
    radius = Int(RADIUS_DEFAULT) #in pixels (<2 causes errors w/ properties)
    	    
    #http://scikit-image.org/docs/dev/api/skimage.draw.html#circle
    def _get_rr_cc(self):
        
        if self.fill:
            return draw.circle(self.cy, self.cx, self.radius)
        else:
            return draw.circle_perimeter(self.cy, self.cx, self.radius)

@provides(ParticleInterface)             
class Ellipse(CenteredParticle):
    """ Orientation supported only for perimeter in scikit """

    ptype=Str('ellipse')    

    yradius = Int(YRADIUS)
    xradius = Int(XRADIUS)

    def _get_rr_cc(self):
    
        if self.fill:
            return draw.ellipse(self.cy, self.cx, self.yradius, 
                                          self.xradius)        
        else:
            return draw.ellipse_perimeter(self.cy, self.cx, self.yradius, 
                                          self.xradius)
@provides(ParticleInterface)        
class Line(Segment):
    """ Line with width """
 
    ptype = Str('line')
    
    width = Int(5) #line width in pixels
    
    @property
    def slope(self):
        """ Return the slope in degrees """
        return degrees(atan( (self.yend - self.ystart) / 
                             (self.xend - self.xstart) ) )  
    @property
    def theta_perp_slope(self):
        """ Return the angle of a line perpendicular to the slope"""
        try:
            slope = self.slope
        except ZeroDivisionError:
            return 0.0
        
        if slope > 0:
            return slope + 90.0
        return slope - 90
    
    def _get_rr_cc(self):
#        return draw.line(self.ystart, self.xstart, self.yend, self.xend)
        lines = []
        theta = radians(self.theta_perp_slope)
        for i in range(1,self.width):
            xs = self.xstart + int(round(i * cos(theta), 0))
            ys = self.ystart + int(round(i * sin(theta), 0))
            xe = self.xend + int(round(i * cos(theta), 0))
            ye = self.yend + int(round(i * sin(theta), 0))            
            
            lines.append(draw.line(xs, ys, xe, ye))
        rr, cc = zip(*(l for l in lines))
        return (np.concatenate(rr), np.concatenate(cc))    

@provides(ParticleInterface)
class BezierCurve(Segment):
    """  """

    ptype=Str('bezier')   
    
    ymid = Int(YMID)
    xmid = Int(XMID)
    
    weight = Float(1.0) #Middle control point weight (sensible defualt value?)
    
    def _get_rr_cc(self):
        return draw.bezier_curve(self.ystart, self.xstart, self.ymid, 
                    self.xmid, self.yend, self.xend, weight=self.weight)

@provides(ParticleInterface)    
class Polygon(Particle):
    """ General polygon which uses only verticies to wrap skimage.draw.polygon
    
    Attributes
    ----------
    
    yverts : array
        Y-coords of verticies of polygon
        
    xverts : array
        X-coords of verticies of polygon

    Notes
    -----
    Polygon verticies ARE order dependent.  In this manner, interesting shapes
    like bowties can be drawn; however, providing verticies in the wrong order
    can result in null output, and otherwise confusing drawings.  Thus, various
    subclasses for common shapes (SQUARE, RECTANGLE) are available.
        
    """
    ptype = Str('polygon')
    
    yverts = Array
    xverts = Array
    
    # Default shape is a bowtie
    def _xverts_default(self):
        return np.array( (220, 220, 280, 280) )
    
    def _yverts_default(self):
        return np.array( (220, 280, 220, 280) )
    
    def __init__(self, *args, **kwargs):
        """ Initialize and run any coord validation imposed by shape."""
        super(Polygon, self).__init__(*args, **kwargs)    
        #self._validate_coords()
    
    def _validate_coords(self):
        """ shape restraints; general polygon has no such restriction"""
        pass
    
    def _get_rr_cc(self):
        print "HI", self.pairs
        print "ROT", rotate_vector(self.pairs, 30.0)
        return draw.polygon(self.yverts, self.xverts)      
    
    @property
    def pairs(self):
        """ Array of (x,y) pairs; can be passed directly into rotation."""
        return np.array(zip(self.xverts, self.yverts))
    
    @classmethod
    def from_xypairs(cls, sets, *args, **kwargs):
        """ Set verticies from list of (x, y) pairs """
        xv, yv = zip( *(xy for xy in sets) )
        return cls(xverts=xv, yverts=yv, *args, **kwargs)
    
    
@provides(ParticleInterface)
class Rectangle(Polygon):
    """ """
    ptype = Str('rectangle')    
    
    # Used to reduce boilerplate with subclasses
    _n_unique = 3 # unique values in set() of rectangle coords
    
    def _validate_coords(self):
        """ Subjugates coordinates to certain shape restraints """
        if len(self.xverts) != 4 or len(self.yverts) != 4:
            raise ShapeError("Rectange requires exactly 4 (x,y) verticies")
        
        xvset, yvset, nu = set(self.xverts), set(self.yverts), self._n_unique 
        
        if len(xvset) != nu or len(yvset) != nu or xvset != yvset:
            raise ShapeError("Rectangle must contain exactly %s unique values"
                            "eg (1,1), (1,5), (8,5), (8,1)  --> 1,5,8" % nu)         
        
    @classmethod
    def from_length_width(cls, center, length, width, *args, **kwargs):
        raise NotImplemented
    
    
@provides(ParticleInterface)
class Square(Rectangle):
    """ """
    ptype = Str('square')      
    _n_unique = 2  # _validate_coords() from rectange
    
    @classmethod
    def from_length_width(cls, center, length, width, *args, **kwargs):
        raise ShapeError("Please use from_length() to construct square")
        #if length != width:
            #raise ShapeError("Length and Width must match for rectangle.")
        #return from_length(cls, center, length)
    
    @classmethod
    def from_length(cls, center, length, *args, **kwargs):
        raise NotImplemented
   
        
if __name__ == '__main__':
    Polygon().rr_cc