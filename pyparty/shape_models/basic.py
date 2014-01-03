from __future__ import division
import logging
import math

import numpy as np

import skimage.draw as draw
from traits.has_traits import CHECK_INTERFACES
from traits.api import HasTraits, Property, provides, Bool, Int, Array,\
     Float, Str

from abstract_shape import Particle, CenteredParticle, ParticleInterface, Segment
from pyparty.utils import rotate_vector
from pyparty.config import RADIUS_DEFAULT, CENTER_DEFAULT, XRADIUS, YRADIUS, \
     XMID, YMID, BEZIERWEIGHT

logger = logging.getLogger(__name__) 
CHECK_INTERFACES = 2 # 2-error, 1-warn, 0-pass

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
            return draw.ellipse(self.cy, self.cx, self.yradius, self.xradius)        
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
        """ Return the slope in math.degrees """
        return math.degrees(math.atan( (self.yend - self.ystart) / 
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
        theta = math.radians(self.theta_perp_slope)
        for i in range(1,self.width):
            xs = self.xstart + int(round(i * math.cos(theta), 0))
            ys = self.ystart + int(round(i * math.sin(theta), 0))
            xe = self.xend + int(round(i * math.cos(theta), 0))
            ye = self.yend + int(round(i * math.sin(theta), 0))            
            
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