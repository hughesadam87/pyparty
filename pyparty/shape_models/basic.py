from __future__ import division
import logging
import math

import numpy as np
import matplotlib.patches as mpatch
import skimage.draw as draw
from traits.api import HasTraits, Property, Int, Array,\
     Float, Str, cached_property

from abstract_shape import Particle, CenteredParticle, Segment
from pyparty.config import RADIUS_DEFAULT, CENTER_DEFAULT, XRADIUS, YRADIUS, \
     XMID, YMID, BEZIERWEIGHT

logger = logging.getLogger(__name__) 

class Circle(CenteredParticle):
    """ skimage.draw.circle wrapper
    
    Attributes
    ----------
    """    
    ptype = Str('circle')
    radius = Int(RADIUS_DEFAULT) #in pixels (<2 causes errors w/ properties)
    unrotated_rr_cc = Property(Array, depends_on='center, radius')    
    
    #http://scikit-image.org/docs/dev/api/skimage.draw.html#circle
    def _get_unrotated_rr_cc(self):
        return draw.circle(self.cy, self.cx, self.radius)
    
    def _get_rr_cc(self):
        """ Overload to prevent rotating a symmetric circle """
        return self.unrotated_rr_cc
    
    def as_patch(self, *args, **kwargs):
        return mpatch.Circle((self.cx, self.cy), self.radius, **kwargs)
        

class Ellipse(CenteredParticle):
    """ skimage.draw.ellipse wrapper """

    ptype=Str('ellipse')    

    yradius = Int(YRADIUS)
    xradius = Int(XRADIUS)
    
    unrotated_rr_cc = Property(Array, depends_on='center, xradius, yradius')    

    def _get_unrotated_rr_cc(self):
        return draw.ellipse(self.cy, self.cx, self.yradius, self.xradius)   
    
    def as_patch(self, *args, **kwargs):
        return mpatch.Ellipse((self.cx, self.cy), self.xradius, self.yradius,
                              angle=self.phi, **kwargs)    


class LineSegment(Segment):
    """ Line with width """
 
    ptype = Str('line')
    width = Int(5) #line width in pixels
    
    @property
    def slope(self):
        """ The slope in degrees """
        return math.degrees(math.atan( (self.yend - self.ystart) / 
                             (self.xend - self.xstart) ) )  
    @property
    def theta_perp_slope(self):
        """ Angle of a line perpendicular to the slope"""
        try:
            slope = self.slope
        except ZeroDivisionError:
            return 0.0
        
        if slope > 0:
            return slope + 90.0
        return slope - 90
    
    def _get_unrotated_rr_cc(self):
        #return draw.line(self.ystart, self.xstart, self.yend, self.xend)
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
    

class BezierCurve(Segment):
    """ skimage.draw bezier curves """

    ptype=Str('bezier')   
    
    ymid = Int(YMID)
    xmid = Int(XMID)
    weight = Float(1.0) #Middle control point weight (sensible default value?)    
    
    unrotated_rr_cc = Property(Array,
        depends_on='ystart, xstart, yend, xend, ymid, xmid, weight') 
        
    def _get_unrotated_rr_cc(self):
        return draw.bezier_curve(self.ystart, self.xstart, self.ymid, 
                    self.xmid, self.yend, self.xend, weight=self.weight)
