#
# (C) Copyright 2013 George Washington University, D.C.
# All right reserved.
#
# This file is open source software distributed according to the terms in LICENSE.txt
#

import logging
import numpy as np

import skimage.draw as draw
from traits.has_traits import CHECK_INTERFACES
from traits.api import HasTraits, Property, implements, Bool, Int, Array,\
     Float, Str

from abstract_shape import Particle, CenteredParticle, ParticleInterface

logger = logging.getLogger(__name__) 
CHECK_INTERFACES = 2 # 2-error, 1-warn, 0-pass

class Circle(CenteredParticle):
    """ description
    
    Attributes
    ----------
    """
    # INTERFACE IS NOT WORKING
    implements(ParticleInterface)        
    
    ptype=Str('circle')
    radius = Int(20) #in pixels (<2 causes errors w/ properties)
    	    
    #http://scikit-image.org/docs/dev/api/skimage.draw.html#circle
    def _get_rr_cc(self):
        
        if self.fill:
            return draw.circle(self.cy, self.cx, self.radius)
        else:
            return draw.circle_perimeter(self.cy, self.cx, self.radius)
        
class Ellipse(CenteredParticle):
    """ """

    implements(ParticleInterface)        
    ptype=Str('ellipse')    

    yradius = Int(2)
    xradius = Int(2)

    def _get_rr_cc(self):
    
        if self.fill:
            return draw.ellipse(self.cy, self.cx, self.yradius, 
                                          self.xradius)        
        else:
            return draw.ellipse_perimeter(self.cy, self.cx, self.yradius, 
                                          self.xradius)
    

class Line(Particle):
    """ """

    implements(ParticleInterface)        
    ptype=Str('line')    
    
    ystart = Int(0) #start position row
    xstart = Int(0)
    yend = Int(2)
    xend = Int(2)
    
    def _get_rr_cc(self):
        return draw.line(self.ystart, self.xstart, self.yend, self.xend)


class BezierCurve(Line):
    """
    Notes
    -----
    Subclassing from line to share attributes (xstart, ystard) but the bezier
    curve is not a subclass of a line per-se.
    """

    implements(ParticleInterface)        
    ptype=Str('bezier')   
    
    ymid = Int(1)
    xmid = Int(1)
    
    weight = Float(1.0) #Middle control point weight (sensible defualt value?)
    
    def _get_rr_cc(self):
        return draw.bezier_curve(self.ystart, self.xstart, self.ymid, 
                    self.xmid, self.yend, self.xend, weight=self.weight)
    
class Polygon(Particle):
    """ description
    
    Attributes
    ----------
    
    yccords : array
        Y-coords of verticies of polygon
        
    xcoords : array
        X-coords of verticies of polygon
    """
    ptype = Str('polygon')
    
    ycoords = Array()
    xcoords = Array()
    
    def _ycoords_default(self):
        return np.array( (1,7,1,4) )
    
    def _xcoords_default(self):
        return np.array( (1,2,8,1) )
    
    def _get_rr_cc(self):
        return draw.polygon(self.ycoords, self.xcoords)                                           

