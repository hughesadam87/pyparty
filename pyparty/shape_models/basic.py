#
# (C) Copyright 2013 George Washington University, D.C.
# All right reserved.
#
# This file is open source software distributed according to the terms in LICENSE.txt
#

import logging
import math

import skimage.draw as draw
from traits.has_traits import CHECK_INTERFACES
from traits.api import HasTraits, Property, implements, Bool, Int, Array, Str

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
    radius = Int(2) #in pixels (<2 causes errors w/ properties)
    fill = Bool(True)
    	    
    #http://scikit-image.org/docs/dev/api/skimage.draw.html#circle
    def _get_rr_cc(self):
        
        if self.fill:
            return draw.circle(self.center[0], self.center[1],
                                   self.radius)
        else:
            return draw.circle_perimeter(self.center[0], 
                                    self.center[1], self.radius)
        
class Ellipse(CenteredParticle):
    """ """

    implements(ParticleInterface)        
    ptype=Str('ellipse')    

    yradius = Int(2)
    xradius = Int(2)

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
        return scikit.draw.bezier_curve(self.ystart, self.xstart, self.ymid, 
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
    
    ycoords = Array( [1,7,1,4] )
    xcoords = Array( [1,2,8,1] )
    
    def _get_rr_cc(self):
        return draw.polygon(self.ycoords, self.xcoords)                                           

