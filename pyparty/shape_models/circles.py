#
# (C) Copyright 2013 George Washington University, D.C.
# All right reserved.
#
# This file is open source software distributed according to the terms in LICENSE.txt
#

import logging
import math

import skimage.draw
from traits.has_traits import CHECK_INTERFACES
from traits.api import HasTraits, Range, ListFloat, Property, \
     implements, Bool, Int, Array, Tuple, Str

from abstract_shape import Particle, ParticleInterface

logger = logging.getLogger(__name__) 
CHECK_INTERFACES = 2 # 2-error, 1-warn, 0-pass

class Circle(Particle):
    """ description
    
    Attributes
    ----------
    """
    # INTERFACE IS NOT WORKING
    implements(ParticleInterface)        
    
    ptype=Str('circle')
    radius = Int(12) #in pixels (<2 causes errors w/ properties)
    	    
    #http://scikit-image.org/docs/dev/api/skimage.draw.html#circle
    def _get_rr_cc(self):
        
        if self.fill:
            return skimage.draw.circle(self.center[0], self.center[1],
                                   self.radius)
        else:
            return skimage.draw.circle_perimeter(self.center[0], 
                                    self.center[1], self.radius)
                                                   

class Dimer(Particle):
    """ description
    
    Attributes
    ----------
    """
    
    ptype = 'dimer'

    overlap = Range(0.0, 1.0)
    orientation = Range(0.0, 2.0 * math.pi)
    radii = ListFloat
    
    def _get_rr_cc(self):
        raise NotImplemented
    
    
if __name__ == '__main__':
    c=Circle()
    print c.radius
        
    d=Dimer()
    print d.overlap, d.orientation, d.radii
    
    # Run pyclean
    try:
        subprocess.Popen('pyclean .', shell=True, stderr=open('/dev/null', 'w'))
    except Exception:
        pass 
