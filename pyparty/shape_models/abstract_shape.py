#
# (C) Copyright 2013 George Washington University, D.C.
# All right reserved.
#
# This file is open source software distributed according to the terms in LICENSE.txt
#

"""
SHape Models API
================

This module specifies the ...

"""
import logging

from traits.has_traits import CHECK_INTERFACES
from traits.api import Interface, implements, HasTraits, Tuple, Array, \
     Bool, Property, Str, Int, cached_property, Instance

from skimage.measure import regionprops
from skimage.measure._regionprops import _RegionProperties

from pyparty.utils import rr_cc_box

logger = logging.getLogger(__name__) 
CHECK_INTERFACES = 2    

class ParticleError(Exception):
    """ """

class ParticleInterface(Interface):
    """ Abstract class for storing particles as light objects which return
        rr and cc indicies for ndarray indexing (see skimage.draw) 
    
    Attributes
    ----------
    
    ptype : str
       Descriptor, used by ParticleManager and other classes
       to segregate out particle types.
       
    Notes
    -----
    Only traits in the contstructor (eg 'foo' vs. Str('foo') and public 
    and public methods (eg foo() vs. _foo()) will be recognized when 
    implementation is enforced.
       
    """

    ptype = Str('')
    psource = Str('')

    # IS HTIS REDUNDANT TO PUT IN HERE, AS IT IS A PROPERTY AFTERALL
    #http://scikit-image.org/docs/dev/api/skimage.draw.html#circle
    def _get_rr_cc(self):
        raise NotImplementedError
     
class Particle(HasTraits):

    implements(ParticleInterface)
    
    ptype = Str('abstract')    
    psource = Str('pyparty_builtin')
    fill = Bool(True)
    aa = Bool(False) #Anti Aliasing

    # Remove with implementation
    rr_cc = Property(Array)    
    ski_descriptor = Instance(_RegionProperties)
       
    #http://scikit-image.org/docs/dev/api/skimage.draw.html#circle
    def _get_rr_cc(self):
        raise NotImplementedError
        
    # May want this to return the translation coordinates
    def boxed(self):
        """ Returns a binary bounding box with object inside"""
        return rr_cc_box(self.rr_cc)
    
    def ski_descriptor(self, attr):
        """ Return scikit image descriptor. """
        # Set RegionProps on first call
        if not hasattr(self, '_ski_descriptor'):                     #TEST IF FASTER W/ TRUE
            self._ski_descriptor = regionprops(self.boxed(), cache=True)[0]
        return getattr(self._ski_descriptor, attr)


class CenteredParticle(Particle):
    """ Base class for particles whose centers are set by user (circle,
        elipse, etc...) as opposed to particles whose center is computed
        after the object is drawn (eg line, beziercurve, polygon)
    """
    
    implements(ParticleInterface)
    pytpe = Str('abstract_centered')
    
    # CENTER = (CX, CY)  not (CY, CX)
    center = Tuple(Int(0),Int(0)) # in pixels 
    cx = Property(Int, depends_on = 'center')
    cy = Property(Int, depends_on = 'center')    

    # Center Property Interface
    # ----------------
    def _get_cx(self):
        return self.center[0]
    
    def _get_cy(self):
        return self.center[1]
    
    def _set_cx(self, value):
        self.center = (value, self.cy)
        
    def _set_cy(self, value):
        self.center = (self.cx, value)    
    

if __name__ == '__main__':
    p=Particle()

    