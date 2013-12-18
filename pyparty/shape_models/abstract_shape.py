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
    psource = Str('unkown')
    fill = Bool(True)

    # Remove with implementation
    rr_cc = Property(Array)
    center = Tuple(Int(0),Int(0)) # in pixels 
    cx = Property(Int, depends_on = 'center')
    cy = Property(Int, depends_on = 'center')
    
    ski_descriptor = Instance(_RegionProperties)
       
    #http://scikit-image.org/docs/dev/api/skimage.draw.html#circle
    def _get_rr_cc(self):
        raise NotImplementedError
    
    def center_to_image(self, image):
        ''' Set the center of particle to middle of image (2darray) '''
        self.center = (image.shape[0] / 2, image.shape[1] / 2)
        
    # Trait Property Interface
    # ----------------
    def _get_cx(self):
        return self.center[0]
    
    def _get_cy(self):
        return self.center[1]
    
    def _set_cx(self, value):
        cx, cy = self.center
        self.center = (value, cy)
        
    def _set_cy(self, value):
        cx, cy = self.center
        self.center = (value, cx)
        
    # May want this to return the translation coordinates
    def boxed(self):
        """ Returns a binary bounding box with object inside"""
        return rr_cc_box(self.rr_cc)
    
    def ski_descriptor(self, attr):
        """ Return scikit image descriptor. """
        # Set RegionProps on first call
        if not self.ski_descriptor:                     #TEST IF FASTER W/ TRUE
            self.ski_descriptor = measure(regionprops(self.boxed(), cache=False))
        return getattr(self.ski_descriptor, attr)
    
        

if __name__ == '__main__':
    p=Particle()

    