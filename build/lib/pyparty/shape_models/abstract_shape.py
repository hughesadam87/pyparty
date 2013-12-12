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

from traits.has_traits import CHECK_INTERFACES
from traits.api import Interface, implements, HasTraits, Tuple, Array, \
     Bool, Property, Str

CHECK_INTERFACES = 2    


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
 
    
class Particle(HasTraits):

    implements(ParticleInterface)
    
    ptype = Str('abstract')    
    fill = Bool(True)

    # Remove with implementation
    rr_cc = Property(Array)
    center = Tuple(Int(0),Int(0)) # in pixels 
    cx = Property(Int, depends_on = 'center')
    cy = Property(Int, depends_on = 'center')

    
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
        
        

if __name__ == '__main__':
    p=Particle()

    