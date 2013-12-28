import numpy as np

from traits.api import Str, Int, Any, Range, Float, Property, implements
import skimage.draw as draw

from pyparty.patterns.elements import basic
from pyparty.shape_models.abstract_shape import CenteredParticle, ParticleInterface

class Dimer(CenteredParticle):
    """ description
    
    Attributes
    ----------
    """

    implements(ParticleInterface)            
    ptype = Str('dimer')

    radius_1 = Int(2)
    radius_2 = Any #Int or None

    overlap = Range(0.0, 1.0, value=1.0)
    orientation = Float(0.0) #In degrees

    d_pp = Property(depends_on = 'radius_1, radius_2, overlap')
    skeleton = Property(depends_on = 'cx, cy, d_pp, orientation')
    
    def _get_d_pp(self):
        if self.radius_2:
            return self.radius_1 + self.radius_2
        else:
            return 2.0 * self.radius_1
    
    
    def _get_skeleton(self):
        return basic(self.cx, self.cy, self.d_pp, n=2, phi=self.orientation)

    
    def _get_rr_cc(self):
        """ Draws two circles based on position of self.skeleton. """
        
        r1 = self.radius_1
        
        if self.radius_2:
            r2 = self.radius_2
        else:
            r2 = 2.0 * self.radius_1
            
        r1 = (1.0 * self.overlap) * r1
        r2 = (1.0 * self.overlap) * r2	
            
        (cx_1, cy_1), (cx_2, cy_2) = self.skeleton
            
        rr_1, cc_1 = draw.circle(cy_1, cx_1, r1)
        rr_2, cc_2 = draw.circle(cy_2, cx_2, r2)
    
        rr = np.concatenate( (rr_1, rr_2) ) 
        cc = np.concatenate( (cc_1, cc_2) )
        return (rr, cc)
    
if __name__ == '__main__':

    d=Dimer(radius_1 = 5, radius_2 = 3, center=(40,30))
    print d.overlap, d.orientation, d.radius_1, d.radius_2
    print d.rr_cc
    
    # Run pyclean
    try:
        subprocess.Popen('pyclean .', shell=True, stderr=open('/dev/null', 'w'))
    except Exception:
        pass 
