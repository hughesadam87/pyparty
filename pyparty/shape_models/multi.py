import numpy as np
import skimage.draw as draw
from traits.api import Str, Int, implements, Float

from pyparty.shape_models.abstract_shape import SimplePattern, ParticleInterface
        
class Dimer(SimplePattern):
    """ Two adjacent circles around center coordinate
    
    Attributes
    ----------
    """

    implements(ParticleInterface)            
    ptype = Str('dimer')          
    _offangle = Float(0.0)
    _n = Int(2)

    def _get_rr_cc(self):
        """ Draws two circles based on position of self.skeleton. """
        
        r1, r2 = self.rs
                        
        (vx_1, vy_1), (vx_2, vy_2) = self.skeleton
                        
        rr_1, cc_1 = draw.circle(vy_1, vx_1, r1)
        rr_2, cc_2 = draw.circle(vy_2, vx_2, r2)
    
        rr = np.concatenate( (rr_1, rr_2) ) 
        cc = np.concatenate( (cc_1, cc_2) )
        return (rr, cc)
    

class Trimer(SimplePattern):
    """ Three adjacent circles around center coordinate
    
    Notes
    -----
    """
    
    implements(ParticleInterface)            
    ptype = Str('trimer')          
    _offangle = Float(30.0)
    _n = Int(3)    

    def _get_rr_cc(self):
        """ Draws two circles based on position of self.skeleton. """
        
        r1, r2, r3 = self.rs
                        
        (vx_1, vy_1), (vx_2, vy_2), (vx_3, vy_3) = self.skeleton
            
        rr_1, cc_1 = draw.circle(vy_1, vx_1, r1)
        rr_2, cc_2 = draw.circle(vy_2, vx_2, r2)
        rr_3, cc_3 = draw.circle(vy_3, vx_3, r3)
        
        rr = np.concatenate( (rr_1, rr_2, rr_3) ) 
        cc = np.concatenate( (cc_1, cc_2, cc_3) )
        return (rr, cc)


class Square(SimplePattern):
    """ Three adjacent circles around center coordinates
    
    Notes
    -----
    """
    
    implements(ParticleInterface)            
    ptype = Str('square')          
    _offangle = Float(45.0)    
    _n = Int(4)
    
    def _get_rr_cc(self):
        """ Draws two circles based on position of self.skeleton. """    
        r1, r2, r3, r4 = self.rs
                        
        (vx_1, vy_1), (vx_2, vy_2), (vx_3, vy_3), (vx_4, vy_4) = self.skeleton
            
        rr_1, cc_1 = draw.circle(vy_1, vx_1, r1)
        rr_2, cc_2 = draw.circle(vy_2, vx_2, r2)
        rr_3, cc_3 = draw.circle(vy_3, vx_3, r3)
        rr_4, cc_4 = draw.circle(vy_4, vx_4, r4)    
    
        rr = np.concatenate( (rr_1, rr_2, rr_3, rr_4) ) 
        cc = np.concatenate( (cc_1, cc_2, cc_3, cc_4) )
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
    
