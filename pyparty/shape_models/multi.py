import numpy as np
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


class Trimer(SimplePattern):
    """ Three adjacent circles around center coordinate
    
    Notes
    -----
    """
    
    implements(ParticleInterface)            
    ptype = Str('trimer')          
    _offangle = Float(30.0)
    _n = Int(3)    


class Square(SimplePattern):
    """ Three adjacent circles around center coordinates
    
    Notes
    -----
    """
    
    implements(ParticleInterface)            
    ptype = Str('square')          
    _offangle = Float(45.0)    
    _n = Int(4)
    
    
if __name__ == '__main__':

    d=Dimer(radius_1 = 5, radius_2 = 3, center=(40,30))
    print d.overlap, d.orientation, d.radius_1, d.radius_2
    print d.rr_cc
    
    # Run pyclean
    try:
        subprocess.Popen('pyclean .', shell=True, stderr=open('/dev/null', 'w'))
    except Exception:
        pass 
    
