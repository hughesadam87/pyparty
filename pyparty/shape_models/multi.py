import numpy as np
from traits.api import Str, Int, Float
from pyparty.shape_models.abstract_shape import SimplePattern
        
class Dimer(SimplePattern):
    """ Two adjacent circles around center coordinate """

    ptype = Str('dimer')          
    _offangle = Float(0.0)
    _n = Int(2)


class Trimer(SimplePattern):
    """ Three adjacent circles around center coordinate """

    ptype = Str('trimer')          
    _offangle = Float(30.0)
    _n = Int(3)    


class Tetramer(SimplePattern):
    """ Four adjacent circles around center coordinates.  Default shape is
    square-like (but can't name class "Square" due to conflicts) """

    ptype = Str('tetramer')          
    _offangle = Float(45.0)    
    _n = Int(4)