#
# (C) Copyright 2013 George Washington University, D.C.
# All right reserved.
#
# This file is open source software distributed according to LICENSE.txt
#

"""
Shape Models API
================

Stores Particle models from reading from external sources
"""

import numpy as np
from traits.api import implements, Str

from pyparty.utils import IntOrNone
from abstract_shape import Particle, ParticleInterface

class LabeledParticle(Particle):
    """ Set rr_cc value from an array.  Merely changed property 
        interface of Particle to a basic attribute."""

    implements(ParticleInterface)
    ptype = Str('gen_label')     #Generic Label
    label = IntOrNone
    
    def __init__(self, rr_cc, *args, **kwargs):
        super(LabeledParticle, self).__init__(*args, **kwargs)
        self._rr_cc = rr_cc
        
    def _get_rr_cc(self):
        return self._rr_cc
    
    def _set_rr_cc(self, value):
        self._rr_cc = value
            
    
    