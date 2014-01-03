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
import logging
import numpy as np
from traits.api import Str, File

from pyparty.trait_types.intornone import IntOrNone
from abstract_shape import Particle

logger = logging.getLogger(__name__) 

class LabeledParticle(Particle):
    """ Any particles whose rr_cc is directly passed in at __init__"""

    ptype = Str('gen_label')     #Generic Label
    label = IntOrNone
    
    # DOES THIS CHANGE/SIMPLIFY UNDER NEW _RR_CC IMPLEMENTATION?
    def __init__(self, rr_cc, *args, **kwargs):
        super(LabeledParticle, self).__init__(*args, **kwargs)
        self._rr_cc = rr_cc
        
    def _get_rr_cc(self):
        return self._rr_cc
    
    def _set_rr_cc(self, value):
        self._rr_cc = value
        
        
#class FromPath(LabeledParticle):
    
    #ptype = Str('gen_fromfile')
    #path = File('')
    
#    __init__ 
       # Get file path, read file, do stuff, somehow get_rr_cc, then call super
       # init
        
    