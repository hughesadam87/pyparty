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
    """ Any particles whose unrotated_rr_cc is directly passed in at __init__"""

    ptype = Str('gen_label')     #Generic Label
    label = IntOrNone
    
    # DOES THIS CHANGE/SIMPLIFY UNDER NEW _unrotated_rr_cc IMPLEMENTATION?
    def __init__(self, unrotated_rr_cc, *args, **kwargs):
        super(LabeledParticle, self).__init__(*args, **kwargs)
        self._unrotated_rr_cc = unrotated_rr_cc
        
    def _get_unrotated_rr_cc(self):
        return self._unrotated_rr_cc
    
    def _set_unrotated_rr_cc(self, value):
        self._unrotated_rr_cc = value
        
        
#class FromPath(LabeledParticle):
    
    #ptype = Str('gen_fromfile')
    #path = File('')
    
#    __init__ 
       # Get file path, read file, do stuff, somehow get_unrotated_rr_cc, then call super
       # init
        
    