#
# (C) Copyright 2013 George Washington University, D.C.
# All right reserved.
#
# This file is open source software distributed according to the terms in LICENSE.txt
from operator import attrgetter
import itertools
import logging

# 3rd Party Imports
import numpy as np
from traits.api import HasTraits, Instance, Str, Tuple, Float, \
    cached_property, Property, List

# Package Imports
from pyparty.shape_models.api import PARTICLETYPES
from pyparty.shape_models.abstract_shape import Particle
from pyparty.descriptors.api import CUSTOM_DESCRIPTORS, SKIMAGE_DESCRIPTORS
from pyparty.config import NAMESEP

logger = logging.getLogger(__name__) 

class ManagerError(Exception):
    """ Particle Manager general Exception """
    
class KeyErrorManager(KeyError):
    """ Particle Manager dictionary interface Exception """

# EVENTUALLY MAKE COLOR TRAIT

class MetaParticle(HasTraits):
    """ Stores a particle and metadata for use by ParticleManager """

    # TO DO: Use color trait
    name = Str()
    color = Tuple( Float(0.0), Float(0.0), Float(1.0) )
    particle = Instance(Particle)
    
    def as_tuple(self):
        return (self.name, self.color, self.particle)
    
    def as_array(self):
        return np.array(self.as_tuple)
    


class ParticleManager(HasTraits):
    """ Container class for creating, storing, managing particles;
        provides API used by higher-level objects.  Provides a property
        interface for easy mapping and slicing.  
    """
        
    _default_color = Tuple(Float(0), Float(0), Float(1) )    
    
    # CONSIDER ADDING FUNCTIONS FOR DIFFERNT COLOR STYLES LATER
    plist = List(MetaParticle)
    
    # Cached properties (requires depends_on for caching)
#    panel = Property(depends_on = 'plist')
      # Private so canvas doesn't publicize them
    _colors = Property(Tuple, depends_on = 'plist')
    _names = Property(Tuple, depends_on = 'plist')
    _namemap = Property(Tuple, depends_on = '_names')

    def _ifilter(self, attr, fcn):
        subset = self._attr_subset(attr)
        raise NotImplementedError

        
    def _subset(self, attr):
        """ Return numpy array subset of self.plist based on attr. """
        
        if attr == 'color':
            out = self._colors

        elif attr == 'name':
            out = self._names

        elif attr in CUSTOM_DESCRIPTORS:
            userfcn = CUSTOM_DESCRIPTORS[attr]
            out = tuple(userfcn(p.boxed()) for p in self.plist)

        elif attr in SKIMAGE_DESCRIPTORS:
            out = tuple(p.ski_descriptor(attr) for p in self.plist)

        else:
            try:
                out = tuple(getattr(p, attr) for p in self.plist)
            except Exception:
                raise ManagerError('%s not understood, must "color", "name", or' 
                               'valid descriptor' % attr)

        return np.array(out)
        
    @cached_property
    def _get__colors(self):
        return tuple(p.color for p in self.plist)

    @cached_property
    def _get__names(self):
        return tuple(p.name for p in self.plist)

    @cached_property
    def _get__namemap(self):
        """ Store light map of name to index for faster name lookup """
        return dict( (name, idx) for idx,name in enumerate(self._names))    
        
    
    # ADD INDEX KEYWORD TO SUPPORT INSERTIONS
    def add(self, particle, name='', idx=None, color=None,
                      *traitargs, **traitkwargs):
        """ If color not passed, default color is used
            If not idx, put in last entry  """
    
        # Make particle from name and arguments
        if isinstance(particle, basestring):
            particle = self._make_particle(particle, *traitargs, **traitkwargs)

        if not idx:
            idx = self.count    
        if not color:
            color = self._default_color
       
        #Generate key if it does not yet exist
        if not name:
            name = particle.ptype + NAMESEP + str(idx)
            
        if name in self._namemap:
            raise ManagerError('particle %s is already named "%s"' % 
                                 (self._namemap[name], name) )
        
        self.plist.insert(idx,  MetaParticle(name=name, color=color, 
                                             particle=particle) )
                
    def _make_particle(self, ptype='', *traitargs, **traitkwargs):
        """ Instantiate a particle through string specifying class type
            via PARTICLETYPES """
        try:
            pclass = PARTICLETYPES[ptype]
        except KeyError:
            raise KeyErrorManager('"%s" is not an understood Particle type.  Choose'
                ' from: [%s]' % (ptype, self.iterable_to_string(self.available())))
        return pclass(*traitargs, **traitkwargs)
    
    def clear(self):
        """ remove all particles in the dictionary. """
        self.plist[:] = [] #Does not change memory address
        
    def apply(self, fcn, *args, **kwargs):
        """ Apply a function to all particles. """
        # Useful for scaling and so on
        
    def reverse(self):
        """ In place since list is inplace """
        self.plist.reverse()
    
    def pop(self, idx):
        self.plist.pop(idx)
        
    def index(self, *names):
        """ Return index given names """
        out = tuple(self._namemap[name] for name in names)
        if len(out) == 1:
            out = out[0]
        return out
        
    # Slicing Interface (HOW HARD WOULD IT BE TO RETURN PARTICLE MANAGER CLASS!)? 
    # -----------
    def __getitem__(self, keyslice):
        """ Supports single name lookup; otherwise defers to list delitem"""

        if hasattr(keyslice, '__iter__'):
            outlist = []
            for idx in keyslice:
                if isinstance(idx, basestring):
                    idx = self._namemap[idx]
                outlist.append(self.plist[idx])
            return outlist
    

        elif isinstance(keyslice, basestring):
            idx = self._namemap[keyslice]
            return self.plist[idx]

        else: # if slice or int
            return self.plist[keyslice]
        
    def __delitem__(self, keyslice):
        """ Supports single name deletion; otherwise defers to list delitem"""

        if isinstance(keyslice, basestring):
            idx = self._namemap[keyslice]
            del self.plist[idx]

        else:
            del self.plist[keyslice]
        

    def __setitem__(self, keyorslice, particle):
        """ Not sure how confident I am in implementing this.  Users can add, 
            and remove particles through add()/remove() API.  Allowing them to
            set through slicing could break the api.  For example, setting 
            many particles to the same value and hence the same name..."""
        raise ManagerError('"%s" object does not support item assigment.' % 
                           self.__class__.__name__)

    def in_region(self, *coords):
        """ Get all particles whose centers are within a rectangular region"""
        raise NotImplementedError
    

    # python properties
    # -------------
        
    @property
    def count(self):
        return len(self.plist)    

    @property
    def idxs(self):
        """ Particle indicies"""
        return range(self.count)
    
    @property
    def particles(self):
        """ Particle objects"""
        return tuple(p.particle for p in self.plist)
    
    @property
    def centers(self):
        """ Return center coordinates of all particles."""
        return self._subset('center')
    
    

    # Full attribute container sorted mappers        
    
    def sortby(self, attr='name', inplace=False):
        """ Sort list INPLACE by descriptor: defaults to name"""

        if attr in ['name', 'color']:
            plistout = sorted(self.plist, key=attrgetter(attr))

        if inplace:
            self.plist[:] = plistout
        else:
            return ParticleManager(plist=plistout)
        
    
    def _rr_cc_color(self):
        """ Returns rr_cc, color for each particle; useful to canvas """
        return [(p[3].rr_cc, p[2]) for p in self.values]
    
    
    @property
    def ptypes(self):
        """ All unique particle types. """
        return tuple(set( self._subset(ptype) ) )
        
    @property
    def ptype_count(self):
        """ Unique particle types and _colors. """
        return ( (typ, self.ptypes.count(typ)) for typ in self.ptypes)
    
    def available(self):
        """ Show all valid particle types."""
        return tuple(sorted(PARTICLETYPES.keys()))    
        
    # Not printing these since ipython does its own printing
    @property
    def stats(self):
        """ Returns a string/TUPLE? of particles and stuff """
        return 

    @property    
    def full_stats(self):
        """ """
        return 
    
    @staticmethod
    def iterable_to_string(iterable):
        """ String formats an interable into a container string of form:
            [1, 'a', 2] -->  ("1", "a", "2") """
 
        return '"%s"' % '", "'.join(iterable)
    

    # Other representations
    # -------------
    
    #@cached_property
    #def _get_panel(self):
        #""" For complex heirarchal slicing, use panel. """
        #from pandas import Panel
        
        
        
if __name__ == '__main__':
    p=ParticleManager()
    p.count
    p.add(particle='circle', key='heythere')
    print p.plist
    
    fooparticle = PARTICLETYPES['circle'](radius=5)
    bazpart = PARTICLETYPES['circle'](radius=5)

    p.add(particle=fooparticle)
    p.add(bazpart, name='afoo')
    print p.count, p.particles
    print p._names
    print 'hiii\n'
    print p['circle_1']
    print p.sortby('name', inplace=False)._names
    p[(1,2)]
