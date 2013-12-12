#
# (C) Copyright 2013 George Washington University, D.C.
# All right reserved.
#
# This file is open source software distributed according to the terms in LICENSE.txt
#


from operator import itemgetter

# 3rd Party Imports
from traits.api import HasTraits, Dict, Instance, Str, Int, Tuple, Float

# Package Imports
from pyparty.shape_models.api import PARTICLETYPES
from pyparty.shape_models.abstract_shape import Particle

class ManagerError(Exception):
    ''' Particle Manager general Exception '''
    
class KeyErrorManager(KeyError):
    ''' Particle Manager dictionary interface Exception '''

# EVENTUALLY MAKE COLOR TRAIT

class MetaParticle(HasTraits):
    ''' Stores a particle and metadata for use by ParticleManager '''

    # TO DO: Use color trait
    index = Int()
    name = Str()
    color = Tuple( Float(0.0), Float(0.0), Float(1.0) )
    particle = Instance(Particle)
    
    def as_tuple(self):
        return (self.index, self.name, self.color, self.particle)


class ParticleManager(HasTraits):
    """ Container class for creating, storing, managing particles;
        provides API used by higher-level objects.  Provides a property
        interface for easy mapping and slicing.  
    """
        
    _default_color = Tuple(Float(0), Float(0), Float(1) )    
    
    # CONSIDER ADDING FUNCTIONS FOR DIFFERNT COLOR STYLES LATER
    pdict = Dict(Str,
                   Tuple(Int, Str, _default_color, Instance(Particle))
                  )
    
    def add_particle(self, particle, key='', color=None,
                     *traitargs, **traitkwargs):
        ''' If color not passed, default color is used '''
        
        if isinstance(particle, basestring):
            particle = self._make_particle(particle, *traitargs, **traitkwargs)

        idx = self.count    
        if not color:
            color = self._default_color
       
        #Generate key if it does not yet exist
        if not key:
            key = particle.ptype + '_' + str(idx)
        
        self.pdict[key] = (idx, key, color, particle)      
                
    def _make_particle(self, ptype='', *traitargs, **traitkwargs):
        ''' Instantiate a particle through string specifying class type
            via PARTICLETYPES '''
        try:
            pclass = PARTICLETYPES[ptype]
        except KeyError:
            raise KeyErrorManager('"%s" is not an understood Particle type.  Choose'
                ' from: [%s]' % (ptype, self.iterable_to_string(self.available())))
        return pclass(*traitargs, **traitkwargs)
    
    def clear(self):
        ''' remove all particles in the dictionary. '''
        self.pdict.clear()
        

    # Slicing Interface (HOW HARD WOULD IT BE TO RETURN PARTICLE MANAGER CLASS!)? 
    # -----------
    def __getitem__(self, keyslice):

        if isinstance(keyslice, slice):
            print 'ITS A SLICE'
        
        try:
            return self.pdict[keyslice]
        except KeyError as KE:
            raise KeyErrorManager(KE.message)
        
    def __delitem__(self, key):
        try:
            del self.pdict[key]
        except KeyError as KE:
            raise KeyErrorManager(KE.message)
        
    # Really need to think about how to do this pragmatically
#    def __setitem__(self, keyorslice, particle):
#        self.add_particle(particle, key=key)
    
    def in_region(self, *coords):
        ''' Get all particles whose centers are within a rectangular region'''
        raise NotImplemented
    
    def _centers(self):
        ''' Return center coordinates of all particles.'''
    
    # Properties
    # ------
    
    @property
    def values(self):
        return tuple(self.pdict.values())
    
    @property
    def count(self):
        return len(self.pdict)    
    
    # Single attribute mappers
    @property
    def names(self):
        return self.pdict.keys()  

    @property
    def idxs(self):
        ''' Tuple of Particle indicies'''
        return tuple(map(itemgetter(0), self.values))  
    
    @property
    def colors(self):
        ''' Tuple of Particle objects'''
        return tuple(map(itemgetter(2), self.values))    
    
    @property
    def particles(self):
        ''' Tuple of Particle objects'''
        return tuple(map(itemgetter(3), self.values))
    

    # Full attribute container sorted mappers        
    def by_idx(self):
        return sorted(self.values, key=itemgetter(0))

    def by_name(self):
        return sorted(self.values, key=itemgetter(1))   
    
    def by_color(self):
        return sorted(self.values, key=itemgetter(2))   
    
    def rr_cc_color(self):
        ''' Returns rr_cc, color for each particle. '''
        
        # THIS WONT BE SORTED.  MAYBE REMOVE DICTIONARY IMPLEMENTATION, 
        # AND DO TUPLES(TUPLES) SO THAT SORT ORDER IS PRESERVED AND IDX
        # CAN BE DROPPED, THEN HAVE A MAP AVAILABLE AS AN ALTERNATE REPR.
        # ALTERNATIVELY, VALUES CAN BE "SORTED" AT PROPERTY CALL RIGHT!?
        return [(p[3].rr_cc, p[2]) for p in self.values]
         # self.particles is not ordered by index               
    
    
    @property
    def ptypes(self):
        ''' All unique particle types. '''
        return tuple(set( self._ptype_list() ) )
        
    @property
    def ptype_count(self):
        ''' Unique particle types and colors. '''
        return ( (typ, self.ptypes.count(typ)) for typ in self.ptypes)
        
    def _ptype_list(self):
        ''' Shared by a few methods. '''
        return [obj.ptype for obj in self.particles]
    
    def available(self):
        ''' Show all valid particle types.'''
        return tuple(sorted(PARTICLETYPES.keys()))    
        
    # Not printing these since ipython does its own printing
    @property
    def stats(self):
        ''' Returns a string/TUPLE? of particles and stuff '''
        return 

    @property    
    def full_stats(self):
        ''' '''
        return 
    
    @staticmethod
    def iterable_to_string(iterable):
        ''' String formats an interable into a container string of form:
            [1, 'a', 2] -->  ("1", "a", "2") '''
 
        return '"%s"' % '", "'.join(iterable)
        
if __name__ == '__main__':
    p=ParticleManager()
    p.count
    p.add_particle(particle='circle', key='heythere')
    print p.pdict
    
    fooparticle = PARTICLETYPES['circle'](radius=5)
    p.add_particle(particle=fooparticle)
    print p.count, p.particles, p.by_idx()
    print p['abstract_1']
    print p[1:'foo']