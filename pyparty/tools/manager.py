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
from pyparty.shape_models.api import GROUPEDNAMES, ALLTYPES
from pyparty.shape_models.abstract_shape import Particle, ParticleError
from pyparty.descriptors.api import CUSTOM_DESCRIPTORS, SKIMAGE_DESCRIPTORS
from pyparty.config import NAMESEP

logger = logging.getLogger(__name__) 

class ManagerError(Exception):
    """ Particle Manager general Exception """
    
class KeyErrorManager(KeyError):
    """ Particle Manager dictionary interface Exception """

# EVENTUALLY MAKE COLOR TRAIT

class MetaParticle(HasTraits):
    """ Stores a particle and metadata for use by ParticleManager.
    
        Notes
        -----
        May be intelligent to store a default index for sorting and restoring
        purposes, but that would cause issue with preserving order when
        inserting (eg adding new particles.)"""

    # TO DO: Use color trait
    name = Str()
    color = Tuple( Float(0.0), Float(0.0), Float(1.0) )
    particle = Instance(Particle)
    
    @property
    def pclass(self):
        return self.particle.__class__.__name__
    
    def __repr__(self):
        """ Puts name into return with memory address:
              (my name) <__main__.Foo  at 0x3002f90>  """
        
        out = super(MetaParticle, self).__repr__() 
        address = out.split()[-1].rstrip('>')
        return '(%s : %s : %s at %s)' % \
           (self.name, self.particle.ptype, self.pclass, address)
    
    def __getattr__(self, attr):
        
        if attr in self.__dict__:
            return getattr(self, attr)
        
        elif attr in CUSTOM_DESCRIPTORS:
            return CUSTOM_DESCRIPTORS[attr](self.particle.boxed())
                          
        elif attr in SKIMAGE_DESCRIPTORS:
            return self.particle.ski_descriptor(attr)        
        
        else:
            raise ParticleError('%s could not be found on self')
    
    def __setattr__(self, attr, value):
        """ Defer attribute calls to to self.particle unless overwriting
            name, color etc... """
        
        if attr not in self.__dict__:
            setattr(self.particle, attr, value)
        else:
            self.__dict__[attr] = value
        
        
class ParticleManager(HasTraits):
    """ Container class for creating, storing, managing particles;
        provides API used by higher-level objects.  Provides a property
        interface for easy mapping and slicing.  
    """
            
    plist = List(MetaParticle)
    _namemap = Property(Tuple, depends_on = 'plist')
   
    @cached_property
    def _get__namemap(self):
        """ Store light map of name to index for faster name lookup.  I verified
            that this updates when plist elemnts are updated as well as obj itself.
            """
        return dict( (pobj.name, idx) for idx,pobj in enumerate(self.plist))    
            
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
       
        #Generate key if it does not yet exist
        if not name:
            name = particle.ptype + NAMESEP + str(idx)
            
        if name in self._namemap:
            raise ManagerError('particle %s is already named "%s"' % 
                                 (self._namemap[name], name) )
        
        if color:
            meta = MetaParticle(name=name, color=color, particle=particle)
        else:
            meta = MetaParticle(name=name, particle=particle)
        

        if idx == self.count:
            self.plist.append(meta)
        else:
            self.plist.insert(idx, meta)
                
    def _make_particle(self, ptype='', *traitargs, **traitkwargs):
        """ Instantiate a particle through string specifying class type
            via PARTICLETYPES """
        try:
            pclass = ALLTYPES[ptype]
        except KeyError:
            raise KeyErrorManager('"%s" is not an understood Particle type.  Choose'
                ' from: [%s]' % (ptype, self.iterable_to_string(self.available())))
        return pclass(*traitargs, **traitkwargs)
    
    
    def map(self, fcn, *args, **kwargs):
        """ Apply a function to all particles.  Can also use functools.partial"""
        self.plist[:] = [fcn(p, *args, **kwargs) for p in self.plist]
        
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
    
    def _unmask_index(self, idx):
        """ Test if object is an integer, string, or isntance of 
            MetaParticle.  Returns index in any case. """

        if isinstance(idx, int) or isinstance(idx, slice):
            return idx

        if isinstance(idx, basestring):
            return self._namemap[idx]

        if isinstance(idx, MetaParticle):
            return self._namemap[idx.name]
        
        raise ManagerError("Index must be of type int, str or MetaParticle"
                           " received %s" % type(idx) )
        
    # Indexing / Iterating 
    # -----------

    def __iter__(self):
        return self.plist.__iter__()
    
    def __getitem__(self, keyslice):
        """ Supports single name lookup; otherwise defers to list delitem"""

        if hasattr(keyslice, '__iter__'):
            # Boolean indexing
            if isinstance(keyslice, np.ndarray):
                boolout = keyslice.astype('bool')
                return [self.plist[idx] for idx, exists in enumerate(boolout) if exists]                
    
            else:    
                return [self.plist[self._unmask_index(idx)] for idx in keyslice]              

        return self.plist[self._unmask_index(keyslice)]

        
    def __delitem__(self, keyslice):
        """ Supports single name deletion; otherwise defers to list delitem.  
            Deletes in place to congruance with list API.
        
        Notes
        -----
        When deleting iteratively, avoid calls to "del" which changes size 
        during iteariton."""

        if hasattr(keyslice, '__iter__'):
            # Boolean indexing
            if isinstance(keyslice, np.ndarray):
                boolout = keyslice.astype('bool')
                self.plist[:] = [self.plist[idx] for idx, exists in
                                 enumerate(boolout) if not exists]
            else:    
                to_remove = [self._make_particle(idx) for idx in keyslice]
                self.plist[:] = [self.plist[idx] for idx in range(self.count)
                                 if idx not in to_remove]

        # Delete single entry
        else:
            del self.plist[self._unmask_index[keyslice]]
        

    def __setitem__(self, keyorslice, particle):
        """ Not sure how confident I am in implementing this.  Users can add, 
            and remove particles through add()/remove() API.  Allowing them to
            set through slicing could break the api.  For example, setting 
            many particles to the same value and hence the same name..."""
        raise ManagerError('"%s" object does not support item assigment.' % 
                           self.__class__.__name__)


    def __getattr__(self, attr):
        """ Return numpy array subset of self.plist based on attr. Attribute
            can be name, color, descriptor or otherwise common attribute of
            Particle class.  
            
            Notes
            -----
            Implemented similar pandas package to allow for attribute-based 
            indexing.
            
            Examples
            --------
            big_particles = p[p.perims > 50] """
        
        out = tuple(getattr(p, attr) for p in self.plist)
        return np.array(out)
    
    def __repr__(self):
       # return self.plist.__repr__()
        return super(ParticleManager, self).__repr__() + self.plist.__repr__()

    def in_region(self, *coords):
        """ Get all particles whose CENTERS are within a rectangular region"""
        raise NotImplementedError
    

    # python properties
    # -------------
        
    @property
    def count(self):
        return len(self.plist)    
    
    # SHOULD REMOVE NAMES/PARTICLES/IDXS AFTER FIX SLICING TO RETURN PMANAGER INSTANCE
    # THEN DOING P.NAME SHOULD ATOMATICALLY RETURN NAMES
    @property
    def names(self):
        """ Particles names; so common, worth doing here"""
        return tuple(p.name for p in self.plist)

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
        """ Sort list by attribute/descriptor"""

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
        """ Unique particle types in plist. """
        return ( (typ, self.ptypes.count(typ)) for typ in self.ptypes)
    
    def available(self):
        """ Show all valid particle types."""
        return GROUPEDNAMES 
        
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
    print p
    p.count
    for i in range(10,15):
        p.add(particle='circle', radius=i)

    p.add(particle='circle',name='afoo', radius=11)

    print p.name, 'hi again'

    print p.sortby('pclass').name

    #print p.name
    #print p.perimeter, type(p.perimeter), p.perimeter.dtype
    #print p[p.perimeter > 50]
    #print p.plist
    #del p[p.perimeter > 50]
    #print p.name